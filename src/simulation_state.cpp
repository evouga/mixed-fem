#include "simulation_state.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include "config.h"

#include "mesh/mesh.h"
#include "mesh/tri2d_mesh.h"
#include "mesh/meshes.h"

// Factories
#include "factories/solver_factory.h"
#include "factories/variable_factory.h"
#include "factories/optimizer_factory.h"
#include "factories/integrator_factory.h"
#include "factories/material_model_factory.h"

// libigl
#include <igl/IO>
#include <igl/remove_unreferenced.h>

using json = nlohmann::json;
using namespace mfem;
using namespace Eigen;

namespace {

  template<typename T>
  void read_and_assign(const nlohmann::json& args, const std::string& key,
      T& value) {
    if (const auto& it = args.find(key); it != args.end()) {
      value = it->get<T>();
    }
  }
  
}

// Read 2D Mesh
// TODO separate 3D or maybe even templated loading
template <int DIM>
void SimState<DIM>::load_mesh(const std::string& path, MatrixXd& V, MatrixXi& T) {
  if constexpr (DIM == 2) {
    MatrixXd NV;
    MatrixXi NT;
    igl::read_triangle_mesh(path,V,T);
    VectorXi VI,VJ;
    igl::remove_unreferenced(V,T,NV,NT,VI,VJ);
    V = NV;
    T = NT;

    // Truncate z data
    MatrixXd tmp;
    tmp.resize(V.rows(),2);
    tmp.col(0) = V.col(0);
    tmp.col(1) = V.col(1);
    V = tmp;
  } else {
    MatrixXd F;
    igl::readMESH(path, V, T, F);
  }
}


template <int DIM>
bool SimState<DIM>::load(const std::string& json_file) {
  // Confirm file is .json
  if (std::filesystem::path(json_file).extension() != ".json") {
    std::cerr << "File: " << json_file << " needs to be json" << std::endl;
    return false;
  }

  // Read and parse file
  json args;
  std::ifstream input(json_file);
  if (input.good()) {
    args = json::parse(input);
  } else {
    std::cerr << "Unable to open file: " << json_file << std::endl;
    return false;
  }

  config_ = std::make_shared<SimConfig>();
  load_params(args);

  OptimizerFactory<DIM> optimizer_factory;
  if (const auto& it = args.find("optimizer"); it != args.end()) {
    std::string name = it->get<std::string>();
    config_->optimizer = optimizer_factory.type_by_name(name);
  }

  IntegratorFactory integrator_factory;
  if (const auto& it = args.find("time_integrator"); it != args.end()) {
    std::string name = it->get<std::string>();
    config_->ti_type = integrator_factory.type_by_name(name);
  }

  MaterialModelFactory material_factory;
  std::vector<std::shared_ptr<MaterialModel>> materials;
  std::vector<std::shared_ptr<MaterialConfig>> mat_configs;
  if (const auto& obj_it = args.find("material_models"); obj_it != args.end())
  {
    for (const auto& obj : *obj_it) {
      std::shared_ptr<MaterialConfig> cfg = std::make_shared<MaterialConfig>();

      read_and_assign(obj, "youngs_modulus", cfg->ym);
      read_and_assign(obj, "poissons_ratio", cfg->pr);
      read_and_assign(obj, "density", cfg->density);

      if (const auto& it = obj.find("energy"); it != obj.end()) {
        std::string name = it->get<std::string>();
        cfg->material_model = material_factory.type_by_name(name);
      }
      Enu_to_lame(cfg->ym, cfg->pr, cfg->la, cfg->mu);
      mat_configs.push_back(cfg);
      materials.push_back(material_factory.create(cfg->material_model, cfg));
    }
  } else {
    mat_configs = {std::make_shared<MaterialConfig>()};
    materials = {material_factory.create(
        mat_configs[0]->material_model, mat_configs[0])};
  }
  material_models_ = materials;

  // TODO mesh factory <DIM>
  std::vector<std::shared_ptr<Mesh>> meshes;
  if (const auto& obj_it = args.find("objects"); obj_it != args.end()) {
    for (const auto& obj : *obj_it) {

      std::string path;
      std::vector<double> offset = {0.0, 0.0, 0.0};
      uint idx = 0;

      // Get File path
      if (const auto& it = obj.find("path"); it != obj.end()) {
        path = it->get<std::string>();
      } else {
        std::cerr << "Object missing path!" << std::endl;
        return false;
      }

      if (const auto& it = obj.find("offset"); it != obj.end()) {
        offset = it->get<std::vector<double>>();
        assert(offset.size() == 3);
      }

      if (const auto& it = obj.find("material_index"); it != obj.end()) {
        idx = it->get<uint>();
        assert(idx < mat_configs.size());
      }

      MatrixXd V;
      MatrixXi T;
      load_mesh(path, V, T);
      for (int i = 0; i < V.cols(); ++i) {
        V.col(i).array() += offset[i];
      }
      if constexpr (DIM == 2) {
        meshes.push_back(std::make_shared<Tri2DMesh>(V, T, materials[idx]));
      } else {
        meshes.push_back(std::make_shared<TetrahedralMesh>(V, T, materials[idx]));
      }
    }
  }
  mesh_ = std::make_shared<Meshes>(meshes);
  x_ = std::make_unique<Displacement<DIM>>(mesh_, config_);

  MixedVariableFactory<DIM> mixed_variable_factory;
  std::set<VariableType> mixed_variables;
  if (const auto& it = args.find("mixed_variables"); it != args.end()) {
    for(const auto& name : it->get<std::vector<std::string>>()) {
      mixed_variables.insert(mixed_variable_factory.type_by_name(name));
    }
  }
  config_->mixed_variables = mixed_variables;
  for (VariableType type : config_->mixed_variables) {
    mixed_vars_.push_back(mixed_variable_factory.create(type, mesh_, config_));
  }

  VariableFactory<DIM> variable_factory;
  std::set<VariableType> variables;
  if (const auto& it = args.find("variables"); it != args.end()) {
    for(const auto& name : it->get<std::vector<std::string>>()) {
      variables.insert(variable_factory.type_by_name(name));
    }
  }
  config_->variables = variables;
  for (VariableType type : config_->variables) {
    vars_.push_back(variable_factory.create(type, mesh_, config_));
  }

  SolverFactory solver_factory;
  if (const auto& it = args.find("linear_solver"); it != args.end()) {
    std::string name = it->get<std::string>();
    config_->solver_type = solver_factory.type_by_name(name);
  }
  solver_ = solver_factory.create(config_->solver_type, mesh_, config_);

  if (const auto& it = args.find("boundary_condition"); it != args.end()) {
    std::vector<std::string> bc_list;
    BoundaryConditions<DIM>::get_script_names(bc_list);
    std::string name = it->get<std::string>();
    for (size_t i = 0; i < bc_list.size(); ++i) {
      if (name == bc_list[i]) {
        config_->bc_type = static_cast<BCScriptType>(i);
        break;
      }
    }
  }

  return true;
}

template <int DIM>
void SimState<DIM>::load_params(const nlohmann::json& args) {


  if (const auto& it = args.find("body_force"); it != args.end()) {
    std::vector<float> ext = it->get<std::vector<float>>();
    assert(ext.size() == 3);
    config_->ext[0] = ext[0];
    config_->ext[1] = ext[1];
    config_->ext[2] = ext[2];
  }

  read_and_assign(args, "dt", config_->h);
  read_and_assign(args, "print_timing", config_->show_timing);
  read_and_assign(args, "print_stats", config_->show_data);
  read_and_assign(args, "enable_ccd", config_->enable_ccd);
  read_and_assign(args, "dhat", config_->dhat);
  read_and_assign(args, "kappa", config_->kappa);
  read_and_assign(args, "max_newton_iterations", config_->outer_steps);
  read_and_assign(args, "max_linesearch_iterations", config_->ls_iters);
}

template class mfem::SimState<3>; // 3D
template class mfem::SimState<2>; // 2D
