#include "polyscope_app.h"

// libigl
#include <igl/readDMAT.h>
#include <igl/boundary_facets.h>
#include <igl/IO>
#include <igl/AABB.h>
#include <igl/in_element.h>
#include <igl/barycentric_coordinates.h>

#include "boundary_conditions.h"
#include <sstream>
#include <fstream>
#include <functional>
#include <string>

using namespace Eigen;
using namespace mfem;

// ./bin/hetero_tet_sim ../models/beam_bone_V.dmat ../models/beam_bone_T.dmat --i=../models/bone_tets.dmat --i=../models/muscle_tets.dmat --y=1e12 --y=1e6
// ./bin/hetero_tet_sim ../models/mesh/simple_joint/tet_mesh_V.dmat ../models/mesh/simple_joint/tet_mesh_T.dmat --i=../models/mesh/simple_joint/bottom_bone_bone_indices.dmat --i=../models/mesh/simple_joint/top_bone_bone_indices.dmat --i=../models/mesh/simple_joint/muscle_muscle_indices.dmat --y=1e12 --y=1e12 --y=1e6
// ./bin/hetero_tet_sim ../models/mesh/biceps/tet_mesh_V.dmat ../models/mesh/biceps/tet_mesh_T.dmat --i=../models/mesh/biceps/scapula_bone_indices.dmat --i=../models/mesh/biceps/humerus_bone_indices.dmat --i=../models/mesh/biceps/forearm_bone_indices.dmat --i=../models/mesh/biceps/biceps_muscle_indices.dmat --y=1e12 --y=1e12 --y=1e12 --y=1e7
struct PolyscopeTetApp : public PolyscopeApp<3> {
  
  void simulation_step() override {
    optimizer->step();
    meshV = mesh->vertices();
    for (size_t i = 0; i < srfs.size(); ++i) {
      srfs[i]->updateVertexPositions(meshV);
    }
  }

  void reset() override {
    optimizer->reset();
    for (size_t i = 0; i < srfs.size(); ++i) {
      srfs[i]->updateVertexPositions(mesh->V0_);
    }
  }

  void write_obj(int step) override {

    for (size_t i = 0; i < trimeshes.size(); ++i) {
      char buffer [50];
      int n = sprintf(buffer, "../output/obj/tet_%d_%04d.obj", i, step); 
      buffer[n] = 0;
      igl::writeOBJ(std::string(buffer),meshV,trimeshes[i]);
    }
  }


  void init(const MatrixXd& V, const MatrixXi& T,
      const std::vector<VectorXi>& indices,
      const std::vector<double>& youngs_moduli) {
    // Read the mesh
    meshV = V;
    meshT = T;
    double fac = meshV.maxCoeff();
    meshV.array() /= fac;
    std::cout << "fac: " << fac << std::endl;

    // Register the mesh with Polyscope
    polyscope::options::autocenterStructures = false;
    if (meshF.size() == 0){ 
      igl::boundary_facets(meshT, meshF);
    }

    trimeshes.clear();    
    for (size_t i = 0; i < indices.size(); ++i) {
      MatrixXi Ti = T(indices[i], Eigen::placeholders::all);
      std::string name = "tet_mesh_" + std::to_string(i);
      MatrixXi Fi;
      igl::boundary_facets(Ti, Fi);
      srfs.push_back(polyscope::registerSurfaceMesh(name, meshV, Fi));
      trimeshes.push_back(Fi);
    }

    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

    meshV0 = meshV;

    // Initial simulation setup
    config = std::make_shared<SimConfig>();

    std::vector<std::shared_ptr<MaterialModel>> material_models;

    for (size_t i = 0; i < indices.size(); ++i) {
      MaterialConfig cfg;
      Enu_to_lame(youngs_moduli[i], 0.45, cfg.la, cfg.mu);
      material_models.push_back(material_factory.create(cfg.material_model,
          std::make_shared<MaterialConfig>(cfg))
      );

    }


    material_config = std::make_shared<MaterialConfig>();
    material = material_factory.create(material_config->material_model,
        material_config);
    config->kappa = material_config->mu;


    mesh = std::make_shared<TetrahedralMesh>(meshV, meshT,
        material_config, indices, material_models);

    optimizer = optimizer_factory.create(config->optimizer, mesh, config);
    optimizer->reset();

    BoundaryConditions<3>::get_script_names(bc_list);
  }
  std::vector<polyscope::SurfaceMesh*> srfs;
  std::vector<MatrixXi> trimeshes;

} app;


int main(int argc, char **argv) {

  // omp_set_num_threads(8);
  // Configure the argument parser
  args::ArgumentParser parser("Mixed FEM");
  args::Positional<std::string> inV(parser, "<rest>.dmat", "rest vertices");
  args::Positional<std::string> inT(parser, "<rest>.dmat", "rest tetrahedra");
  args::ValueFlagList<std::string> indexPaths(parser, "indexes", "indices into the tet mesh",{"i"});
  args::ValueFlagList<double> ymlist(parser, "ym", "young modulus for each index set",{"y"});

  // Parse args
  std::vector<std::string> index_files;
  std::vector<double> youngs_moduli;

  try {
    parser.ParseCLI(argc, argv);

    std::cout << "Paths: " << std::endl;
    for (auto&& path : indexPaths) {
        std::cout << ' ' << path << std::endl;
        index_files.push_back(path);
    }
    
    for (double ym : ymlist) {
      youngs_moduli.push_back(ym);
      std::cout << " YM: " << ym << std::endl;
    }
    assert(youngs_moduli.size() == index_files.size());
    
  } catch (args::Help) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError e) {
    std::cerr << e.what() << std::endl;

    std::cerr << parser;
    return 1;
  }

  // Options
  polyscope::options::autocenterStructures = true;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;

  // Initialize polyscope
  polyscope::init();

  std::string Vfn = args::get(inV);
  std::string Tfn = args::get(inT);
  std::cout << "loading: " << Vfn << std::endl;
  MatrixXd V;
  igl::readDMAT(Vfn, V);
  std::cout << "V.rows() : " << V.rows() << ", " << V.cols() << std::endl;

  MatrixXi T;
  igl::readDMAT(Tfn, T);
  std::cout << "T.rows() : " << T.rows() << ", " << T.cols() << std::endl;

  std::vector<VectorXi> Tids;
  for (size_t i = 0; i < index_files.size(); ++i) {
    VectorXi T;
    igl::readDMAT(index_files[i], T);
    Tids.push_back(T);
  }
  app.init(V,T,Tids,youngs_moduli);

  // Add the callback
  polyscope::state::userCallback = std::bind(&PolyscopeApp<3>::callback, app);

  // Show the gui
  polyscope::show();

  return 0;
}

