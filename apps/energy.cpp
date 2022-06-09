#include "polyscope/polyscope.h"

// libigl
#include <igl/boundary_facets.h>
#include <igl/IO>

// Polyscope
#include "args/args.hxx"
#include "json/json.hpp"

#include "mesh/mesh.h"
#include "energies/material_model.h"
#include "energies/stable_neohookean.h"

#include "optimizers/mixed_alm_optimizer.h"
#include "optimizers/mixed_admm_optimizer.h"
#include "optimizers/mixed_sqp_optimizer.h"
#include "optimizers/mixed_sqp_pd_optimizer.h"
#include "optimizers/newton_optimizer.h"
#include "boundary_conditions.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;


using namespace Eigen;

// The mesh, Eigen representation
MatrixXd meshV, meshV0;
MatrixXi meshF;
MatrixXi meshT;

using namespace mfem;
std::shared_ptr<SimConfig> config;
std::shared_ptr<MaterialModel> material;
std::shared_ptr<MaterialConfig> material_config;
std::shared_ptr<Mesh> tet_object;
std::shared_ptr<NewtonOptimizer> optimizer;

std::vector<std::string> bc_list;

void simulation_step() {
  optimizer->step();
  meshV = tet_object->vertices();
}

// ./bin/energy -p /media/ty/ECB0AB91B0AB60B6/blendering/Gecko_1e8/fem_data -n 100 -r ../models/gecko.mesh
int main(int argc, char **argv) {
  // Configure the argument parser
  args::ArgumentParser parser("Mixed FEM", "Example: ./bin/energy -r ../models/coarse_bunny.mesh  \
      --path ../output/ -n 100 -ym 1e5");
  args::ValueFlag<std::string> rest_arg(parser, "<file_name>.mesh", "Rest state mesh", {'r', "rest"});
  args::ValueFlag<std::string> path_arg(parser, "/path/to/ <sim_x0_step> files", "Path to x0 and v data", {'p', "path"});
  args::ValueFlag<int> n_arg(parser, "integer", "number of steps", {'n'});
  args::ValueFlag<double> ym_arg(parser,"double", "Youngs modulus", {"ym"});

  // Parse args
  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }
  std::string rest_fn = args::get(rest_arg);
  std::string path_fn = args::get(path_arg);
  double ym = args::get(ym_arg);
  int n = args::get(n_arg);
  std::cout << "Path: " << path_fn << std::endl;
  std::cout << "n: " << n << std::endl;
  std::cout << "YM: " << ym << std::endl;
  fs::path dir(path_fn);

    // Read the mesh
  igl::readMESH(rest_fn, meshV, meshT, meshF);
  double fac = meshV.maxCoeff();
  meshV.array() /= fac;

  if (meshF.size() == 0){ 
    igl::boundary_facets(meshT, meshF);
  }
  std::cout << "V : " << meshV.rows() << " T: " << meshT.rows() << std::endl;

  // Initial simulation setup
  config = std::make_shared<SimConfig>();
  config->bc_type = BC_NULL;

  material_config = std::make_shared<MaterialConfig>();
  Enu_to_lame(ym, material_config->pr,
      material_config->la, material_config->mu);
  material = std::make_shared<StableNeohookean>(material_config);
  tet_object = std::make_shared<TetrahedralMesh>(meshV, meshT,
      material, material_config);
  optimizer = std::make_shared<NewtonOptimizer>(tet_object,config);
  optimizer->reset();


  for (int i = 1; i <= n; ++i) {
    char buffer[50];
    int n = sprintf(buffer, "sim_v_%04d.dmat", i); 
    buffer[n] = 0;
    std::string v_fn(buffer);

    n = sprintf(buffer, "sim_x0_%04d.dmat", i); 
    buffer[n] = 0;
    std::string x_fn(buffer);
    //     fs::path dir ("/tmp");
    fs::path v_file(v_fn);
    fs::path x_file(x_fn);
    fs::path x_full_path = dir / x_file;
    fs::path v_full_path = dir / v_file;

    // std::cout << "x_full_path" << x_full_path << std::endl;

    VectorXd v;
    igl::readDMAT(std::string(x_full_path), v);
    // std::cout << x_full_path << std::endl;
    double KE = v.transpose() * optimizer->M_ * v;
    std::cout << KE << std::endl;
  }

  // optimizer->x0_ = x0;
  // optimizer->vt_ = v;
  // int nsteps = x.cols();
  // for (int i = 0; i < nsteps; ++i) {
  //   optimizer->x_ = optimizer->P_ * x.col(i);
  //   optimizer->b_ = x.col(i) - optimizer->P_ .transpose()*optimizer->P_*x.col(i);
  //   optimizer->build_lhs();
  //   optimizer->build_rhs();

  //   // Compute search direction
  //   double decrement;
  //   optimizer->substep(i, decrement);
  //   std::cout << decrement << std::endl;
  // }

  return 0;
}

