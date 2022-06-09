#include "polyscope/polyscope.h"

// libigl
#include <igl/boundary_facets.h>
#include <igl/IO>

// Polyscope
#include "args/args.hxx"
#include "json/json.hpp"

#include "mesh/mesh.h"
#include "energies/material_model.h"
#include "optimizers/mixed_alm_optimizer.h"
#include "optimizers/mixed_admm_optimizer.h"
#include "optimizers/mixed_sqp_optimizer.h"
#include "optimizers/mixed_sqp_pd_optimizer.h"
#include "optimizers/newton_optimizer.h"
#include "boundary_conditions.h"
#include "energies/stable_neohookean.h"
#include <sstream>
#include <fstream>

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

int main(int argc, char **argv) {
  // Configure the argument parser
  args::ArgumentParser parser("Mixed FEM", "Example: ./bin/decrement -r ../models/coarse_bunny.mesh \
      -x ../output/sim_x_0017.dmat \
      --x0 ../output/sim_x0_0017.dmat \
       -v ../output/sim_v_0017.dmat --ym 1e5");
  args::ValueFlag<std::string> rest_arg(parser, "<file_name>.mesh", "Rest state mesh", {'r', "rest"});
  args::ValueFlag<std::string> x_arg(parser, "sim_x_<step>.dmat", "x values for step", {'x'});
  args::ValueFlag<std::string> x0_arg(parser, "sim_x0_<step>.dmat", "x0 value for step", {"x0"});
  args::ValueFlag<std::string> v_arg(parser, "sim_v_<step>.dmat", "v value for step", {'v'});
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
  // x_, x0_, v_b_(?) for now dont and just assume pinning constraint that is same
  std::string rest_fn = args::get(rest_arg);
  std::string x_fn = args::get(x_arg);
  std::string x0_fn = args::get(x0_arg);
  std::string v_fn = args::get(v_arg);
  double ym = args::get(ym_arg);
  std::cout << "Rest mesh: " << rest_fn << std::endl;

  // Read the mesh
  igl::readMESH(rest_fn, meshV, meshT, meshF);
  double fac = meshV.maxCoeff();
  meshV.array() /= fac;

  if (meshF.size() == 0){ 
    igl::boundary_facets(meshT, meshF);
  }
  std::cout << "V : " << meshV.rows() << " T: " << meshT.rows() << std::endl;
  meshV0 = meshV;

  MatrixXd x;
  VectorXd x0, v;
  igl::readDMAT(x_fn, x);
  igl::readDMAT(x0_fn, x0);
  igl::readDMAT(v_fn, v);

  std::cout  << "x: " << x.rows() << " " << x.cols() << std::endl;
  std::cout << "x0: " << x0.rows() << " " << x0.cols() << std::endl;
  std::cout << "v: " << v.rows() << " " << v.cols() << std::endl;


  // Initial simulation setup
  // TODO! need to have to code to serialize/deserial config
  config = std::make_shared<SimConfig>();
  config->bc_type = BC_HANGENDS;

  material_config = std::make_shared<MaterialConfig>();
  Enu_to_lame(ym, material_config->pr,
      material_config->la, material_config->mu);
  material = std::make_shared<StableNeohookean>(material_config);
  tet_object = std::make_shared<TetrahedralMesh>(meshV, meshT,
      material, material_config);
  optimizer = std::make_shared<NewtonOptimizer>(tet_object,config);
  optimizer->reset();

  optimizer->x0_ = x0;
  optimizer->vt_ = v;
  int nsteps = x.cols();
  for (int i = 0; i < nsteps; ++i) {
      optimizer->x_ = optimizer->P_ * x.col(i);
      optimizer->b_ = x.col(i) - optimizer->P_ .transpose()*optimizer->P_*x.col(i);
      optimizer->build_lhs();
      optimizer->build_rhs();

      // Compute search direction
      double decrement;
      optimizer->substep(i, decrement);
      std::cout << decrement << std::endl;
    }

  return 0;
}

