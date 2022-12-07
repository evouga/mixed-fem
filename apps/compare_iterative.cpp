
// libigl
#include <igl/boundary_facets.h>
#include <igl/IO>

#include "args/args.hxx"
#include "json/json.hpp"

#include "factories/optimizer_factory.h"
#include "config.h"

using namespace Eigen;
using namespace mfem;

const int DIM = 2;

std::shared_ptr<Optimizer<DIM>> optimizer;
std::shared_ptr<Optimizer<DIM>> newton_optimizer;
std::shared_ptr<Mesh> mesh;

double min_gradient;
std::vector<double> curr_gradients;

static double newton_gradient(const SimState<DIM>& state) {
  // Get full configuration vector
  VectorXd x = state.x_->value();
  state.x_->unproject(x);

  const SimState<DIM>& newton_state = newton_optimizer->state();

  newton_state.x_->value() = state.x_->value();
  VectorXd rhs = newton_state.x_->rhs();

  for (auto& var : newton_state.vars_) {
    var->update(x, state.x_->integrator()->dt());
    rhs += var->rhs();
  }
  return rhs.norm();
}

void callback(const SimState<DIM>& state) {
  double grad = newton_gradient(state);
  min_gradient = grad;
  curr_gradients.push_back(grad);
  std::cout << "gradient: " << grad << std::endl;
}

int main(int argc, char **argv) {
  // Configure the argument parser
  args::ArgumentParser parser("Mixed FEM");
  args::Positional<std::string> inFile(parser, "json", "input scene json file");

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

  std::string filename = args::get(inFile);

  // Read and parse file
  std::ifstream input(filename);
  if (!input.good()) {
    std::cerr << "Unable to open file: " << filename << std::endl;
    return 0;
  }
  nlohmann::json args = nlohmann::json::parse(input);
  OptimizerFactory<DIM> factory;

  std::vector<LinearSolverType> types = {
    //LinearSolverType::SOLVER_EIGEN_CG_IC, // Primal condensation 
    //LinearSolverType::SOLVER_MINRES_ID, // Indefinite
    //LinearSolverType::SOLVER_AMGCL//,  // Dual condensation
    LinearSolverType::SOLVER_SUBSPACE
    //LinearSolverType::SOLVER_ADMM
  };
  int N = types.size();
  int M = args["max_newton_iterations"];


  VectorXd min_gradients(N);
  MatrixXd gradients(N, M);
  std::cout << "M: " << M << std::endl;

  for (int i = 0; i < N; ++i) {
    // Read and parse file
    std::ifstream input(filename);
    nlohmann::json args = nlohmann::json::parse(input);

    curr_gradients.clear();
    min_gradient = std::numeric_limits<double>::max();
    //tols(i) = std::pow(10,exps(i));
    double tol = 0.00000001;

    SimState<DIM> state;
    state.load(args);
    optimizer = factory.create(state.config_->optimizer, state);
    optimizer->state().config_->itr_tol = tol;//tols(i);
    optimizer->state().config_->max_iterative_solver_iters = 1e5;
    optimizer->state().config_->solver_type = types[i];
    optimizer->reset();
    optimizer->callback = callback;
    
    // Switch to newton FEM
    // Note just supports stretch right now
    args.erase("mixed_variables");
    args["variables"] = {"stretch"};

    SimState<DIM> newton_state;
    newton_state.load(args);
    newton_optimizer = factory.create(newton_state.config_->optimizer, newton_state);
    newton_optimizer->reset();

    optimizer->step();
    min_gradients(i) = min_gradient;
    gradients.row(i) = Map<RowVectorXd>(curr_gradients.data(), M);
  }
  std::cout << "min_gradients: " << min_gradients << std::endl;
  igl::writeDMAT("../output/convergence.dmat", gradients);
  igl::writeDMAT("../output/finalconvergence_converge.dmat", min_gradients);
  //igl::writeDMAT("../output/ym_values.dmat", tols);
  return 0;
}

