
// libigl
#include <igl/boundary_facets.h>
#include <igl/IO>

#include "args/args.hxx"
#include "json/json.hpp"

#include "factories/optimizer_factory.h"

using namespace Eigen;
using namespace mfem;

const int DIM = 2;

std::shared_ptr<Optimizer<DIM>> optimizer;
std::shared_ptr<Optimizer<DIM>> newton_optimizer;
std::shared_ptr<Mesh> mesh;

std::vector<double> curr_gradients;

static double newton_gradient(const SimState<DIM>& state) {
  // Get full configuration vector
  VectorXd x = state.x_->value();
  state.x_->unproject(x);

  SparseMatrix<double, RowMajor> lhs;
  VectorXd rhs;

  const SimState<DIM>& newton_state = newton_optimizer->state();
  newton_state.x_->value() = state.x_->value();

  // Add LHS and RHS from each variable
  lhs = newton_state.x_->lhs();
  rhs = newton_state.x_->rhs();

  for (auto& var : newton_state.vars_) {
    var->update(x, state.x_->integrator()->dt());
    lhs += var->lhs();
    rhs += var->rhs();
  }

  // Uncomment to use the newton decrement instead
  //newton_state.solver_->compute(lhs);
  //VectorXd dx = newton_state.solver_->solve(rhs);
  //return dx.lpNorm<Infinity>();
  return rhs.norm();
}

void callback(const SimState<DIM>& state) {
  curr_gradients.push_back(newton_gradient(state));
}

int main(int argc, char **argv) {
  // Configure the argument parser
  args::ArgumentParser parser("Mixed FEM");
  args::Positional<std::string> inFile(parser, "json", "input scene json file");
  args::ValueFlag<int> n_arg(parser, "integer", "number of steps", {'n'});

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
  int N = args::get(n_arg); // number of iterations

  // Read and parse file
  std::ifstream input(filename);
  if (!input.good()) {
    std::cerr << "Unable to open file: " << filename << std::endl;
    return -1;
  }
  nlohmann::json args = nlohmann::json::parse(input);
  int M = args["max_newton_iterations"];

  OptimizerFactory<DIM> factory;
  MatrixXd gradients(N, M);

  // Create state
  SimState<DIM> state;
  state.load(args);
  optimizer = factory.create(state.config_->optimizer, state);
  optimizer->reset();
  optimizer->callback = callback;

  // Create args for standard FEM optimizer
  args.erase("mixed_variables");
  args["variables"] = {"stretch"};

  // Create newton FEM optimizer
  SimState<DIM> newton_state;
  newton_state.load(args);
  newton_optimizer = factory.create(newton_state.config_->optimizer,
      newton_state);
  newton_optimizer->reset();

  for (int i = 0; i < N; ++i) {
    curr_gradients.clear();

    newton_optimizer->state().x_ = std::make_unique<Displacement<DIM>>(
        *optimizer->state().x_);

    // Simulate the i-th timestep
    optimizer->step();

    // Add data
    gradients.row(i) = Map<RowVectorXd>(curr_gradients.data(), M);
  }
  igl::writeDMAT("../output/convergence.dmat", gradients);
  return 0;
}
