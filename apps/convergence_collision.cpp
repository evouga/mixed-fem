
// libigl
#include <igl/boundary_facets.h>
#include <igl/IO>

#include "args/args.hxx"
#include "json/json.hpp"

#include "factories/optimizer_factory.h"

using namespace Eigen;
using namespace mfem;

const int DIM = 2;

std::shared_ptr<Optimizer<DIM>> newton_optimizer;
std::shared_ptr<Optimizer<DIM>> optimizer;
std::vector<double> curr_decrements;
double final_decrement;

static double newton_decrement(const SimState<DIM>& state) {
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

  //newton_state.solver_->compute(lhs);
  //VectorXd dx = newton_state.solver_->solve(rhs);

  // Use infinity norm of deltas as termination criteria
  // double decrement = dx.norm();
  double decrement = rhs.norm();
  return decrement;
}

void callback(const SimState<DIM>& state) {
  double decrement = newton_decrement(state);
  final_decrement = decrement;
  std::cout << "decrement: " << decrement << std::endl;
  curr_decrements.push_back(decrement);
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

  OptimizerFactory<DIM> factory;
  std::string filename = args::get(inFile);

  // Read and parse file
  nlohmann::json args;
  std::ifstream input(filename);
  if (input.good()) {
    args = nlohmann::json::parse(input);
  } else {
    std::cerr << "Unable to open file: " << filename << std::endl;
    return -1;
  }

  int M = args["max_newton_iterations"];
  VectorXd dhats(8);
  dhats << 0.01, 0.005, 0.0025, 0.001, 0.00075, 0.0005, 0.00025, 0.0001;
  int N = dhats.size();

  MatrixXd decrements(N,M);
  VectorXd final_decrements(N);

  for (int i = 0; i < N; ++i) {
    // Read and parse file
    std::ifstream input(filename);
    nlohmann::json args = nlohmann::json::parse(input);
    curr_decrements.clear();
    args["dhat"] = dhats(i);

    SimState<DIM> state;
    state.load(args);
    optimizer = factory.create(state.config_->optimizer, state);
    optimizer->reset();
    optimizer->callback = callback;

    // Switch to newton FEM
    // Note just supports stretch right now
    args.erase("mixed_variables");
    args["variables"] = {"stretch", "collision"};

    SimState<DIM> newton_state;
    newton_state.load(args);
    newton_optimizer = factory.create(newton_state.config_->optimizer, newton_state);
    newton_optimizer->reset();

    // Do one dummy step
    optimizer->step();
    // curr_decrements.clear();
    // optimizer->step();

    final_decrements(i) = final_decrement;
    decrements.row(i) = Map<RowVectorXd>(curr_decrements.data(), M);
  }
  std::cout << "final_decrements: " << final_decrements << std::endl;
  igl::writeDMAT("../output/convergence.dmat", decrements);
  igl::writeDMAT("../output/finalconvergence_converge.dmat", final_decrements);
  igl::writeDMAT("../output/dhat_values.dmat", dhats);
  return 0;
}

