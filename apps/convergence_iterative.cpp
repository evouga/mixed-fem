// Evaluate iterative solve convergence
// libigl
#include <igl/boundary_facets.h>
#include <igl/IO>

#include "args/args.hxx"
#include "json/json.hpp"

#include "factories/optimizer_factory.h"
#include "optimizers/newton_optimizer.h"

using namespace Eigen;
using namespace mfem;

const int DIM = 3;

std::shared_ptr<NewtonOptimizer<DIM>> optimizer;
std::shared_ptr<Mesh> mesh;
std::vector<double> curr_relres;

static std::vector<double> solver_residual(const SimState<DIM>& state) {

  LinearSolver<double,DIM>* linsolver = optimizer->linear_solver();
  std::vector<double> relres = linsolver->residuals();
  if (relres.size() == 0) {
    return std::vector<double>(1, 0.0);
  } else {
    return relres;
  }
}

void callback(const SimState<DIM>& state) {
  if (curr_relres.size() == 0) {
    curr_relres = solver_residual(state);
  }
}

int main(int argc, char **argv) {
  // Configure the argument parser
  args::ArgumentParser parser("Mixed FEM");
  args::Positional<std::string> inFile(parser, "json", "input scene json file");
  args::ValueFlag<int> n_arg(parser, "integer", "number of steps", {'n'});
  args::ValueFlag<int> m_arg(parser, "integer", "number of solver iters", {'m'});

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
  int M = args::get(m_arg); // number of iterations

  // Read and parse file
  std::ifstream input(filename);
  if (!input.good()) {
    std::cerr << "Unable to open file: " << filename << std::endl;
    return -1;
  }
  nlohmann::json args = nlohmann::json::parse(input);

  OptimizerFactory<DIM> factory;
  MatrixXd relres(N, M);

  // Create state
  SimState<DIM> state;
  state.load(args);
  state.config_->max_iterative_solver_iters = M;
  state.config_->itr_tol = 0;
  state.config_->itr_save_residuals = true;
  optimizer = std::make_unique<NewtonOptimizer<DIM>>(state);
  optimizer->reset();
  optimizer->callback = callback;

  for (int i = 0; i < N; ++i) {
    curr_relres.clear();

    // Simulate the i-th timestep
    optimizer->step();

    if (curr_relres.size() < M) {
      int sz = (M - curr_relres.size());
      for (int j = 0; j < sz; ++j) {
        curr_relres.push_back(curr_relres.back());
      }
    }

    // Add data
    relres.row(i) = Map<RowVectorXd>(curr_relres.data(), M);
  }
  std::cout << "relres: " << relres << std::endl;
  igl::writeDMAT("../output/relres.dmat", relres);
  return 0;
}
