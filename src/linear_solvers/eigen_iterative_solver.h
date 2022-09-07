#pragma once

#include "linear_solver.h"

namespace mfem {

  template <typename Solver, typename Scalar, int Ordering>
  class EigenIterativeSolver : public LinearSolver<Scalar, Ordering> {
  public:

    EigenIterativeSolver(std::shared_ptr<SimConfig> config) {
      solver_.setMaxIterations(config->max_iterative_solver_iters);
      solver_.setTolerance(config->itr_tol);
    }

    void compute(const Eigen::SparseMatrix<Scalar, Ordering>& A) override {
      solver_.compute(A);
    }

    Eigen::VectorXx<Scalar> solve(const Eigen::VectorXx<Scalar>& b) override {
      Eigen::VectorXx<Scalar> x = solver_.solve(b);
      std::cout << "- CG iters: " << solver_.iterations() << std::endl;
      std::cout << "- CG error: " << solver_.error() << std::endl;
      return x;
    }

  private:

    Solver solver_;

  };


}
