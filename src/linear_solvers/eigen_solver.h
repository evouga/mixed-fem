#pragma once

#include "linear_solver.h"

namespace mfem {

  template <typename Solver, typename Scalar, int Ordering>
  class EigenSolver : public LinearSolver<Scalar, Ordering> {
  public:

    EigenSolver() : has_init_(false) {}

    void compute(const Eigen::SparseMatrix<Scalar, Ordering>& A) override {
      //if (!has_init_) { // Can't do this for collisions since pattern changes
        solver_.analyzePattern(A);
        has_init_ = true;
      //}
      solver_.factorize(A);
      if (solver_.info() != Eigen::Success) {
       std::cerr << "prefactor failed! " << std::endl;
       exit(1);
      }
    }

    Eigen::VectorXx<Scalar> solve(const Eigen::VectorXx<Scalar>& b) override {
      assert(has_init_);
      return solver_.solve(b);
    }

  private:

    Solver solver_;
    bool has_init_;

  };


}
