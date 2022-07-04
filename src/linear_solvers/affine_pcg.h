#pragma once

#include "linear_solver.h"
#include "pcg.h"

namespace mfem {

  struct AffinePCGInput {

  };

  template <typename Solver, typename Scalar, int Ordering>
  class AffinePCG : public LinearSolver<Scalar, Ordering> {
  public:

    AfficePCG() : has_init_(false) {

    }

    void compute(const Eigen::SparseMatrix<Scalar, Ordering>& A) override {
      if (!has_init_) {
        solver_.analyzePattern(A);
        has_init_ = true;
      }
      solver_.factorize(A);
    }

    Eigen::VectorXx<Scalar> solve(const Eigen::MatrixBase<Scalar>& b) override {
      assert(has_init_);
      return solver_.compute(b);
    }

  private:

    Solver solver_;
    bool has_init_;

  };


}
