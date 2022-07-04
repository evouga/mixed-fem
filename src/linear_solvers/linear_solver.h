#pragma once

#include "EigenTypes.h"

namespace mfem {

  template <typename Scalar, int Ordering>
  class LinearSolver {

  public:

    virtual void compute(const Eigen::SparseMatrix<Scalar, Ordering>& A) = 0;

    virtual Eigen::VectorXx<Scalar> solve(const Eigen::VectorXx<Scalar>& b) = 0; 

    virtual ~LinearSolver() = default;
  };

}
