#pragma once

#include <EigenTypes.h>

namespace mfem {

  // Positive-definite fix a matrix and also invert it
  // using the eigen-decomposition.
  template<typename Derived, typename Scalar>
  void psd_fix_invert(Eigen::MatrixBase<Derived>& A,
      Eigen::MatrixBase<Derived>& Ainv,
      Eigen::Matrix<Scalar,Eigen::MatrixBase<Derived>::RowsAtCompileTime,1>& evals,
      Scalar tol=1e-6) {
  

    Eigen::SelfAdjointEigenSolver<Derived> es(A);
    bool is_fixed = false;
    evals = es.eigenvalues().real();

    for (int i = 0; i < evals.size(); ++i) {
      if (evals(i) < tol) {
        evals(i) = tol;
        is_fixed = true;
      }
    }

    A = es.eigenvectors().real() * evals.asDiagonal()
        * es.eigenvectors().real().transpose();

    const int N = Eigen::MatrixBase<Derived>::RowsAtCompileTime;
    Eigen::Matrix<Scalar,N,1> eval_inv = evals.array().inverse();
    Ainv = es.eigenvectors().real() * eval_inv.asDiagonal()
      * es.eigenvectors().real().transpose();
  }
}
