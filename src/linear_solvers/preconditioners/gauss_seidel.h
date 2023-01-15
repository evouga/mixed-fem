#pragma once

#include "EigenTypes.h"

namespace Eigen {
  template <typename Scalar>
  class GaussSeidelPreconditioner
  {
      typedef Matrix<Scalar,Dynamic,1> Vector;
      typedef SparseMatrix<Scalar> MatType;

    public:
      typedef typename Vector::StorageIndex StorageIndex;
      enum {
        ColsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic
      };
   
      GaussSeidelPreconditioner() : is_initialized_(false) {}
   
      EIGEN_CONSTEXPR Index rows() const EIGEN_NOEXCEPT { return A_.rows(); }
      EIGEN_CONSTEXPR Index cols() const EIGEN_NOEXCEPT { return A_.cols(); }
   
      GaussSeidelPreconditioner& analyzePattern(const MatType&) {
        return *this;
      }
   
      GaussSeidelPreconditioner& factorize(const MatType& mat) {
        return *this;
      }
   
      GaussSeidelPreconditioner& compute(const MatType& mat) {
        is_initialized_ = true;
        A_ = mat;
        return *this;
      }

      // Eigen iterative solver API
      Scalar error() { return error_; }
      int iterations() { return iters_; }
      void setTolerance(Scalar tol) { tol_ = tol; }
      void setMaxIterations(int max_iters) { max_iters_ = max_iters; }
   
      template<typename Rhs, typename Dest>
      void _solve_impl(const Rhs& b, Dest& x) const {
        x.setZero();
        _solve_with_guess_impl(b, x);
      }

      template<typename Rhs,typename Dest>
      void _solve_with_guess_impl(const Rhs& b, Dest& x) const {
        auto AL = A_.template triangularView<Lower>();
        auto AU = A_.template triangularView<StrictlyUpper>();

        Scalar b_norm = b.norm();

        for (iters_ = 0; iters_ < max_iters_; ++iters_) {
          x = AL.solve(b - AU*x);

          error_ = (A_*x - b).norm() / b_norm;
          if (error_ < tol_) {
            break;
          }
        }
      }
   
      template<typename Rhs> inline const Solve<GaussSeidelPreconditioner, Rhs>
      solve(const MatrixBase<Rhs>& b) const {
        eigen_assert(is_initialized_ && 
            "GaussSeidelPreconditioner is not initialized.");
        return Solve<GaussSeidelPreconditioner, Rhs>(*this, b.derived());
      }
      

      template<typename Rhs,typename Guess>
      inline const SolveWithGuess<GaussSeidelPreconditioner, Rhs, Guess>
      solveWithGuess(const MatrixBase<Rhs>& b, const Guess& x0) const {
        eigen_assert(is_initialized_ && 
            "GaussSeidelPreconditioner is not initialized.");
        return SolveWithGuess<GaussSeidelPreconditioner, Rhs, Guess>(
            *this, b.derived(), x0);
      }
   
      ComputationInfo info() { return Success; }
   
    protected:
      bool is_initialized_;
      int max_iters_ = 100;
      mutable int iters_ = 0;
      mutable Scalar error_;
      Scalar tol_ = 0;
      MatType A_;
  };
}