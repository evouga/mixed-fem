#pragma once

#include "EigenTypes.h"

namespace Eigen {

  template <typename Scalar, typename MatType>
  class LumpedPreconditioner
  {
      typedef Matrix<Scalar,Dynamic,1> Vector;

    public:
      typedef typename Vector::StorageIndex StorageIndex;
      enum {
        ColsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic
      };
   
      LumpedPreconditioner() : is_initialized_(false) {}
   
      EIGEN_CONSTEXPR Index rows() const EIGEN_NOEXCEPT { return invdiag_.size(); }
      EIGEN_CONSTEXPR Index cols() const EIGEN_NOEXCEPT { return invdiag_.size(); }
   
      LumpedPreconditioner& analyzePattern(const MatType&) {
        return *this;
      }
   
      LumpedPreconditioner& factorize(const MatType& mat) {
        invdiag_.resize(mat.cols());
        Vector diag = mat * Vector::Ones(mat.cols());
        invdiag_ = 1.0 / (diag.array().abs() / mat.cols());
        is_initialized_ = true;
        return *this;
      }
   
      LumpedPreconditioner& compute(const MatType& mat) {
        return factorize(mat);
      }
   
      template<typename Rhs, typename Dest>
      void _solve_impl(const Rhs& b, Dest& x) const {
        x = invdiag_.array() * b.array() ;
      }
   
      template<typename Rhs> inline const Solve<LumpedPreconditioner, Rhs>
      solve(const MatrixBase<Rhs>& b) const {
        eigen_assert(is_initialized_ && 
            "LumpedPreconditioner is not initialized.");
        eigen_assert(invdiag_.size() == b.rows()
            && "LumpedPreconditioner::solve(): invalid"
            "number of rows of the right hand side matrix b");
        return Solve<LumpedPreconditioner, Rhs>(*this, b.derived());
      }
   
      ComputationInfo info() { return Success; }
   
    protected:
      Vector invdiag_;
      bool is_initialized_;
  };
}