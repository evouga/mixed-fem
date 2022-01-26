#pragma once

#include <EigenTypes.h>

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

void diag_compliance(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    const Eigen::VectorXd& vols, double alpha,
    std::vector<Eigen::Triplet<double>>& trips);

namespace Eigen {
  
  template <typename _Scalar>
  class FemPreconditioner
  {
      typedef _Scalar Scalar;
      typedef Matrix<Scalar,Dynamic,1> Vector;

    public:
      typedef typename Vector::StorageIndex StorageIndex;
      enum {
        ColsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic
      };
   
      FemPreconditioner() : m_isInitialized(false) {}
   
      EIGEN_CONSTEXPR Index rows() const EIGEN_NOEXCEPT { return lhs_.rows(); }
      EIGEN_CONSTEXPR Index cols() const EIGEN_NOEXCEPT { return lhs_.cols(); }
   
      template<typename MatType>
      FemPreconditioner& init(const MatType& lhs ) {
        lhs_ = lhs;
        solver_.compute(lhs_);
        m_isInitialized = true;
        return *this;
      }

      template<typename MatType>
      FemPreconditioner& analyzePattern(const MatType& )
      {
        return *this;
      }
   
      template<typename MatType>
      FemPreconditioner& factorize(const MatType& mat)
      {
        return *this;
      }
   
      template<typename MatType>
      FemPreconditioner& compute(const MatType& mat)
      {
        return factorize(mat);
      }
   
      template<typename Rhs, typename Dest>
      void _solve_impl(const Rhs& b, Dest& x) const
      {
        x = solver_.solve(b);
      }
   
      template<typename Rhs> inline const Solve<FemPreconditioner, Rhs>
      solve(const MatrixBase<Rhs>& b) const
      {
        eigen_assert(m_isInitialized && 
                "FemPreconditioner is not initialized.");
        eigen_assert(lhs_.rows()==b.rows()
                  && "FemPreconditioner::solve(): "
                  " invalid number of rows of the right hand side matrix b");
        return Solve<FemPreconditioner, Rhs>(*this, b.derived());
      }
   
      ComputationInfo info() { return Success; } // always winning baby
   
    protected:
      bool m_isInitialized;

      SparseMatrixd lhs_;
      #if defined(SIM_USE_CHOLMOD)
      CholmodSimplicialLDLT<SparseMatrixd> solver_;
      #else
      SimplicialLDLT<SparseMatrixd> solver_;
      #endif
  };
}
