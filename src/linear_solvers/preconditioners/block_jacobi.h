#pragma once

#include "simulation_state.h"
#include "EigenTypes.h"
#include "variables/mixed_stretch.h"

namespace Eigen {

  template <typename Scalar, int DIM>
  class BlockJacobiPreconditioner {
      typedef Matrix<Scalar,Dynamic,1> Vector;
      typedef SparseMatrix<double, RowMajor> MatType;

    public:
      typedef typename Vector::StorageIndex StorageIndex;
      enum {
        ColsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic
      };
   
      BlockJacobiPreconditioner() : is_initialized_(false), state_(nullptr) {}
   
      EIGEN_CONSTEXPR Index rows() const EIGEN_NOEXCEPT { return state_->size(); }
      EIGEN_CONSTEXPR Index cols() const EIGEN_NOEXCEPT { return state_->size(); }

      void init(const mfem::SimState<DIM>* state) {
        is_initialized_ = true;
        state_ = state;
      }
   
      BlockJacobiPreconditioner& analyzePattern(const MatType&) {
        return *this;
      }
   
      BlockJacobiPreconditioner& factorize(const MatType& mat) {
        return *this;
      }
   
      BlockJacobiPreconditioner& compute(const MatType& mat) {
        A_ = mat;
        return *this;
      }
   
      template<typename Rhs, typename Dest>
      void _solve_impl(const Rhs& b, Dest& x) const {

        int N = b.size() / DIM;
        x.resize(b.size());

        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
          Matrix<Scalar,DIM,DIM> Aii = A_.template block(i*DIM,i*DIM, DIM,DIM);
          const Matrix<Scalar, DIM, 1>& bi = b.template segment<DIM>(i*DIM);
          Ref<Matrix<Scalar, DIM, 1>> xi = x.template segment<DIM>(DIM*i);
          xi = Aii.inverse() * bi;
          //std::cout << "i : " << i << " bi: \n" << bi << std::endl;
          //std::cout << "i : " << i << " xi: \n" << xi << std::endl;
          //std::cout << "i : " << i << " Aii: \n" << Aii << std::endl;
        }
      }
   
      template<typename Rhs>
      inline const Solve<BlockJacobiPreconditioner, Rhs>
      solve(const MatrixBase<Rhs>& b) const {
        eigen_assert(is_initialized_ && 
            "BlockJacobiPreconditioner is not initialized.");
        return Solve<BlockJacobiPreconditioner, Rhs>(*this, b.derived());
      }
   
      ComputationInfo info() { return Success; }
   
    protected:
      MatType A_;
      const mfem::SimState<DIM>* state_;
      bool is_initialized_;
  };
}
