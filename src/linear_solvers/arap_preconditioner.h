#pragma once

#include "simulation_state.h"
#include "EigenTypes.h"

namespace Eigen {

  // Block jacobi diagonal preconditioner. Inverts the NxN blocks of the sparse matrix along the diagonal
  void block_jacobi_preconditioner(SparseMatrix<double,RowMajor>& A, const VectorXd& b, VectorXd& x) {
// assume A is a sparse matrix with blocks of size nxn
    // assume b is a vector with n blocks
    // assume x is a vector with n blocks
    // assume A is symmetric
    // assume A is positive definite
    // assume A is block diagonal
    // assume A is square
    // assume A is stored in row major order
    // assume A is stored in compressed sparse row format

    // get the number of blocks
    int n = b.size();
    // get the number of rows in each block
    int m = A.rows()/n;

    // loop over the blocks
    for (int i = 0; i < n; i++) {
      // // get the diagonal block
      // SparseMatrix<double> Aii = A.block(i*m,i*m,m,m);
      // // get the diagonal block of b
      // VectorXd bi = b.segment(i*m,m);
      // // get the diagonal block of x
      // VectorXd& xi = x.segment(i*m,m);
      // // solve the block system
      // xi = Aii.colPivHouseholderQr().solve(bi);
    }
  }

  template <typename Scalar, int DIM>
  class ArapPreconditioner {
      typedef Matrix<Scalar,Dynamic,1> Vector;
      typedef SparseMatrix<double, RowMajor> MatType;

    public:
      typedef typename Vector::StorageIndex StorageIndex;
      enum {
        ColsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic
      };
   
      ArapPreconditioner() : is_initialized_(false), state_(nullptr) {}
   
      EIGEN_CONSTEXPR Index rows() const EIGEN_NOEXCEPT { return state_->size(); }
      EIGEN_CONSTEXPR Index cols() const EIGEN_NOEXCEPT { return state_->size(); }

      void init(const mfem::SimState<DIM>* state) {
        std::cout << "hey" << std::endl;

        if (state->mixed_vars_.size() == 0) {
          throw std::runtime_error("Using ARAP preconditioner without mixed vars");
        }

        // Conjure up some silly stiffness
        double k = state->config_->h * state->config_->h;
        double ym = 1e4;
        double pr = 0.45;
        double mu = ym/(2.0*(1.0+pr));
        k *= mu;

        SparseMatrixd Gx;
        state->mixed_vars_[0]->jacobian_x(Gx);

        // TODO double multiplying by areas currently. Make a volume diagonal matrix
        MatType L = k * Gx * Gx.transpose();
        L += state->mesh_->mass_matrix(); 
        solver_.compute(L);
        if (solver_.info() != Eigen::Success) {
          throw std::runtime_error("ARAP preconditioner factorization failed!");
        }
        // const T* c = dynamic_cast<const *>(var);
        // if (!c) return false;
        // if (state->mixed_vars_) {
        // } else {
        //   throw std::runtime_error("Using ARAP preconditioner without mixed-stretch");
        // }
        is_initialized_ = true;
        state_ = state;
      }
   
      ArapPreconditioner& analyzePattern(const MatType&) {
        return *this;
      }
   
      ArapPreconditioner& factorize(const MatType& mat) {
        mat.block(1,1,1,1);
        // for (auto& var : state_->mixed_vars_) {
        //   SparseMatrixd Gx;
        //   var->jacobian_x(Gx);
        //   SparseMatrixdRowMajor L = Gx.transpose() * Mlumpinv_ * Gx; 
        //   for (int i = 0; i < L.rows(); ++i) {
        //     L.coeffRef(i,i) += 1e-8;
        //   }

        //   // SparseMatrix<double> A, C;
        //   // var->hessian(A);
        //   // var->jacobian_mixed(C);
        //   //
        //   Linv.factorize(L);
        //   if (Linv.info() != Eigen::Success) {
        //    std::cerr << "Linv prefactor failed! " << std::endl;
        //    exit(1);
        //   }
        // }
        return *this;
      }
   
      ArapPreconditioner& compute(const MatType& mat) {
        // static int step = 0;
        // if (step == 0) {
        //   std::cout << "factorize" << std::endl;
        //   factorize(mat);
        // }
        // step = (step + 1) % state_->config_->outer_steps;
        return *this;
      }
   
      template<typename Rhs, typename Dest>
      void _solve_impl(const Rhs& b, Dest& x) const {
        x = solver_.solve(b);
      }
   
      template<typename Rhs>
      inline const Solve<ArapPreconditioner, Rhs>
      solve(const MatrixBase<Rhs>& b) const {
        eigen_assert(is_initialized_ && 
            "ArapPreconditioner is not initialized.");
        return Solve<ArapPreconditioner, Rhs>(*this, b.derived());
      }
   
      ComputationInfo info() { return Success; }
   
    protected:
      SimplicialLLT<MatType, Upper|Lower> solver_;
      const mfem::SimState<DIM>* state_;
      bool is_initialized_;
  };

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
