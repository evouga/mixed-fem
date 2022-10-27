#pragma once

#include "EigenTypes.h"
#include "block_matrix.h"

namespace Eigen {

  template <typename Scalar, int DIM>
  class LumpedPreconditioner
  {
      typedef Matrix<Scalar,Dynamic,1> Vector;
      typedef BlockMatrix<DIM> MatType;

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
        invdiag_ = 1.0 / diag.array().abs();
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

  template <typename Scalar, int DIM>
  class BlockDiagonalPreconditioner
  {
      typedef Matrix<Scalar,Dynamic,1> Vector;
      typedef BlockMatrix<DIM> MatType;

    public:
      typedef typename Vector::StorageIndex StorageIndex;
      enum {
        ColsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic
      };
   
      BlockDiagonalPreconditioner() : is_initialized_(false), state_(nullptr) {}
   
      EIGEN_CONSTEXPR Index rows() const EIGEN_NOEXCEPT { return state_->size(); }
      EIGEN_CONSTEXPR Index cols() const EIGEN_NOEXCEPT { return state_->size(); }

      void init(const mfem::SimState<DIM>* state) {
        std::cout << "hey" << std::endl;
        // Mass matrix inverse block
        SparseMatrixdRowMajor M = state->mesh_->mass_matrix();
        Minv.compute(state->mesh_->mass_matrix());
        if (Minv.info() != Eigen::Success) {
         std::cerr << "M prefactor failed! " << std::endl;
         exit(1);
        }
        // Form lumped mass matrix
        VectorXd Mlump = state->mesh_->mass_matrix() * VectorXd::Ones(M.cols());

        // Invert lumped mass matrix
        Mlumpinv_.resize(Mlump.size(), Mlump.size());
        std::vector<Triplet<double>> trips;
        for (int i = 0; i < Mlump.size(); ++i) {
          trips.push_back(Triplet<double>(i, i, 1.0/Mlump(i)));
        }
        Mlumpinv_.setFromTriplets(trips.begin(),trips.end());

        // Laplacian block
        for (const auto& var : state->mixed_vars_) {
          SparseMatrixd Gx;
          var->jacobian_x(Gx);
          SparseMatrixdRowMajor L = Gx.transpose() * Mlumpinv_ * Gx; 
          for (int i = 0; i < L.rows(); ++i) {
            L.coeffRef(i,i) += 1e-8;
          }

          Linv.compute(L);
          if (Linv.info() != Eigen::Success) {
           std::cerr << "Linvprefactor failed! " << std::endl;
           exit(1);
          } else {
            std::cout << "Linv ok:" << std::endl;

          }
        }
        is_initialized_ = true;
        state_ = state;
      }
   
      BlockDiagonalPreconditioner& analyzePattern(const MatType&) {
        return *this;
      }
   
      BlockDiagonalPreconditioner& factorize(const MatType& mat) {
        for (auto& var : state_->mixed_vars_) {
          SparseMatrixd Gx;
          var->jacobian_x(Gx);
          SparseMatrixdRowMajor L = Gx.transpose() * Mlumpinv_ * Gx; 
          for (int i = 0; i < L.rows(); ++i) {
            L.coeffRef(i,i) += 1e-8;
          }
          Linv.factorize(L);
          if (Linv.info() != Eigen::Success) {
           std::cerr << "Linvprefactor failed! " << std::endl;
           exit(1);
          } else {
            std::cout << "Linv ok:" << std::endl;
          }
        }
        return *this;
      }
   
      BlockDiagonalPreconditioner& compute(const MatType& mat) {
        return factorize(mat);
      }
   
      template<typename Rhs, typename Dest>
      void _solve_impl(const Rhs& b, Dest& x) const {

        //std::cout << "x norm: " << x.norm() << std::endl;
        x.setZero();
        x.head(state_->x_->size()) = Minv.solve(b.head(state_->x_->size()));

        int curr_row = state_->x_->size();
        for (auto& var : state_->mixed_vars_) {
          var->product_hessian_inv(b.segment(curr_row, var->size()),
              x.segment(curr_row, var->size()));
          curr_row += var->size();

          //Ref<VectorXd> out = x.segment(curr_row, var->size_dual()); // aliasing?
          //var->product_jacobian_mixed(b.segment(curr_row, var->size_dual()),
          //    out,true);
          //VectorXd out2 = out;
          //out2.setZero();
          //var->product_hessian(out,
          //    out2);
          //out.setZero();
          //var->product_jacobian_mixed(out2,
          //    out,true);
          //var->product_hessian(b.segment(curr_row, var->size_dual()),
          //    x.segment(curr_row, var->size_dual()));
          //
          x.segment(curr_row, var->size_dual()) = Linv.solve(
              b.segment(curr_row, var->size_dual()));
          curr_row += var->size_dual();
        }
      }
   
      template<typename Rhs>
      inline const Solve<BlockDiagonalPreconditioner, Rhs>
      solve(const MatrixBase<Rhs>& b) const {
        eigen_assert(is_initialized_ && 
            "BlockDiagonalPreconditioner is not initialized.");
        return Solve<BlockDiagonalPreconditioner, Rhs>(*this, b.derived());
      }
   
      ComputationInfo info() { return Success; }
   
    protected:
      SparseMatrixdRowMajor Mlumpinv_;
      SimplicialLLT<SparseMatrixdRowMajor, Upper|Lower> Minv;
      SimplicialLDLT<SparseMatrixdRowMajor, Upper|Lower> Linv;
      const mfem::SimState<DIM>* state_;
      bool is_initialized_;
  };
}
