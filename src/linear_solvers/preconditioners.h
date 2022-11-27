#pragma once

#include "EigenTypes.h"
#include "block_matrix.h"

namespace Eigen {

  template <typename Scalar, typename MatType>
  class LumpedPreconditioner
  {
      typedef Matrix<Scalar,Dynamic,1> Vector;
      //typedef SparseMatrix<Scalar> MatType;

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


          // SparseMatrix<double> A, C;
          // var->hessian(A);
          // var->jacobian_mixed(C);

          Linv.factorize(L);
          if (Linv.info() != Eigen::Success) {
           std::cerr << "Linv prefactor failed! " << std::endl;
           exit(1);
          }
        }
        return *this;
      }
   
      BlockDiagonalPreconditioner& compute(const MatType& mat) {
        static int step = 0;
        if (step == 0) {
          std::cout << "factorize" << std::endl;
          factorize(mat);
        }
        step = (step + 1) % state_->config_->outer_steps;
        return *this;
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


  template <typename Scalar, int DIM>
  class SplitSolverPreconditioner
  {
      typedef Matrix<Scalar,Dynamic,1> Vector;
      typedef BlockMatrix<DIM> MatType;

    public:
      typedef typename Vector::StorageIndex StorageIndex;
      enum {
        ColsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic
      };
   
      SplitSolverPreconditioner() : is_initialized_(false), state_(nullptr) {}
   
      EIGEN_CONSTEXPR Index rows() const EIGEN_NOEXCEPT { return state_->size(); }
      EIGEN_CONSTEXPR Index cols() const EIGEN_NOEXCEPT { return state_->size(); }

      void init(const mfem::SimState<DIM>* state) {
        // Mass matrix inverse block
        SparseMatrixdRowMajor M = state->mesh_->mass_matrix();
        Minv_.compute(state->mesh_->mass_matrix());
        if (Minv_.info() != Eigen::Success) {
         std::cerr << "M prefactor failed! " << std::endl;
         exit(1);
        }
        is_initialized_ = true;
        state_ = state;
      }
   
      SplitSolverPreconditioner& analyzePattern(const MatType&) {
        return *this;
      }
   
      SplitSolverPreconditioner& factorize(const MatType& mat) {
        return *this;
      }
   
      SplitSolverPreconditioner& compute(const MatType& mat) {
        return *this;
      }
   
      template<typename Rhs, typename Dest>
      void _solve_impl(const Rhs& b, Dest& x) const {
        int max_iters = 100;

        x.setZero();

        Dest x0 = x;
        Rhs bi = b;

        assert(state_->vars_.size() == 0);

        for (int i = 0; i < max_iters; ++i) {
          int n = state_->x_->size();
          bi.setZero();
          x.setZero();

          
          // Compute x RHS using lagrange multipliers
          int curr_row = n;
          for (const auto& var : state_->mixed_vars_) {
            curr_row += var->size();
            const VectorXd& la = x0.segment(curr_row, var->size_dual());
            var->product_jacobian_x(la, bi.head(n), true);
            curr_row += var->size_dual();
          }

          // Solve for 'x' updates
          bi.head(n) = (b.head(n) - bi.head(n)).eval();
          x.head(n) = Minv_.solve(bi.head(n));

          curr_row = n;
          for (const auto& var : state_->mixed_vars_) {
            // Mixed variable RHS
            const VectorXd& la0 = x0.segment(curr_row + var->size(),
                var->size_dual());
            Ref<VectorXd> bs = bi.segment(curr_row,var->size());
            var->product_jacobian_mixed(la0, bs);
            bs = (b.segment(curr_row,var->size()) - bs).eval();
            // std::cout << "b s norm :" << b.segment(curr_row,var->size()).norm() << std::endl;

            // Solve for mixed variable
            Ref<VectorXd> x_s = x.segment(curr_row, var->size());
            var->product_hessian_inv(bs, x_s);
            curr_row += var->size();

            // Mixed dual variable RHS
            Ref<VectorXd> b_la = bi.segment(curr_row, var->size_dual());
            var->product_jacobian_x(x.head(n), b_la, false);
            var->product_jacobian_mixed(x_s, b_la);

            // Update mixed dual variable
            x.segment(curr_row, var->size_dual()) =  la0 +
                std::pow( std::max(0.0, 1.0 - 1*i/(max_iters*1.0)),2) * (-b.segment(curr_row, var->size_dual()) + b_la);
            curr_row += var->size_dual();
          }

          double conv = (x-x0).norm();
          if (conv < 1e-8) {
            // std::cout << "Residual: " << i << " = " << (x-x0).template lpNorm<Eigen::Infinity>() << std::endl; 
            // std::cout << "Residual x: " << i << " = " << (x-x0).head(n).template lpNorm<Eigen::Infinity>()  << std::endl; 
            // std::cout << "Residual s: " << i << " = " << (x-x0).segment(n, state_->mixed_vars_[0]->size()).template lpNorm<Eigen::Infinity>() << std::endl; 
            // break;
          }
          if ((i % 100)== 0) {
            std::cout << "Residual: " << i << " = " << (x-x0).template lpNorm<Eigen::Infinity>() << std::endl; 
            // std::cout << "Residual x: " << i << " = " << (x-x0).head(n).template lpNorm<Eigen::Infinity>()  << std::endl; 
            // std::cout << "Residual s: " << i << " = " << (x-x0).segment(n, state_->mixed_vars_[0]->size()).template lpNorm<Eigen::Infinity>() << std::endl; 
          }
          x0 = x;
        }

      }
   
      template<typename Rhs>
      inline const Solve<SplitSolverPreconditioner, Rhs>
      solve(const MatrixBase<Rhs>& b) const {
        eigen_assert(is_initialized_ && 
            "SplitSolverPreconditioner is not initialized.");
        return Solve<SplitSolverPreconditioner, Rhs>(*this, b.derived());
      }
   
      ComputationInfo info() { return Success; }
   
    protected:
      SimplicialLLT<SparseMatrixdRowMajor, Upper|Lower> Minv_;
      const mfem::SimState<DIM>* state_;
      bool is_initialized_;
  };

  template <typename Scalar, int DIM>
  class ADMMPreconditioner
  {
      typedef Matrix<Scalar,Dynamic,1> Vector;
      typedef BlockMatrix<DIM> MatType;

    public:
      typedef typename Vector::StorageIndex StorageIndex;
      enum {
        ColsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic
      };
   
      ADMMPreconditioner() : is_initialized_(false), state_(nullptr) {}
   
      EIGEN_CONSTEXPR Index rows() const EIGEN_NOEXCEPT { return state_->size(); }
      EIGEN_CONSTEXPR Index cols() const EIGEN_NOEXCEPT { return state_->size(); }

      void init(const mfem::SimState<DIM>* state) {
        // Mass matrix inverse block
        SparseMatrixdRowMajor M = state->mesh_->mass_matrix();
        is_initialized_ = true;
        state_ = state;
      }
   
      ADMMPreconditioner& analyzePattern(const MatType&) {
        return *this;
      }
   
      ADMMPreconditioner& factorize(const MatType& mat) {
        return *this;
      }
   
      ADMMPreconditioner& compute(const MatType& mat) {
        return *this;
      }
   
      template<typename Rhs, typename Dest>
      void _solve_impl(const Rhs& b, Dest& x) const {
        int max_iters = 100;

        x.setZero();

        Dest x0 = x;
        Rhs bi = b;

        assert(state_->vars_.size() == 0);

        double rho = .02;

        // Laplacian block
        SparseMatrixdRowMajor M = state_->mesh_->mass_matrix();
        SparseMatrixd H,C;
        state_->mixed_vars_[0]->hessian(H);
        state_->mixed_vars_[0]->jacobian_mixed(C);
        SparseMatrixdRowMajor HC = H + rho * C*C;

        for (const auto& var : state_->mixed_vars_) {
          SparseMatrixd Gx;
          var->jacobian_x(Gx);
          // TODO volume weighting
          // std::cout << "vol : " << state_->mesh_->volumes().size() << std::endl;
          // std::cout << "Gx : " << Gx.rows() << " cols: " << Gx.cols() << std::endl;
          SparseMatrixdRowMajor L = Gx * Gx.transpose(); 
          M += rho * L;
        }

        SimplicialLLT<SparseMatrixdRowMajor, Upper|Lower> Minv_(M);
        SimplicialLLT<SparseMatrixdRowMajor, Upper|Lower> Hinv_(HC);

        if (Minv_.info() != Eigen::Success) {
         std::cerr << "M prefactor failed! " << std::endl;
         exit(1);
        }
        if (Hinv_.info() != Eigen::Success) {
         std::cerr << "H prefactor failed! " << std::endl;
         exit(1);
        }
        // g(x) = 0.5*rho*||Bx + Cs||^2 
        // argmin_x g(x) 
        //    g'(x) = 0 = rho*B'*(Bx+Cs)
        //    rho B'B x = -rho B'(Cs)
        // argmin_s g(x)
        //    g'(s) = 0 = rho*C*(Bx+Cs)
        //    rho C'C s = - rho C*(Bx)

        for (int i = 0; i < max_iters; ++i) {
          int n = state_->x_->size();
          bi.setZero();
          x.setZero();

          
          // Compute x RHS using lagrange multipliers
          int curr_row = n;
          for (const auto& var : state_->mixed_vars_) {
            Ref<VectorXd> x_s = x.segment(curr_row, var->size());
            VectorXd Cs(x_s.size());
            VectorXd b_rho(n);
            Cs.setZero();
            b_rho.setZero();
            var->product_jacobian_mixed(x_s, Cs);
            var->product_jacobian_x(Cs, b_rho, true);
            bi.head(n) += b_rho * rho;

            curr_row += var->size();
            const VectorXd& la = x0.segment(curr_row, var->size_dual());
            var->product_jacobian_x(la, bi.head(n), true);
            curr_row += var->size_dual();
          }

          // Solve for 'x' updates
          bi.head(n) = b.head(n) - bi.head(n) ;
          x.head(n) = Minv_.solve(bi.head(n));

          curr_row = n;
          for (const auto& var : state_->mixed_vars_) {
            // Mixed variable RHS
            const VectorXd& la0 = x0.segment(curr_row + var->size(),
                var->size_dual());
            Ref<VectorXd> bs = bi.segment(curr_row,var->size());
            var->product_jacobian_mixed(la0, bs);

            // Mixed variable quadratic penalty RHS
            // rho C*(Bx)
            VectorXd Bx(la0.size());
            VectorXd b_rho(la0.size());
            Bx.setZero();
            b_rho.setZero();
            var->product_jacobian_x(x.head(n), Bx, false);
            var->product_jacobian_mixed(Bx, b_rho);
            bs += b_rho * rho;

            bs = (b.segment(curr_row,var->size()) - bs).eval();
            // std::cout << "b s norm :" << b.segment(curr_row,var->size()).norm() << std::endl;

            // Solve for mixed variable
            Ref<VectorXd> x_s = x.segment(curr_row, var->size());
            // var->product_hessian_inv(bs, x_s);
            x_s = Hinv_.solve(bs);
            curr_row += var->size();

            // Mixed dual variable RHS
            Ref<VectorXd> b_la = bi.segment(curr_row, var->size_dual());
            var->product_jacobian_x(x.head(n), b_la, false);
            var->product_jacobian_mixed(x_s, b_la);

            // Update mixed dual variable
            x.segment(curr_row, var->size_dual()) =  la0 +
                /*std::pow( std::max(0.0, 1.0 - 1*i/(max_iters*100.0)),2) */ (-b.segment(curr_row, var->size_dual()) + b_la);
            curr_row += var->size_dual();
          }

          double conv = (x-x0).norm();
          if (conv < 1e-8) {
            // std::cout << "Residual: " << i << " = " << (x-x0).template lpNorm<Eigen::Infinity>() << std::endl; 
            // std::cout << "Residual x: " << i << " = " << (x-x0).head(n).template lpNorm<Eigen::Infinity>()  << std::endl; 
            // std::cout << "Residual s: " << i << " = " << (x-x0).segment(n, state_->mixed_vars_[0]->size()).template lpNorm<Eigen::Infinity>() << std::endl; 
            // break;
          }
          if ((i % 100)== 0) {
            std::cout << "Residual: " << i << " = " << (x-x0).template lpNorm<Eigen::Infinity>() << std::endl; 
            // std::cout << "Residual x: " << i << " = " << (x-x0).head(n).template lpNorm<Eigen::Infinity>()  << std::endl; 
            // std::cout << "Residual s: " << i << " = " << (x-x0).segment(n, state_->mixed_vars_[0]->size()).template lpNorm<Eigen::Infinity>() << std::endl; 
          }
          x0 = x;
        }

      }
   
      template<typename Rhs>
      inline const Solve<ADMMPreconditioner, Rhs>
      solve(const MatrixBase<Rhs>& b) const {
        eigen_assert(is_initialized_ && 
            "ADMMPreconditioner is not initialized.");
        return Solve<ADMMPreconditioner, Rhs>(*this, b.derived());
      }
   
      ComputationInfo info() { return Success; }
   
    protected:
      // SimplicialLLT<SparseMatrixdRowMajor, Upper|Lower> Minv_;
      // SimplicialLLT<SparseMatrixdRowMajor, Upper|Lower> Hinv_;
      const mfem::SimState<DIM>* state_;
      bool is_initialized_;
  };
}
