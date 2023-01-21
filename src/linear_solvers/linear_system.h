#pragma once

#include "EigenTypes.h"
#include "simulation_state.h"
#include "block_matrix.h"

namespace mfem {

  template<typename Scalar>
  class SystemMatrixPD {

  public:

    typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> MatrixType;
    typedef Eigen::VectorXx<Scalar> VectorType;

    template<int DIM>
    void pre_solve(const SimState<DIM>* state) {
      // Add LHS and RHS from each variable
      lhs_ = state->x_->lhs();
      rhs_ = state->x_->rhs();

      for (auto& var : state->vars_) {
        lhs_ += var->lhs();
        rhs_ += var->rhs();
      }
      for (auto& var : state->mixed_vars_) {
        lhs_ += var->lhs();
        rhs_ += var->rhs();
      }
    }

    template<int DIM>
    void post_solve(const SimState<DIM>* state, Eigen::VectorXd& dx) {
      // TODO projected external force not full one!
      // double h = state->x_->integrator()->dt();
      // double fac = h*h*(1.0 - state->config_->inertia_blend_factor);

      state->x_->delta() = dx;// + fac * state->mesh_->external_force();
;
      for (auto& var : state->mixed_vars_) {
        var->solve(dx);
      }
    }

    const MatrixType& A() const {
      return lhs_;
    }

    const VectorType& b() const {
      return rhs_;
    }


  private:
    
    // linear system left hand side
    MatrixType lhs_; 

    // linear system right hand side
    VectorType rhs_;       
  };

  template<typename Scalar>
  class SystemMatrixThrust {

  public:

    typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> MatrixType;
    typedef Eigen::VectorXx<Scalar> VectorType;

    void pre_solve(const SimState<3,STORAGE_THRUST>* state);

    void post_solve(const SimState<3,STORAGE_THRUST>* state,
        Eigen::VectorXd& dx);

    const MatrixType& A() const { return lhs_; }
    const VectorType& b() const { return rhs_; }

  private:
    MatrixType lhs_; // linear system left hand side
    VectorType rhs_; // linear system right hand side       
  };

  template<typename Scalar, int DIM>
  class SystemMatrixIndefinite {
  public:

    typedef Eigen::BlockMatrix<DIM> MatrixType;
    typedef Eigen::VectorXx<Scalar> VectorType;

    void pre_solve(const SimState<DIM>* state) {
      lhs_.attach_state(state);

      // Set rhs system
      rhs_.resize(lhs_.rows());

      rhs_.head(state->x_->size()) = -state->x_->gradient();

      int curr_row = state->x_->size();
      for (const auto& var : state->mixed_vars_) {
        rhs_.segment(curr_row, var->size()) = -var->gradient_mixed();
        curr_row += var->size();
        rhs_.segment(curr_row, var->size_dual()) = -var->gradient_dual();
        curr_row += var->size_dual();
      }
      assert(rhs_.size() == curr_row);
    }

    void post_solve(const SimState<DIM>* state, Eigen::VectorXd& dx) {

      state->x_->delta() = dx.head(state->x_->size());

      int curr_row = state->x_->size();
      for (auto& var : state->mixed_vars_) {
        var->delta() = dx.segment(curr_row, var->size());
        curr_row += var->size();
        var->lambda() = dx.segment(curr_row, var->size_dual());
        curr_row += var->size_dual();
      }
    }

    const MatrixType& A() const {
      return lhs_;
    }

    const VectorType& b() const {
      return rhs_;
    }


  private:
    
    // linear system left hand side
    MatrixType lhs_; 

    // linear system right hand side
    VectorType rhs_;       
  };

  template<int DIM>
  class DualCondensedSystem {
  public:

    typedef Eigen::SparseMatrix<double,Eigen::RowMajor> MatrixType;
    typedef Eigen::VectorXd Vector;

    void pre_solve(const mfem::SimState<DIM>* state) {
      attach_state(state);
      
      Eigen::SparseMatrixdRowMajor M = state->mesh_->mass_matrix();
      Vector ones = Vector::Ones(M.cols());
      Vector lumped = M * ones;
      Msqrtinv_ = lumped.array().rsqrt();
      Minv_ = lumped.array().inverse();

      // Set rhs system
      int size = 0;
      for (const auto& var : state_->mixed_vars_) {
        size += var->size();
      }
      rhs_.resize(size);

      Vector fp = Msqrtinv_ * -state->x_->gradient().array();

      int curr_row = 0;
      for (const auto& var : state->mixed_vars_) {
        Eigen::Ref<Vector> out = rhs_.segment(curr_row, var->size());

        // B - (n x (nele * N))
        Eigen::SparseMatrix<double> B;
        var->jacobian_x(B);

        // C - (nele*N x nele*N)
        Eigen::SparseMatrix<double> C;
        var->jacobian_mixed(C);

        // D^2 - (nele*N x nele*N)
        Eigen::SparseMatrix<double> H;
        var->hessian_inv(H);

        // G = B * M^{-1/2}
        Eigen::SparseMatrix<double> G = Vector(Msqrtinv_).asDiagonal() * B;
        lhs_ = G.transpose() * G + C*C*H;

        // fp = M^{-1/2} bx
        // G*fp = B * M^{-1} * bx  
        rhs_ = B.transpose() * (Minv_.asDiagonal() * -state->x_->gradient());

        // fy = H^{-1/2} by
        // D*fy = C * H^{-1} * by
        rhs_ -= C * H * var->gradient_mixed();
        rhs_ += var->gradient_dual();
        curr_row += var->size();
        //saveMarket(lhs_, "lhs2.mkt");
        //saveMarket(rhs_, "rhs2.mkt");
      }
      assert(rhs_.size() == curr_row);
    }

    void post_solve(const mfem::SimState<DIM>* state, Eigen::VectorXd& dx)
    {
      state->x_->delta().setZero();

      Vector fp = Msqrtinv_ * -state->x_->gradient().array();

      int curr_row = 0;
      for (auto& var : state->mixed_vars_) {
        const Vector& la = dx.segment(curr_row, var->size());
        var->lambda() = la;

        // fp = M^{-1/2} bx , G = B * M^{-1/2}
        // p = fp - G' * la = M^{-1/2} (bx - B' * la)
        // dx = (M^{-1/2}p) = M^{-1} (bx - B'*la)
        var->product_jacobian_x(la,state->x_->delta(),true);
        state->x_->delta() = (Minv_.array() 
            * (-state->x_->gradient() - state->x_->delta()).array());
        
        // fy = H^{-1/2} by , D = C * H^{-1/2}
        // y = fy - D' * la = H^{-1/2} (bs - C * la)
        // ds = (H^{-1/2}p) = H^{-1} (bs - C*la)
        var->delta().setZero();
        Vector tmp = Vector::Zero(var->size());
        var->product_jacobian_mixed(la, tmp);
        tmp = -var->gradient_mixed() - tmp;
        var->product_hessian_inv(tmp, var->delta());
        curr_row += var->size();
        //saveMarket(state->x_->delta(), "x_sub.mkt");
        //saveMarket(var->delta(), "s_sub.mkt");
        //saveMarket(var->lambda(), "la_sub.mkt");
      }
    }

    const MatrixType& A() const {
      return lhs_;
    }

    const Vector& b() const {
      return rhs_;
    }
   
    DualCondensedSystem() : state_(nullptr) {}
   
    const mfem::SimState<DIM>& state() const { return *state_; }

    void attach_state(const mfem::SimState<DIM>* state) {
      state_ = state;
    }

  private:

    const mfem::SimState<DIM>* state_;

    Eigen::ArrayXd Msqrtinv_;

    // linear system right and left hand side
    MatrixType lhs_;
    Vector rhs_;       
    Vector Minv_;
  };
}