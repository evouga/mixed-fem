#pragma once

#include "simulation_state.h"
#include "EigenTypes.h"
#include "variables/mixed_collision.h"
#include "variables/mixed_stretch.h"

namespace Eigen {
  
  // We have a system of the form
  // [M+K 0   Dx'][dx] = [gx]
  // [0   Hd  Ds'][dD] = [gd]
  // [Dx  Ds  0  ][dl] = [gl]
  //
  // For now let's say M+K is fixed and prefactored
  // H is the diagonal distance hessian, and Dx and Ds are the jacobians.
  // 
  // The equivalent quadratic form for this system is 
  // E(dx,dD,dl) = 0.5 * (
  //        dx^T M+K dx + dl Dx dx - dx^T gx 
  //      + dD^T Hd dD  + dl Ds dD - dD^T gd
  //      + dl^T Dx^T dx + dl^T Ds^T dD - dl^T gl)
  // 
  // So minimization takes the form
  // dx*, dD*, dl* = argmin_{dx,dD,dl} E(dx,dD,dl)
  //
  // In dual ascent we alternate between solving for each variable, so that
  // for the k+1-th iteration we have
  // dx_{k+1} = argmin_{dx} E(dx,dD_k,dl_k)
  // dD_{k+1} = argmin_{dD} E(dx_{k+1},dD,dl_k)
  //
  // We can solve for dx by setting the derivative of E with respect to dx to 0
  // and solving for dx. This gives
  // dx_{k+1} = (M+K)^{-1} (gx - Dx' dl_k) and for dD we get
  // dD_{k+1} = (Hd)^{-1} (gd - Ds' dl_k) and for dl we get
  // 
  // And for the multiplier update we use:
  // dl_{k+1} = dl_k - (Dx dx_{k+1} + Ds dD_{k+1}) - gl 
  // (gl is the current constraint violation so it needs to be here)
  template <typename Scalar, int DIM>
  class DualAscentPreconditioner {
      typedef Matrix<Scalar,Dynamic,1> Vector;
      typedef SparseMatrix<double, RowMajor> MatType;

    public:
      typedef typename Vector::StorageIndex StorageIndex;
      enum {
        ColsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic
      };
   
      DualAscentPreconditioner() : is_initialized_(false), state_(nullptr) {}
   
      EIGEN_CONSTEXPR Index rows() const EIGEN_NOEXCEPT { return state_->size(); }
      EIGEN_CONSTEXPR Index cols() const EIGEN_NOEXCEPT { return state_->size(); }

      void init(const mfem::SimState<DIM>* state) {
        if (state->mixed_vars_.size() == 0) {
          throw std::runtime_error("Using ARAP preconditioner without mixed vars");
        }

        for (int i = 0; i < state->mixed_vars_.size(); ++i) {
          if (const mfem::MixedStretch<DIM>* stretch = dynamic_cast<
              const mfem::MixedStretch<DIM>*>(state->mixed_vars_[i].get())) {
            stretch_ = stretch;
          }

          if (const mfem::MixedCollision<DIM>* collision = dynamic_cast<
              const mfem::MixedCollision<DIM>*>(state->mixed_vars_[i].get())) {
            collision_ = collision;
          }
        }

        if (stretch_ == nullptr) {
          throw std::runtime_error("Using DA without stretch");
        }
        if (collision_ == nullptr) {
          throw std::runtime_error("Using DA without collision");
        }

        // Conjure up some silly stiffness TODO use actual stiffness from maximum of material properties
        double ym = 1e6;
        double pr = 0.45;
        double mu = ym/(2.0*(1.0+pr));
        double k = state->config_->h * state->config_->h;
        k *= mu;

        SparseMatrixd Gx;
        stretch_->jacobian_x(Gx);

        // Gx has volume in it, so we need to divide by volume
        // to avoid double multiplying the laplacian
        const VectorXd& vols = state->mesh_->volumes();
        SparseMatrix<double, RowMajor> W;

        int N = std::pow(state->mesh_->V_.cols(),2);
        W.resize(N*vols.size(), N*vols.size());
 
        std::vector<Triplet<double>> trips;
        for (int i = 0; i < vols.size(); ++i) {
          for (int j = 0; j < N; ++j) {
            trips.push_back(Triplet<double>(N*i+j, N*i+j, 1.0 / vols(i)));
          }
        }
        W.setFromTriplets(trips.begin(),trips.end());

        // Form laplacian and compute inverse of M + k*L system
        L_ = Gx * W * Gx.transpose();
        solver_.compute(state->mesh_->mass_matrix() + k*L_);
        if (solver_.info() != Eigen::Success) {
          throw std::runtime_error("DualAscentPreconditioner factorization failed!");
        }

        std::cout << "post solve " << std::endl;
        is_initialized_ = true;
        state_ = state;
      }

      void rebuild_factorization() {
        // Conjure up some silly stiffness
        double k = state_->config_->h * state_->config_->h;
        double max_stress = stretch_->max_stresses().maxCoeff();
        k *= max_stress;

        solver_.compute(state_->mesh_->mass_matrix() + k*L_);
        if (solver_.info() != Eigen::Success) {
          throw std::runtime_error("DualAscentPreconditioner factorization"
              " failed!");
        }
      }
   
      DualAscentPreconditioner& analyzePattern(const MatType&) { return *this; }
      DualAscentPreconditioner& factorize(const MatType& mat) { return *this; }
      DualAscentPreconditioner& compute(const MatType& mat) {
        //  static int step = 5;
        //  if (step == 0) {
        //    rebuild_factorization();
        //  }
        //  step = (step + 1) % 10;

        return *this;
      }
   
      template<typename Rhs, typename Dest>
      void _solve_impl(const Rhs& b, Dest& x) const {

        // Build gx
        gx_ = state_->x_->rhs() + stretch_->rhs();

        // Build gd (collision gradient)
        gd_ = -collision_->gradient_mixed();
        
        gl_ = -collision_->gradient();



        // dx_{k+1} = (M+K)^{-1} (gx - Dx' dl_k) and for dD we get
        // dD_{k+1} = (Hd)^{-1} (gd - Ds' dl_k) and for dl we get
        // 
        // And for the multiplier update we use:
        // dl_{k+1} = dl_k - (Dx dx_{k+1} + Ds dD_{k+1}) - gl 


        x = solver_.solve(b);
      }
   
      template<typename Rhs>
      inline const Solve<DualAscentPreconditioner, Rhs>
      solve(const MatrixBase<Rhs>& b) const {
        eigen_assert(is_initialized_ && 
            "DualAscentPreconditioner is not initialized.");
        return Solve<DualAscentPreconditioner, Rhs>(*this, b.derived());
      }
   
      ComputationInfo info() { return Success; }
   
    protected:
      SimplicialLLT<MatType, Upper|Lower> solver_;
      const mfem::SimState<DIM>* state_;
      bool is_initialized_;
      MatType L_;
      Vector gx_;
      Vector gd_;
      Vector gl_;
      const mfem::MixedStretch<DIM>* stretch_ = nullptr;
      const mfem::MixedCollision<DIM>* collision_ = nullptr;
  };
}