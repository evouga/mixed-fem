#pragma once

#include "simulation_state.h"
#include "EigenTypes.h"
#include "variables/mixed_stretch.h"

namespace Eigen {

  template <typename Scalar, int DIM>
  class LaplacianPreconditioner {
      typedef Matrix<Scalar,Dynamic,1> Vector;
      typedef SparseMatrix<double, RowMajor> MatType;

    public:
      typedef typename Vector::StorageIndex StorageIndex;
      enum {
        ColsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic
      };
   
      LaplacianPreconditioner() : is_initialized_(false), state_(nullptr) {}
   
      EIGEN_CONSTEXPR Index rows() const EIGEN_NOEXCEPT { return state_->size(); }
      EIGEN_CONSTEXPR Index cols() const EIGEN_NOEXCEPT { return state_->size(); }

      void init(const mfem::SimState<DIM>* state) {
        std::cout << "hey" << std::endl;

        if (state->mixed_vars_.size() == 0) {
          throw std::runtime_error("Using ARAP preconditioner without mixed vars");
        }

        double ym = 1e6;
        double pr = 0.45;
        double mu = ym/(2.0*(1.0+pr));
        double k = state->config_->h * state->config_->h;
        k *= mu;
        SparseMatrixd Gx;
        state->mixed_vars_[0]->jacobian_x(Gx);

        std::cout << "pre solve " << std::endl;
        // TODO double multiplying by areas currently. Make a volume diagonal matrix
        const VectorXd& vols = state->mesh_->volumes();
        SparseMatrix<double, RowMajor> W;

        std::cout << "pre solve " << std::endl;
        int N = std::pow(state->mesh_->V_.cols(),2);
        W.resize(N*vols.size(), N*vols.size());
 
        std::vector<Triplet<double>> trips;
        for (int i = 0; i < vols.size(); ++i) {
          for (int j = 0; j < N; ++j) {
            trips.push_back(Triplet<double>(N*i+j, N*i+j, 1.0 / vols(i)));
          }
        }
        std::cout << "pre solve " << std::endl;
        W.setFromTriplets(trips.begin(),trips.end());
        std::cout << "pre solve " << std::endl;
        L_ = Gx * W * Gx.transpose();
        std::cout << "pre solve " << std::endl;
        solver_.compute(state->mesh_->mass_matrix() + k*L_);
        if (solver_.info() != Eigen::Success) {
          throw std::runtime_error("ARAP preconditioner factorization failed!");
        }
        std::cout << "post solve " << std::endl;
        is_initialized_ = true;
        state_ = state;
      }

      void rebuild_factorization() {
        // Conjure up some silly stiffness
        double k = state_->config_->h * state_->config_->h;
        const mfem::MixedStretch<DIM>* c = dynamic_cast<
            const mfem::MixedStretch<DIM>*>(state_->mixed_vars_[0].get());
        if (c) {
          std::cout << "max stresses: " << c->max_stresses().size() << std::endl;
          double max_stress = c->max_stresses().maxCoeff();
          std::cout << "MAXIMUM stress" << max_stress << std::endl;
          k *= max_stress;
        } else {
          throw std::runtime_error("Using ARAP preconditioner without mixed-stretch");
        }
        solver_.compute(state_->mesh_->mass_matrix() + k*L_);
        if (solver_.info() != Eigen::Success) {
          throw std::runtime_error("ARAP preconditioner factorization failed!");
        }
      }
   
      LaplacianPreconditioner& analyzePattern(const MatType&) {
        return *this;
      }
   
      LaplacianPreconditioner& factorize(const MatType& mat) {
        return *this;
      }
   
      LaplacianPreconditioner& compute(const MatType& mat) {
         static int step = 5;
         if (step == 0) {
           std::cout << "factorize" << std::endl;
           rebuild_factorization();
         }
         step = (step + 1) % 10;
        return *this;
      }
   
      template<typename Rhs, typename Dest>
      void _solve_impl(const Rhs& b, Dest& x) const {
        x = solver_.solve(b);
      }
   
      template<typename Rhs>
      inline const Solve<LaplacianPreconditioner, Rhs>
      solve(const MatrixBase<Rhs>& b) const {
        eigen_assert(is_initialized_ && 
            "LaplacianPreconditioner is not initialized.");
        return Solve<LaplacianPreconditioner, Rhs>(*this, b.derived());
      }
   
      ComputationInfo info() { return Success; }
   
    protected:
      SimplicialLLT<MatType, Upper|Lower> solver_;
      const mfem::SimState<DIM>* state_;
      bool is_initialized_;
      MatType L_;
  };
}
