#pragma once

#include "simulation_state.h"
#include "EigenTypes.h"
#include "variables/mixed_stretch.h"

namespace Eigen {

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
   
      ArapPreconditioner& analyzePattern(const MatType&) {
        return *this;
      }
   
      ArapPreconditioner& factorize(const MatType& mat) {
        return *this;
      }
   
      ArapPreconditioner& compute(const MatType& mat) {
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
      MatType L_;
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
