#pragma once

#include "simulation_state.h"
#include "EigenTypes.h"
#include "variables/mixed_stretch.h"
#include "utils/psd_fix.h"
#include "igl/partition.h"
#include "igl/facet_components.h"
#include "mesh/meshes.h"
#include "rigid_inertia_com.h"

namespace Eigen {

  template <typename Scalar, int DIM>
  class DeflatedBlockJacobiPreconditioner {
      typedef Matrix<Scalar,Dynamic,1> Vector;
      typedef SparseMatrix<double, RowMajor> MatType;
      typedef Matrix<Scalar,12,1> Vector12;
      typedef Matrix<Scalar,12,12> Matrix12;

    public:
      typedef typename Vector::StorageIndex StorageIndex;
      enum {
        ColsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic
      };
   
      DeflatedBlockJacobiPreconditioner() : is_initialized_(false),
          state_(nullptr), num_partitions_(2) {}
   
      EIGEN_CONSTEXPR Index rows() const EIGEN_NOEXCEPT { return state_->size(); }
      EIGEN_CONSTEXPR Index cols() const EIGEN_NOEXCEPT { return state_->size(); }

      void init(const mfem::SimState<DIM>* state) {
        is_initialized_ = true;
        state_ = state;
        MatrixXd& V = state_->mesh_->Vref_;

        // Cast state mesh to meshes datatype
        const auto& mesh = state_->mesh_;
        const mfem::Meshes* meshes = dynamic_cast<const mfem::Meshes*>(mesh.get());
        if (meshes == nullptr) {
          std::cout << "Error: DeflatedBlockJacobiPreconditioner"
              "only works with Meshes" << std::endl;
          exit(1);
        }

        // initialize basis matrices for each partition
        num_partitions1_ = meshes-> meshes().size();
        W1_.resize(num_partitions1_);
        Winv1_.resize(num_partitions1_);

        size_t sz_V = 0;
        for (int i = 0; i < num_partitions1_; ++i) {
          const auto& mesh = meshes->meshes()[i];
          W1_[i].resize(meshes->Vref_.size(), 12);
          W1_[i].setZero();

          // Compute center of mass
          Matrix3d I;
          Vector3d c;
          double mass = 0;
          sim::rigid_inertia_com(I, c, mass, mesh->Vref_, mesh->F_, 1.0);
          for (int j = 0; j < mesh->Vref_.rows(); ++j) {
            W1_[i].template block<3,3>(3*(j+sz_V), 0) = Matrix3d::Identity()*(mesh->Vref_(j,0) - c(0));
            W1_[i].template block<3,3>(3*(j+sz_V), 3) = Matrix3d::Identity()*(mesh->Vref_(j,1) - c(1));
            W1_[i].template block<3,3>(3*(j+sz_V), 6) = Matrix3d::Identity()*(mesh->Vref_(j,2) - c(2));
            W1_[i].template block<3,3>(3*(j+sz_V), 9) = Matrix3d::Identity();
          }
          std::cout << "c : " << c.transpose() << std::endl;
          sz_V += mesh->Vref_.rows();
        }

        // Project out dirichlet BCs
        for (int i = 0; i < W1_.size(); ++i) {
          W1_[i] = state->mesh_->projection_matrix() * W1_[i];
        }
        std::cout << "num partitions 1" << std::endl;

        //---------------------------------------------------------------------

        num_partitions_ = mesh->partition_ids_.maxCoeff() + 1;
        std::cout << "Num partitions: " << num_partitions_ << std::endl;
        W_.resize(num_partitions_);
        Winv_.resize(num_partitions_);

        MatrixXd c(num_partitions_, DIM); // partition centroids
        c.setZero();
        const VectorXi& G = mesh->partition_ids_;
        std::vector<int> partition_sizes(num_partitions_, 0);
        // compute centroids
        for (int i = 0; i < V.rows(); ++i) {
          c.row(G(i)) += V.row(i);
          ++partition_sizes[G(i)];
        }

        // normalize centroids
        for (int i = 0; i < num_partitions_; ++i) {
          c.row(i) /= partition_sizes[i];
        }

        for (int i = 0; i < num_partitions_; ++i) {
          //Wi.resize(3*partition_sizes[i], 12);
          W_[i].resize(V.size(), 12);
          W_[i].setZero();
          // partition_vmap[i].resize(partition_sizes[i], 0);
        }

        // compute basis for each partition
        std::vector<int> partition_indices(num_partitions_, 0);

        for (int i = 0; i < V.rows(); ++i) {
          int g = G(i);
          // int ii = 3*partition_indices[g];
          W_[g].template block<3,3>(3*i, 0) = Matrix3d::Identity()*(V(i,0) - c(g,0));
          W_[g].template block<3,3>(3*i, 3) = Matrix3d::Identity()*(V(i,1) - c(g,1));
          W_[g].template block<3,3>(3*i, 6) = Matrix3d::Identity()*(V(i,2) - c(g,2));
          W_[g].template block<3,3>(3*i, 9) = Matrix3d::Identity();
          // partition_vmap[g][partition_indices[g]] = i;
          ++partition_indices[g];
        }
        // Project out dirichlet BCs
        for (int i = 0; i < W_.size(); ++i) {
          W_[i] = state->mesh_->projection_matrix() * W_[i];
        }
      }
   
      DeflatedBlockJacobiPreconditioner& analyzePattern(const MatType&) {
        return *this;
      }
   
      DeflatedBlockJacobiPreconditioner& factorize(const MatType& mat) {
        return *this;
      }
   
      DeflatedBlockJacobiPreconditioner& compute(const MatType& mat) {
        A_ = mat;
        // Compute coarse space inverses
        #pragma omp parallel for
        for (int i = 0; i < num_partitions_; ++i) {
          Matrix12 WKW = W_[i].transpose() * A_ * W_[i];
          Winv_[i] = WKW.inverse();
        }
        for (int i = 0; i < num_partitions1_; ++i) {
          Matrix12 WKW = W1_[i].transpose() * A_ * W1_[i];
          Winv1_[i] = WKW.inverse();
        }
        return *this;
      }

      void guess(const Vector& f, Vector& x0) {
        Vector Au = A_ * x0;

        // Solve at first level
        for (int i =0; i < num_partitions1_; ++i) {
          x0 += W1_[i] * (Winv1_[i] * (W1_[i].transpose() * (f - Au)));          
          // Matrix12 WKW = W1_[i].transpose() * A_ * W1_[i];
          // x0 += W1_[i] * (WKW.lu().solve((W1_[i].transpose() * (f - Au))));
        }
        // Update Au
        Au = A_ * x0;

        // Solve for each partition
        for (int i = 0; i < num_partitions_; ++i) {
          x0 += W_[i] * (Winv_[i] * (W_[i].transpose() * (f - Au)));

          // std::cout << "Affine residual: " << (Winv_[i] * (W_[i].transpose() * (f - Au))) << std::endl;
          // x0 -= W_[i] * (Winv_[i] * (W_[i].transpose() * Au));
          //x0 -= W_[i] * (Winv_[i] * (W_[i].transpose() * x0));
        }
      }
   
      template<typename Rhs, typename Dest>
      void _solve_impl(const Rhs& b, Dest& x) const {

        int N = b.size() / DIM;
        x.resize(b.size());

        // Apply block-jacobi preconditioner
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
          Matrix<Scalar,DIM,DIM> Aii = A_.template block(i*DIM,i*DIM,DIM,DIM);
          const Matrix<Scalar, DIM, 1>& bi = b.template segment<DIM>(i*DIM);
          Ref<Matrix<Scalar, DIM, 1>> xi = x.template segment<DIM>(DIM*i);
          xi = Aii.inverse() * bi;
        }

        // Vector Ay = A_ * x;

        // Apply deflation
        for (int i = 0; i < num_partitions_; ++i) {
          // x += W_[i] * (Winv_[i] * (W_[i].transpose() * (b - Ay)));
          // std::cout << "Affine residual: " << (Winv_[i] * (W_[i].transpose() * (b - Ay))) << std::endl;
          // x += W_[i] * (Winv_[i] * (W_[i].transpose() * b));
        }
      }
   
      template<typename Rhs>
      inline const Solve<DeflatedBlockJacobiPreconditioner, Rhs>
      solve(const MatrixBase<Rhs>& b) const {
        eigen_assert(is_initialized_ && 
            "DeflatedBlockJacobiPreconditioner is not initialized.");
        return Solve<DeflatedBlockJacobiPreconditioner, Rhs>(*this, b.derived());
      }
   
      ComputationInfo info() { return Success; }
   
    protected:
      MatType A_;
      const mfem::SimState<DIM>* state_;
      bool is_initialized_;
      int num_partitions_;
      std::vector<Matrix<Scalar,Dynamic,12>> W_; // per-partition basis
      std::vector<Matrix<Scalar,Dynamic,12>> Winv_; // per-partition basis

      int num_partitions1_;
      std::vector<Matrix<Scalar,Dynamic,12>> W1_; // per-partition basis
      std::vector<Matrix<Scalar,Dynamic,12>> Winv1_; // per-partition basis
  };
}
