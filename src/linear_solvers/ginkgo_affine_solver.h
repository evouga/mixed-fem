#pragma once

#include <ginkgo/ginkgo.hpp>
#include "simulation_state.h"
#include "mesh/meshes.h"

namespace mfem {

  template<typename Scalar, int DIM>
  class AffineSolver 
      : public gko::EnableLinOp<AffineSolver<Scalar,DIM>>,
        public gko::EnableCreateMethod<AffineSolver<Scalar,DIM>> {

  public:
    using vec = gko::matrix::Dense<Scalar>;
    using State = SimState<DIM,STORAGE_THRUST>;
    using Matrix = GkoFemMatrix<Scalar,DIM>;

    AffineSolver(std::shared_ptr<const gko::Executor> exec,
        gko::dim<2> size={}, State* state=nullptr, Matrix* A=nullptr)
        : gko::EnableLinOp<AffineSolver<Scalar,DIM>>(exec, size), state_(state),
        A_(A)
    {
      one_ = gko::initialize<vec>({1.0}, exec);
      neg_one_ = gko::initialize<vec>({-1.0}, exec);

      if (state != nullptr) {
        tmp_1_ = vec::create(exec, gko::dim<2>{12, 1});
        tmp_2_ = vec::create(exec, gko::dim<2>{12, 1});
        x_tmp_ = vec::create(exec, gko::dim<2>{size[0], 1});
        x_tmp2_ = vec::create(exec, gko::dim<2>{size[0], 1});
        r_ = vec::create(exec, gko::dim<2>{size[0], 1});

        // Compute affine basis matrices
        using namespace Eigen;

        MatrixXd& V = state->mesh_->Vref_;

        // Cast state mesh to meshes datatype
        const auto& mesh = state->mesh_;
        const Meshes* meshes = dynamic_cast<const Meshes*>(mesh.get());
        if (meshes == nullptr) {
          std::cout << "Error: AffineSolver"
              "only works with Meshes type" << std::endl;
          exit(1);
        }

        num_partitions_ = meshes->meshes().size();
        R_h_.resize(num_partitions_);
        R2_h_.resize(num_partitions_);
        E_h_.resize(num_partitions_);
        Einv_h_.resize(num_partitions_);

        // initialize basis matrices for each partition
        size_t sz_V = 0;
        for (int i = 0; i < num_partitions_; ++i) {
          const auto& mesh = meshes->meshes()[i];
          R_h_[i].resize(meshes->Vref_.size(), 12);
          R_h_[i].setZero();
          std::cout << "size free for partition " << i << ": " << mesh->projection_matrix().rows() << std::endl;
          // Compute center of mass
          Matrix3d I;
          Vector3d c;
          double mass = 0;
          sim::rigid_inertia_com(I, c, mass, mesh->Vref_, mesh->F_, 1.0);
          for (int j = 0; j < mesh->Vref_.rows(); ++j) {
            auto I = Matrix3d::Identity();
            R_h_[i].block<3,3>(3*(j+sz_V), 0) = I*(mesh->Vref_(j,0) - c(0));
            R_h_[i].block<3,3>(3*(j+sz_V), 3) = I*(mesh->Vref_(j,1) - c(1));
            R_h_[i].block<3,3>(3*(j+sz_V), 6) = I*(mesh->Vref_(j,2) - c(2));
            R_h_[i].block<3,3>(3*(j+sz_V), 9) = I;
          }

          const std::vector<int>& free_map = state->mesh_->free_map_;

          // node span for this partition
          int min_idx = -1;
          int max_idx = -1;
          int num_free = 0;
          for (int j = 0; j < mesh->Vref_.rows(); ++j) {
            if (free_map[j+sz_V] == -1) continue;
            if (min_idx == -1) min_idx = free_map[j+sz_V];
            max_idx = free_map[j+sz_V];
            num_free++;
          }

          R2_h_[i].resize(3*num_free, 12);
          int curr = 0;
          for (int j = 0; j < mesh->Vref_.rows(); ++j) {
            if (free_map[j+sz_V] == -1) continue;
            R2_h_[i].block<3,12>(3*curr, 0) = R_h_[i].block<3,12>(3*(j+sz_V), 0);
            curr++;
          }
          min_idx = 3*min_idx;
          max_idx = 3*(max_idx+1);
          std::cout << "R2_h size: " << R2_h_[i].rows() << " x " << R2_h_[i].cols() << std::endl;
          std::cout << " curr: " << curr << std::endl;
          std::cout << " min_idx: " << min_idx << std::endl;
          std::cout << " max_idx: " << max_idx << std::endl;
          partition_bounds_.push_back(std::make_pair(min_idx, max_idx));
          sz_V += mesh->Vref_.rows();
        }

        // Project out dirichlet BCs and copy to ginkgo dense matrix
        for (int i = 0; i < R_h_.size(); ++i) {
          E_.push_back(vec::create(exec, gko::dim<2>{12, 12}));
          E2_.push_back(vec::create(exec, gko::dim<2>{12, 12}));
          Einv_.push_back(vec::create(exec, gko::dim<2>{12, 12}));

          R_h_[i] = state->mesh_->projection_matrix() * R_h_[i];

          // Make row-major before copying to ginkgo
          R_h_[i].transposeInPlace();

          R_.push_back(
              vec::create(exec, gko::dim<2>{R_h_[i].cols(), R_h_[i].rows()}));
          
          double* ptr = R_h_[i].data();
          exec->copy_from(exec->get_master().get(), R_h_[i].size(), ptr, R_[i]->get_values());

          RT_.push_back(gko::as<vec>(R_[i]->transpose()));

          R2_h_[i].transposeInPlace();
          R2_.push_back(
              vec::create(exec, gko::dim<2>{R2_h_[i].cols(), R2_h_[i].rows()}));
          ptr = R2_h_[i].data();
          exec->copy_from(exec->get_master().get(), R2_h_[i].size(), ptr, R2_[i]->get_values());
          RT2_.push_back(gko::as<vec>(R2_[i]->transpose()));

          // Compute E_M = RT * M * R, which is fixed throughout the simulation
          auto& M = state->mesh_->mass_matrix();
          E_M_h_.push_back(R_h_[i] * M * R_h_[i].transpose());
          // std::cout << "E_M_h size: \n" << E_M_h_[i].rows() << " x " << E_M_h_[i].cols() << std::endl;
          // std::cout << " E _M _ H: " << E_M_h_[i] << std::endl;
        }

      }
    }

  protected:


    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {
      auto dense_b = gko::as<vec>(b);
      auto dense_x = gko::as<vec>(x);

      // Compute R^T * A * R per partition
      int sz = dense_b->get_size()[0];
      // auto tmp = vec::create(this->get_executor(), gko::dim<2>{sz, 12});

      OptimizerData::get().timer.start("1", "ABD");

      for (int i = 0; i < num_partitions_; ++i) {

        // A_->apply(lend(R_[i]), lend(tmp));
        // RT_[i]->apply(lend(tmp), lend(E2_[i]));


        int sz = partition_bounds_[i].second - partition_bounds_[i].first;
        auto tmp_i = vec::create(this->get_executor(), gko::dim<2>{sz, 12});

        // First compute A * R (N x 12 matrix)
        for (auto& var : state_->mixed_vars_) {
          auto stretch_var = dynamic_cast<MixedStretchGpu<DIM,STORAGE_THRUST>*>(
              var.get());
          if (stretch_var != nullptr) {
            stretch_var->apply_submatrix(
                tmp_i->get_values(), 
                R2_[i]->get_values(), 12,
                partition_bounds_[i].first, partition_bounds_[i].second);
            // x->add_scaled(one, lend(x_tmp));
          }

          // Note: this wont work if we reorder stretch and collision in the mixed_vars_ vector
          auto collision_var = dynamic_cast<
              MixedCollisionGpu<DIM,STORAGE_THRUST>*>(var.get());
          if (collision_var != nullptr) {
            if (collision_var->size() > 0) {
              auto tmp_c = vec::create(this->get_executor(), gko::dim<2>{sz, 12});
              collision_var->apply_submatrix(
                  tmp_c->get_values(), 
                  R2_[i]->get_values(), 12,
                  partition_bounds_[i].first, partition_bounds_[i].second);
              tmp_i->add_scaled(lend(one_), lend(tmp_c));
            }
          }
        }
        // A_->apply(lend(R_[i]), lend(tmp));

        // Compute R^T * A * R (12 x 12 matrix)
        RT2_[i]->apply(lend(tmp_i), lend(E_[i]));
      }
      OptimizerData::get().timer.stop("1", "ABD");


      // Compute residual r = b - A * x
      r_->copy_from(lend(dense_b));
      A_->apply(lend(neg_one_), lend(dense_x), lend(one_), lend(r_));

      auto master = this->get_executor()->get_master();

      OptimizerData::get().timer.start("2", "ABD");

      // Compute per partition inverse of E on the CPU
      for (int i = 0; i < num_partitions_; ++i) {
        // Copy from device to host
        master->copy_from(this->get_executor().get(),
            E_h_[i].size(), E_[i]->get_values(), E_h_[i].data());

        // std::cout << "E_h_[" << i << "] : " << std::endl;
        // std::cout << (E_M_h_[i] + E_h_[i]) << std::endl;
        // std::cout << "E_M_h_[" << i << "] : " << std::endl;

        // E2
        // Eigen::Matrix12d E2;
        // master->copy_from(this->get_executor().get(),
        //     E2.size(), E2_[i]->get_values(), E2.data());
        // std::cout << "E2[" << i << "] : " << std::endl;
        // std::cout << E2 << std::endl;
        // std::cout << " diff " << (E_M_h_[i] + E_h_[i] - E2).norm() << std::endl;

        // Compute inverse of E
        Einv_h_[i] = (E_M_h_[i] + E_h_[i]).inverse();

        // Copy back to device
        this->get_executor()->copy_from(master.get(),
            Einv_h_[i].size(), Einv_h_[i].data(),
            Einv_[i]->get_values());
      }
      OptimizerData::get().timer.stop("2", "ABD");
      OptimizerData::get().timer.start("3", "ABD");


      // Compute per-partition update to x
      for (int i = 0; i < num_partitions_; ++i) {
        // Compute tmp_1_ = R^T * r
        RT_[i]->apply(lend(r_), lend(tmp_1_));

        // Compute Einv = E^-1
        // Einv_[i]->copy_from(lend(E_[i]));
        // Einv_[i]->compute_inverse();
        Einv_[i]->apply(lend(tmp_1_), lend(tmp_2_));

        // Compute R * x_affine (tmp_2_)
        R_[i]->apply(lend(tmp_2_), lend(x_tmp_));

        // Update x
        dense_x->add_scaled(lend(one_), lend(x_tmp_));
      }
      OptimizerData::get().timer.stop("3", "ABD");

      for (auto& var : state_->mixed_vars_) {
        auto stretch_var = dynamic_cast<MixedStretchGpu<DIM,STORAGE_THRUST>*>(
            var.get());
        if (stretch_var != nullptr) {
          stretch_var->free_matrix();
        }
      }


      // get dense x norm
      auto norm = gko::initialize<vec>({1.0}, this->get_executor());
      dense_x->compute_norm2(lend(norm));
      std::cout << "x norm : " << this->get_executor()->copy_val_to_host(norm->get_const_values()) << std::endl;
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override {
      auto dense_b = gko::as<vec>(b);
      auto dense_x = gko::as<vec>(x);
      // auto tmp_x = dense_x->clone();
      this->apply_impl(b, lend(x_tmp2_));
      dense_x->scale(beta);
      dense_x->add_scaled(alpha, lend(x_tmp2_));
    }
  private:
    State* state_;
    std::shared_ptr<vec> one_;
    std::shared_ptr<vec> neg_one_;
    std::shared_ptr<vec> r_;
    std::shared_ptr<vec> tmp_1_;
    std::shared_ptr<vec> tmp_2_;
    std::shared_ptr<vec> x_tmp2_;
    std::shared_ptr<vec> x_tmp_;

    int num_partitions_;

     // per-partition basis
    std::vector<Eigen::MatrixXd> R_h_;
    std::vector<Eigen::MatrixXd> R2_h_;
    std::vector<std::shared_ptr<vec>> E_; // R^T * A * R
    std::vector<std::shared_ptr<vec>> E2_; // R^T * A * R
    std::vector<std::shared_ptr<vec>> Einv_;
    mutable std::vector<Eigen::Matrix12d> Einv_h_;
    mutable std::vector<Eigen::Matrix12d> E_h_;
    std::vector<Eigen::Matrix12d> E_M_h_;
    std::vector<std::shared_ptr<vec>> R_;
    std::vector<std::shared_ptr<vec>> RT_;
    std::vector<std::shared_ptr<vec>> R2_;
    std::vector<std::shared_ptr<vec>> RT2_;
    std::vector<std::pair<int, int>> partition_bounds_;
    Matrix* A_;
  };
}