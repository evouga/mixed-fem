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
        E_h_.resize(num_partitions_);
        Einv_h_.resize(num_partitions_);

        // initialize basis matrices for each partition
        size_t sz_V = 0;
        for (int i = 0; i < num_partitions_; ++i) {
          const auto& mesh = meshes->meshes()[i];
          R_h_[i].resize(meshes->Vref_.size(), 12);
          R_h_[i].setZero();

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
          // std::cout << "c : " << c.transpose() << std::endl;
          sz_V += mesh->Vref_.rows();
        }

        // Project out dirichlet BCs and copy to ginkgo dense matrix
        for (int i = 0; i < R_h_.size(); ++i) {
          E_.push_back(vec::create(exec, gko::dim<2>{12, 12}));
          Einv_.push_back(vec::create(exec, gko::dim<2>{12, 12}));

          R_h_[i] = state->mesh_->projection_matrix() * R_h_[i];

          // Make row-major before copying to ginkgo
          R_h_[i].transposeInPlace();

          R_.push_back(
              vec::create(exec, gko::dim<2>{R_h_[i].cols(), R_h_[i].rows()}));
          
          double* ptr = R_h_[i].data();
          exec->copy_from(exec->get_master().get(), R_h_[i].size(), ptr, R_[i]->get_values());

          RT_.push_back(gko::as<vec>(R_[i]->transpose()));
        }

      }
    }

  protected:

  

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {
      auto dense_b = gko::as<vec>(b);
      auto dense_x = gko::as<vec>(x);

      // Compute R^T * A * R per partition
      int sz = dense_b->get_size()[0];
      auto tmp = vec::create(this->get_executor(), gko::dim<2>{sz, 12});

      OptimizerData::get().timer.start("1", "ABD");

      for (int i = 0; i < num_partitions_; ++i) {
        // First compute A * R (N x 12 matrix)
        A_->apply(lend(R_[i]), lend(tmp));

        // Compute R^T * A * R (12 x 12 matrix)
        RT_[i]->apply(lend(tmp), lend(E_[i]));
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
        // std::cout << E_h_[i] << std::endl;

        // Compute inverse of E
        Einv_h_[i] = E_h_[i].inverse();

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
    std::vector<std::shared_ptr<vec>> E_; // R^T * A * R
    std::vector<std::shared_ptr<vec>> Einv_;
    mutable std::vector<Eigen::Matrix12d> Einv_h_;
    mutable std::vector<Eigen::Matrix12d> E_h_;
    std::vector<std::shared_ptr<vec>> R_;
    std::vector<std::shared_ptr<vec>> RT_;
    Matrix* A_;
  };
}