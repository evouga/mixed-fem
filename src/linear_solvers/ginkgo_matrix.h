#pragma once

#include <ginkgo/ginkgo.hpp>
#include "variables/mixed_stretch_gpu.h"
#include "variables/mixed_collision_gpu.h"
#include "simulation_state.h"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include "utils/block_csr_apply.h"
// #include <thrust/execution_policy.h>

namespace mfem {

  template<typename Scalar, int DIM>
  class GkoFemMatrix 
      : public gko::EnableLinOp<GkoFemMatrix<Scalar,DIM>>,
        public gko::EnableCreateMethod<GkoFemMatrix<Scalar,DIM>> {

  public:
    using vec = gko::matrix::Dense<Scalar>;
    using State = SimState<DIM,STORAGE_THRUST>;

    GkoFemMatrix(std::shared_ptr<const gko::Executor> exec,
        gko::dim<2> size={}, State* state=nullptr)
        : gko::EnableLinOp<GkoFemMatrix<Scalar,DIM>>(exec, size), state_(state)
    {
      one_ = gko::initialize<vec>({1.0}, exec);

      if (state != nullptr) {
        x_tmp_ = vec::create(exec, gko::dim<2>{size[0], 1});
        x_tmp2_ = vec::create(exec, gko::dim<2>{size[0], 1});

        // Call apply on each mixed var
        for (auto& var : state->mixed_vars_) {
          auto stretch_var = dynamic_cast<MixedStretchGpu<DIM,STORAGE_THRUST>*>(
              var.get());
          if (stretch_var != nullptr) {
            stretch_var->set_executor(exec);
          }

          auto collision_var = dynamic_cast<
              MixedCollisionGpu<DIM,STORAGE_THRUST>*>(var.get());
          if (collision_var != nullptr) {
            collision_var->set_executor(exec);
          }
        }

      }
    }

  protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {
      auto dense_b = gko::as<vec>(b);
      auto dense_x = gko::as<vec>(x);

      // auto res = gko::initialize<vec>({0.0}, this->get_executor());
      // dense_b->compute_norm2(lend(res));
      // std::cout << "b norm :\n";
      // write(std::cout, lend(res));
      // res = gko::initialize<vec>({0.0}, this->get_executor());
      // dense_x->compute_norm2(lend(res));
      // std::cout << "x norm :\n";
      // write(std::cout, lend(res));

      // functor to apply assembly free multiply
      struct multiply_operation : gko::Operation {
        multiply_operation(const vec* b, vec* x, vec* one, vec* x_tmp, State* state)
            : b{b}, x{x}, one{one}, x_tmp{x_tmp}, state{state}
        {}

        void run(std::shared_ptr<const gko::CudaExecutor> exec) const override {
          // Get device pointers for input and output
          if (state == nullptr) {
            std::cout << "calling run with nullptr state!" << std::endl;
          }
          const Scalar* b_ptr = b->get_const_values();
          Scalar* x_ptr = x->get_values();

          // Get the number of columns of b
          int b_cols = b->get_size()[1];
          // std::cout << "b_cols: " << b_cols << std::endl;
          // std::cout << " x size: " << x->get_size()[0] << " " << x->get_size()[1] << std::endl;
          // std::cout << " b size: " << b->get_size()[0] << " " << b->get_size()[1] << std::endl;

          // TODO don't make this clone on every multiplication WTF!!@!KL!@J:!L@KJE:#OIJR
          Scalar* tmp_ptr = x_tmp->get_values();

          // First call apply on the position variable
          // which is just M * x
          state->x_->apply(x_ptr, b_ptr, b_cols);

          // Call apply on each mixed var
          for (auto& var : state->mixed_vars_) {
            // var->apply(tmp_ptr, b_ptr);

            auto stretch_var = dynamic_cast<MixedStretchGpu<DIM,STORAGE_THRUST>*>(
                var.get());
            if (stretch_var != nullptr) {
              bcsr_apply<DIM>(exec, stretch_var->assembler(), tmp_ptr, b_ptr,
                  b_cols);
              x->add_scaled(one, lend(x_tmp));
            }

            auto collision_var = dynamic_cast<
                MixedCollisionGpu<DIM,STORAGE_THRUST>*>(var.get());
            if (collision_var != nullptr) {
              if (collision_var->size() > 0) {
                bcsr_apply<DIM>(exec, collision_var->assembler(), tmp_ptr,
                    b_ptr, b_cols);
                x->add_scaled(one, lend(x_tmp));
              }
            }
          }
        }
        const vec* b;
        vec* x;
        vec* one;
        vec* x_tmp;
        State* state;
      }; 

      std::shared_ptr<vec> tmp = x_tmp_;
      
      // If tmp size is not the same as x, assign new vec to tmp
      // with the proper sizes.
      if (tmp->get_size()[0] != x->get_size()[0] ||
          tmp->get_size()[1] != x->get_size()[1]) {
        tmp = vec::create(this->get_executor(), x->get_size());
      }

      OptimizerData::get().timer.start("linsolv apply", "GKO");
      this->get_executor()->run(multiply_operation(dense_b, dense_x,
          one_.get(), tmp.get(), state_));
      OptimizerData::get().timer.stop("linsolv apply", "GKO");
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
    std::shared_ptr<vec> x_tmp_;
    std::shared_ptr<vec> x_tmp2_;

  };

  template<typename Scalar, int DIM>
  class GkoBlockJacobiPreconditioner 
      : public gko::EnableLinOp<
            GkoBlockJacobiPreconditioner<Scalar,DIM>>,
        public gko::EnableCreateMethod<
            GkoBlockJacobiPreconditioner<Scalar,DIM>> {
  public:

    using vec = gko::matrix::Dense<Scalar>;
    using array = gko::array<int>;
    using State = SimState<DIM,STORAGE_THRUST>;

    static void init_block_csr(int* row_offsets, int* col_indices, int nrows);

    static void compute_inv(double* diag, int nrows);

    static void apply_inv_diag(const double* diag, const double* b,
        double* x, int nrows);

    GkoBlockJacobiPreconditioner(std::shared_ptr<const gko::Executor> exec,
        gko::dim<2> size={}, State* state=nullptr)
        : gko::EnableLinOp<GkoBlockJacobiPreconditioner<Scalar,DIM>>(
            exec, size), state_(state)
    {
      if (state != nullptr) {
        nrows_ = size[0] / DIM;
        x_tmp_ = vec::create(exec, gko::dim<2>{size[0], 1});

        // diag_ = vec::create(exec, gko::dim<2>{nrows_ * DIM * DIM, 1});
        diag_ = gko::array<Scalar>(exec, nrows_ * DIM * DIM);
        col_indices_ = gko::array<int>(exec, nrows_);
        row_offsets_ = gko::array<int>(exec, nrows_ + 1);

        // // Initialize col_indices to sequence
        init_block_csr(row_offsets_.get_data(), col_indices_.get_data(), nrows_);
        diag_ptr_ = diag_.get_data();


        matrix_ = gko::matrix::Fbcsr<Scalar,int>::create_const(
            exec, gko::dim<2>{nrows_ * DIM, nrows_ * DIM}, DIM,
            diag_.as_const_view(), col_indices_.as_const_view(),
            row_offsets_.as_const_view());
            

            // std::move(col_indices_), std::move(row_offsets_),
            // std::move(diag_));

        update();
      }
    }

    void update() {
      OptimizerData::get().timer.start("preconditioner_update", "GKO");

      state_->x_->extract_diagonal(diag_ptr_);

      for (auto& var : state_->mixed_vars_) {

        auto stretch_var = dynamic_cast<MixedStretchGpu<DIM,STORAGE_THRUST>*>(
            var.get());
        if (stretch_var != nullptr) {
          // std::cout << "extracting stretch diagonal" << std::endl;
          stretch_var->assembler()->extract_diagonal(diag_ptr_);
        }

        auto collision_var = dynamic_cast<
            MixedCollisionGpu<DIM,STORAGE_THRUST>*>(var.get());
        if (collision_var != nullptr) {
          // std::cout << "extracting collision diagonal" << std::endl;
          if (collision_var->size() > 0) {
            collision_var->assembler()->extract_diagonal(diag_ptr_);
          }
        }
        //var->extract_diagonal(diag_ptr_);
      }
      compute_inv(diag_ptr_, nrows_);

      OptimizerData::get().timer.stop("preconditioner_update", "GKO");

    }

  protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {
      auto dense_b = gko::as<vec>(b);
      auto dense_x = gko::as<vec>(x);

      OptimizerData::get().timer.start("preconditioner", "GKO");
      matrix_->apply(lend(dense_b), lend(dense_x));
      OptimizerData::get().timer.stop("preconditioner", "GKO");
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override {
      auto dense_b = gko::as<vec>(b);
      auto dense_x = gko::as<vec>(x);
      // auto tmp_x = dense_x->clone();
      this->apply_impl(b, lend(x_tmp_));
      dense_x->scale(beta);
      dense_x->add_scaled(alpha, lend(x_tmp_));
    }
  private:
    State* state_;
    int nrows_;
    gko::array<Scalar> diag_;
    gko::array<int> col_indices_;
    gko::array<int> row_offsets_;
    Scalar* diag_ptr_; // just using thrust for memory management :p

    // Fbcsr matrix
    std::shared_ptr<const gko::matrix::Fbcsr<Scalar,int>> matrix_;
    std::shared_ptr<vec> x_tmp_;


  };
}