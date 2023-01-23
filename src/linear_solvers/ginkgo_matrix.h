#pragma once

#include <ginkgo/ginkgo.hpp>
#include "variables/mixed_stretch_gpu.h"
#include "simulation_state.h"
#include <thrust/device_vector.h>

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

        // Cast mixed variable to stretch gpu variable
        auto stretch_var = dynamic_cast<MixedStretchGpu<DIM,STORAGE_THRUST>*>(
            state->mixed_vars_[0].get());
        if (stretch_var != nullptr) {
          stretch_var->set_executor(exec);
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

          // TODO don't make this clone on every multiplication WTF!!@!KL!@J:!L@KJE:#OIJR
          Scalar* tmp_ptr = x_tmp->get_values();

          // First call apply on the position variable
          // which is just M * x
          state->x_->apply(x_ptr, b_ptr);

          // Call apply on each mixed var
          for (auto& var : state->mixed_vars_) {
            var->apply(tmp_ptr, b_ptr);
            x->add_scaled(one, lend(x_tmp));
          }
        }
        const vec* b;
        vec* x;
        vec* one;
        vec* x_tmp;
        State* state;
      }; 
      OptimizerData::get().timer.start("linsolv apply", "GKO");
      this->get_executor()->run(multiply_operation(dense_b, dense_x,
          one_.get(), x_tmp_.get(), state_));
      OptimizerData::get().timer.stop("linsolv apply", "GKO");
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override {
      auto dense_b = gko::as<vec>(b);
      auto dense_x = gko::as<vec>(x);
      auto tmp_x = dense_x->clone();
      this->apply_impl(b, lend(tmp_x));
      dense_x->scale(beta);
      dense_x->add_scaled(alpha, lend(tmp_x));
    }
  private:
    State* state_;
    std::shared_ptr<vec> one_;
    std::shared_ptr<vec> x_tmp_;

  };


  template<typename Scalar, int DIM>
  class GkoBlockJacobiPreconditioner 
      : public gko::EnableLinOp<
            GkoBlockJacobiPreconditioner<Scalar,DIM>>,
        public gko::EnableCreateMethod<
            GkoBlockJacobiPreconditioner<Scalar,DIM>> {
  public:

    using vec = gko::matrix::Dense<Scalar>;
    using State = SimState<DIM,STORAGE_THRUST>;

    static void apply_inv_diag(const double* diag, const double* b,
        double* x, int nrows);

    GkoBlockJacobiPreconditioner(std::shared_ptr<const gko::Executor> exec,
        gko::dim<2> size={}, State* state=nullptr)
        : gko::EnableLinOp<GkoBlockJacobiPreconditioner<Scalar,DIM>>(
            exec, size), state_(state)
    {
      one_ = gko::initialize<vec>({1.0}, exec);

      if (state != nullptr) {
        int nrows = size[0] / DIM;
        std::cout << "nrows: " << nrows << std::endl;
        diag_ = vec::create(exec, gko::dim<2>{nrows * DIM * DIM, 1});
        diag_ptr_ = diag_->get_values();
        update();
      }
    }

    void update() {
      OptimizerData::get().timer.start("preconditioner_update", "GKO");

      state_->x_->extract_diagonal(diag_ptr_);

      // Cast mixed variable to stretch gpu variable
      auto stretch_var = dynamic_cast<MixedStretchGpu<DIM,STORAGE_THRUST>*>(
          state_->mixed_vars_[0].get());
      if (stretch_var != nullptr) {
        std::cout << "extracting stretch diagonal" << std::endl;
        stretch_var->extract_diagonal(diag_ptr_);
      }
      OptimizerData::get().timer.stop("preconditioner_update", "GKO");

    }

  protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {
      auto dense_b = gko::as<vec>(b);
      auto dense_x = gko::as<vec>(x);

      // functor to apply assembly free multiply
      struct multiply_operation : gko::Operation {
        multiply_operation(const vec* b, vec* x, Scalar* diag)
            : b{b}, x{x}, diag{diag}
        {}
        // TODO compute inv separately
        void run(std::shared_ptr<const gko::CudaExecutor> exec) const override {
          // Get device pointers for input and output
          const Scalar* b_ptr = b->get_const_values();
          Scalar* x_ptr = x->get_values();
          apply_inv_diag(diag, b_ptr, x_ptr, b->get_size()[0] / DIM);
        }

        Scalar* diag;
        const vec* b;
        vec* x;
      };
      OptimizerData::get().timer.start("preconditioner", "GKO");
      this->get_executor()->run(multiply_operation(dense_b, dense_x, diag_ptr_));
      OptimizerData::get().timer.stop("preconditioner", "GKO");

    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override {
      auto dense_b = gko::as<vec>(b);
      auto dense_x = gko::as<vec>(x);
      auto tmp_x = dense_x->clone();
      this->apply_impl(b, lend(tmp_x));
      dense_x->scale(beta);
      dense_x->add_scaled(alpha, lend(tmp_x));
    }
  private:
    State* state_;
    std::shared_ptr<vec> one_;
    // thrust::device_vector<Scalar> diag_;
    std::shared_ptr<vec> diag_;
    Scalar* diag_ptr_; // just using thrust for memory management :p
  };
}