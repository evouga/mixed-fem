#pragma once

#include "linear_solver.h"
// #include <unsupported/Eigen/SparseExtra>
#include <thrust/device_vector.h>
// #include <ginkgo/ginkgo.hpp>
#include <ginkgo/ginkgo.hpp>

#include "utils/block_csr.h"
#include "variables/mixed_stretch_gpu.h"

namespace mfem {

  template<typename Scalar, int DIM>
  class MfemMatrix 
      : public gko::EnableLinOp<MfemMatrix<Scalar,DIM>>,
        public gko::EnableCreateMethod<MfemMatrix<Scalar,DIM>> {

    friend class gko::EnablePolymorphicObject<MfemMatrix<Scalar,DIM>, gko::LinOp>;
    friend class gko::EnableCreateMethod<MfemMatrix<Scalar,DIM>>;

  public:

    MfemMatrix(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size={},
        SimState<DIM,STORAGE_THRUST>* state=nullptr)
        : gko::EnableLinOp<MfemMatrix<Scalar,DIM>>(exec, size), state_(state) {
      one_ = gko::initialize<vec>({1.0}, exec);
      if (state == nullptr) {
        std::cout << "we're fucked. state is nullptr" << std::endl;
      } else {
        std::cout << "state is not nullptr" << std::endl;
        
        // Cast mixed variable to stretch gpu variable
        auto stretch_var = dynamic_cast<MixedStretchGpu<DIM,STORAGE_THRUST>*>(
            state->mixed_vars_[0].get());
        if (stretch_var != nullptr) {
          stretch_var->set_executor(exec);
        }
      }
    }

  protected:
    using vec = gko::matrix::Dense<Scalar>;
    using State = SimState<DIM,STORAGE_THRUST>;

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
        multiply_operation(const vec* b, vec* x, vec* one, State* state)
            : b{b}, x{x}, one{one}, state{state}
        {}

        void run(std::shared_ptr<const gko::CudaExecutor> exec) const override {
          // Get device pointers for input and output
          // auto one = gko::initialize<vec>({1.0}, exec);
          if (state == nullptr) {
            std::cout << "calling run with nullptr state!" << std::endl;
          }
          const Scalar* b_ptr = b->get_const_values();
          Scalar* x_ptr = x->get_values();

          auto tmp_x = x->clone();
          Scalar* tmp_ptr = tmp_x->get_values();

          // First call apply on the position variable
          // which is just M * x
          state->x_->apply(x_ptr, b_ptr);

          // Call apply on each mixed var
          for (auto& var : state->mixed_vars_) {
            var->apply(tmp_ptr, b_ptr);
            x->add_scaled(one, lend(tmp_x));
          }
        }
        const vec* b;
        vec* x;
        vec* one;
        State* state;
      };
      this->get_executor()->run(multiply_operation(dense_b, dense_x, one_.get(), state_));
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
    SimState<DIM,STORAGE_THRUST>* state_;
    std::shared_ptr<vec> one_;
  };

  template <typename Scalar, int DIM, StorageType STORAGE = STORAGE_THRUST>
  class GinkgoSolver : public LinearSolver<Scalar,DIM,STORAGE> {

    typedef LinearSolver<Scalar, DIM, STORAGE> Base;
    using vec = gko::matrix::Dense<Scalar>;
    using preconditioner_type = gko::preconditioner::Jacobi<Scalar, int>;
    using cg = gko::solver::Cg<Scalar>;
    using ResidualCriterionFactory =
        typename gko::stop::ResidualNorm<Scalar>::Factory;
  public:

    GinkgoSolver(SimState<DIM,STORAGE>* state)
        : LinearSolver<Scalar,DIM,STORAGE>(state),
        has_init_(false) {
      omp_exec_ = gko::OmpExecutor::create();
      cuda_exec_ = gko::CudaExecutor::create(0, omp_exec_);

      size_t size = Base::state_->x_->value().size();
      x_ = vec::create(cuda_exec_, gko::dim<2>(size, 1));


      std::shared_ptr<gko::log::Stream<Scalar>> stream_logger =
      gko::log::Stream<Scalar>::create(
          gko::log::Logger::polymorphic_object_events_mask,
          std::cout);
      cuda_exec_->add_logger(stream_logger);

      const double reduction_factor{1e-4};

      residual_criterion_ =
          ResidualCriterionFactory::create()
              .with_reduction_factor(reduction_factor)
              .on(cuda_exec_);
      // residual_criterion->add_logger(stream_logger);

      solver_ = cg::build()
          .with_criteria(
              residual_criterion_,
              gko::stop::Iteration::build().with_max_iters(1000u).on(cuda_exec_))
          .on(cuda_exec_)
          ->generate(MfemMatrix<Scalar,DIM>::create(cuda_exec_,
              gko::dim<2>{size}, Base::state_));

    }

    void solve() override {
      system_matrix_.pre_solve(Base::state_);
      size_t size = Base::state_->x_->value().size();

      auto& thrust_b = system_matrix_.b();
      const Scalar* b_ptr = thrust::raw_pointer_cast(thrust_b.data());
      Scalar* x_ptr = x_->get_values();
      x_->fill(0.0); // initialize guess to 0

      // Wrap rhs data in a gko vector
      auto b = gko::matrix::Dense<double>::create_const(cuda_exec_,
          gko::dim<2>{size,1},
          gko::array<double>::const_view(cuda_exec_, size, b_ptr), 1);
      
      // Solve for x
      solver_->apply(lend(b), lend(x_));

      // dx l2 norm
      auto res = gko::initialize<vec>({0.0}, cuda_exec_);
      // x->compute_norm2(lend(res));
      // std::cout << "x norm:\n";
      // write(std::cout, lend(res));

      // std::shared_ptr<gko::log::Record> record_logger = gko::log::Record::create(
      //     gko::log::Logger::executor_events_mask |
      //     gko::log::Logger::criterion_check_completed_mask);
      // cuda_exec_->add_logger(record_logger);
      // residual_criterion_->add_logger(record_logger);

      // auto residual =
      //     record_logger->get().criterion_check_completed.back()->residual.get();
      // auto residual_d = gko::as<vec>(residual);
      // // print_vector("Residual", residual_d);

      // res = gko::initialize<vec>({0.0}, cuda_exec_);
      // residual_d->compute_norm2(lend(res));
      // std::cout << "residual_d norm:\n";
      // write(std::cout, lend(res));

      system_matrix_.post_solve(Base::state_, x_ptr);
    }

  private:
    SystemMatrixThrustGpu<Scalar> system_matrix_;
    std::shared_ptr<gko::OmpExecutor> omp_exec_;
    std::shared_ptr<gko::CudaExecutor> cuda_exec_;
    std::shared_ptr<ResidualCriterionFactory> residual_criterion_;
    // Solver solver_;
    bool has_init_;
    std::shared_ptr<vec> x_;
    std::shared_ptr<cg> solver_;

  };


}
