#pragma once

#include "linear_solver.h"
// #include <thrust/device_vector.h>
#include <ginkgo/ginkgo.hpp>
#include "ginkgo_matrix.h"

#include "utils/block_csr.h"
#include "ginkgo_affine_solver.h"

namespace mfem {

  template <typename Solver, 
      typename Scalar, int DIM, StorageType STORAGE = STORAGE_THRUST>
  class GinkgoSolver : public LinearSolver<Scalar,DIM,STORAGE> {

    typedef LinearSolver<Scalar, DIM, STORAGE> Base;
    using vec = gko::matrix::Dense<Scalar>;
    using preconditioner_type = gko::preconditioner::Jacobi<Scalar, int>;
    // using cg = gko::solver::Cg<Scalar>;
    // using solver = gko::solver::Minres<Scalar>;
    using bj = GkoBlockJacobiPreconditioner<Scalar,DIM>;
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

      const double reduction_factor{state->config_->itr_tol};

      residual_criterion_ =
          ResidualCriterionFactory::create()
              .with_reduction_factor(reduction_factor)
              .on(cuda_exec_);
      // residual_criterion->add_logger(stream_logger);

      // Create preconditioner and save as linop
      precond_ = GkoBlockJacobiPreconditioner<Scalar,DIM>::create(cuda_exec_,
          gko::dim<2>{size}, Base::state_);

      fem_matrix_ = GkoFemMatrix<Scalar,DIM>::create(cuda_exec_,
                gko::dim<2>{size}, Base::state_);

      solver_ = Solver::build()
          .with_criteria(
              residual_criterion_,
              gko::stop::Iteration::build()
              .with_max_iters(state->config_->max_iterative_solver_iters)
              .on(cuda_exec_))
          .on(cuda_exec_)
          ->generate(fem_matrix_);
      solver_->set_preconditioner(precond_);

      abd_ = AffineSolver<Scalar,DIM>::create(cuda_exec_,
          gko::dim<2>{size}, Base::state_, fem_matrix_.get());
    }

    void solve() override {
      system_matrix_.pre_solve(Base::state_);
      size_t size = Base::state_->x_->value().size();
      precond_->update();
      auto& thrust_b = system_matrix_.b();
      const Scalar* b_ptr = thrust::raw_pointer_cast(thrust_b.data());
      Scalar* x_ptr = x_->get_values();
      x_->fill(0.0); // initialize guess to 0

      // Wrap rhs data in a gko vector
      auto b = gko::matrix::Dense<double>::create_const(cuda_exec_,
          gko::dim<2>{size,1},
          gko::array<double>::const_view(cuda_exec_, size, b_ptr), 1);

      // Apply explicit predictor guess
      if (Base::state_->config_->itr_explicit_guess) {
        Base::state_->x_->explicit_predictor(x_ptr, true);
        auto neg_one = gko::initialize<vec>({-gko::one<double>()}, cuda_exec_);
        x_->scale(lend(neg_one));
      }

      if (Base::state_->config_->itr_abd_guess) {
        OptimizerData::get().timer.start("ABD", "GKO");
        abd_->apply(lend(b), lend(x_));
        OptimizerData::get().timer.stop("ABD", "GKO");

      }
      
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
    std::shared_ptr<Solver> solver_;
    std::shared_ptr<bj> precond_;
    std::shared_ptr<AffineSolver<Scalar,DIM>> abd_;
    std::shared_ptr<GkoFemMatrix<Scalar,DIM>> fem_matrix_;

  };


}
