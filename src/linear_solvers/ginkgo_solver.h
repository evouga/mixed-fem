#pragma once

#include "linear_solver.h"
// #include <unsupported/Eigen/SparseExtra>
#include <thrust/device_vector.h>
// #include <ginkgo/ginkgo.hpp>
#include <ginkgo/ginkgo.hpp>

#include "utils/block_csr.h"

namespace mfem {

  template<typename Scalar, int DIM>
  class AssemblyFreeMatrix 
      : public gko::EnableLinOp<AssemblyFreeMatrix<Scalar,DIM>>,
        public gko::EnableCreateMethod<AssemblyFreeMatrix<Scalar,DIM>> {
  public:
    AssemblyFreeMatrix(std::shared_ptr<const gko::Executor> exec, size_t size)
        : gko::EnableLinOp<AssemblyFreeMatrix<Scalar,DIM>>(exec,
              gko::dim<DIM>{size}) {}

  protected:
    using vec = gko::matrix::Dense<Scalar>;
    using coef_type = gko::array<Scalar>;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {
      auto dense_b = gko::as<vec>(b);
      auto dense_x = gko::as<vec>(x);

      // functor to apply assembly free multiply
      struct multiply_operation : gko::Operation {
        multiply_operation(const coef_type& coefficients, const vec* b,
            vec* x) : coefficients{coefficients}, b{b}, x{x}
        {}

        void run(std::shared_ptr<const gko::CudaExecutor>) const override {
          // Get device pointers for input and output
          const Scalar* b_ptr = b->get_const_values();
          Scalar* x_ptr = x->get_values();

          // This is where the magic happens
            // stencil_kernel(x->get_size()[0], coefficients.get_const_data(),
            //                b->get_const_values(), x->get_values());
        }

        const coef_type& coefficients;
        const vec* b;
        vec* x;
      };
      this->get_executor()->run(
          multiply_operation(coefficients, dense_b, dense_x));
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
    coef_type coefficients;
  };

  template <typename SystemMatrix, typename Scalar, int DIM,
      StorageType STORAGE = STORAGE_EIGEN>
  class GinkgoSolver : public LinearSolver<Scalar,DIM,STORAGE> {

    typedef LinearSolver<Scalar, DIM, STORAGE> Base;
    using vec = gko::matrix::Dense<Scalar>;
    using mtx = gko::matrix::Csr<Scalar, int>;
    using solver_type = gko::solver::Cg<Scalar>;
    using preconditioner_type = gko::preconditioner::Jacobi<Scalar, int>;
    using val_array = gko::array<Scalar>;
    using idx_array = gko::array<int>;

  public:

    GinkgoSolver(SimState<DIM,STORAGE>* state)
        : LinearSolver<Scalar,DIM,STORAGE>(state),
        has_init_(false) {

      // SHARED PTR to executor
      auto master = gko::OmpExecutor::create();
      const auto exec = gko::CudaExecutor::create(0, master);
      const auto app_exec = exec->get_master();
       std::cout << "GinkgoSolver initialization done!" << std::endl;
    }

    void solve() override {
      system_matrix_.pre_solve(Base::state_);
        // saveMarket(system_matrix_.A(), "lhs.mkt");

      // 1. Get rhs double* from system, make array from that, and then rhs vector
      // 2. Get initial guess double* from state, make array from that, and then vector as well
      // 3. Create CG solver (in constructor), then here give it the LinOp, call apply on RHS and u
      // 4. post solve give double* solution to system
      // solver_.compute(system_matrix_.A());
      // if (solver_.info() != Eigen::Success) {
      //  std::cerr << "prefactor failed! " << std::endl;
      //  exit(1);
      // }
      // tmp_ = solver_.solve(system_matrix_.b());
      //std::cout << "rhs norm() : " << system_matrix_.b().norm() << std::endl;
      // system_matrix_.post_solve(Base::state_, tmp_);
    }

  private:
    SystemMatrix system_matrix_;
    
    // Solver solver_;
    bool has_init_;

  };


}
