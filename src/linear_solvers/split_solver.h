#pragma once

#include "linear_solver.h"
#include "linear_system.h"
#include "preconditioners.h"

namespace mfem {

  template <typename Solver,typename Scalar, int DIM>
  class SplitSolver : public LinearSolver<Scalar, DIM> {

    typedef LinearSolver<Scalar, DIM> Base;

  public:

    SplitSolver(SimState<DIM>* state) : LinearSolver<Scalar,DIM>(state) {
      solver_.init(state);
    }

    void solve() override {
      system_matrix_.pre_solve(Base::state_);

      // Should be a no-op
      solver_.compute(system_matrix_.A());
      if (solver_.info() != Eigen::Success) {
       std::cerr << "prefactor failed! " << std::endl;
       exit(1);
      }

      // Apply preconditioner as solver
      tmp_ = solver_.solve(system_matrix_.b());

      std::cout << "error: " << (system_matrix_.A() * tmp_ - system_matrix_.b()).norm() / system_matrix_.b().norm() << std::endl;
      system_matrix_.post_solve(Base::state_, tmp_);

    }

  private:
    SystemMatrixIndefinite<Scalar,DIM> system_matrix_;
    Solver solver_;
    Eigen::VectorXx<Scalar> tmp_;
  };


}
