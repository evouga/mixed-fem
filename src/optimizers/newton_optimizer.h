#pragma once

#include "optimizers/optimizer.h"
#include "linear_solvers/linear_solver.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

namespace mfem {

  // Mixed FEM Sequential Quadratic Program
  template <int DIM>
  class NewtonOptimizer : public Optimizer<DIM,STORAGE_EIGEN> {

    typedef Optimizer<DIM,STORAGE_EIGEN> Base;

  public:
    
    NewtonOptimizer(SimState<DIM,STORAGE_EIGEN>& state)
        : Optimizer<DIM,STORAGE_EIGEN>(state) {}

    static std::string name() {
      return "newton";
    }

    void step() override;
    void reset() override;

    // Get the linear solver instance
    LinearSolver<double, DIM>* linear_solver() {
      return linear_solver_.get();
    }

  private:

    using Base::state_;

    // Update gradients, LHS, RHS for a new configuration
    void update_system();

    // Linear solve
    void substep(double& decrement);

    std::unique_ptr<LinearSolver<double, DIM>> linear_solver_;
  };
}
