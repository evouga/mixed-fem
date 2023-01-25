#pragma once

#include "optimizers/optimizer.h"
#include "linear_solvers/linear_solver.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

namespace mfem {

  // Mixed FEM Sequential Quadratic Program
  template <int DIM>
  class NewtonOptimizerGpu : public Optimizer<DIM,STORAGE_THRUST> {

    typedef Optimizer<DIM,STORAGE_THRUST> Base;

  public:
    
    NewtonOptimizerGpu(SimState<DIM,STORAGE_THRUST>& state)
        : Optimizer<DIM,STORAGE_THRUST>(state) {}

    static std::string name() {
      return "newton";
    }

    void step() override;
    void reset() override;

  private:

    using Base::state_;

    // Update gradients, LHS, RHS for a new configuration
    void update_system();

    // Linear solve
    void substep(double& decrement);

    std::unique_ptr<LinearSolver<double,DIM,STORAGE_THRUST>> linear_solver_;
    thrust::device_vector<double> x_full_;
    thrust::device_vector<double> x_;
  };
}
