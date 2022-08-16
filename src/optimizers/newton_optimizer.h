#pragma once

#include "optimizers/optimizer.h"
#include "linear_solvers/linear_solver.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

namespace mfem {

  // Mixed FEM Sequential Quadratic Program
  template <int DIM>
  class NewtonOptimizer : public Optimizer<DIM> {

    typedef Optimizer<DIM> Base;

  public:
    
    NewtonOptimizer(SimState<DIM>& state)
        : Optimizer<DIM>(state) {}

    static std::string name() {
      return "newton";
    }

    void step() override;
    void reset() override;

  private:

    using Base::state_;

    // Update gradients, LHS, RHS for a new configuration
    void update_system();

    void substep(double& decrement);

    // linear system left hand side
    Eigen::SparseMatrix<double, Eigen::RowMajor> lhs_; 

    // linear system right hand side
    Eigen::VectorXd rhs_;       

  };
}
