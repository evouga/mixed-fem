#pragma once

#include "optimizers/mixed_sqp_optimizer.h"
#include "variables/stretch.h"
#include "variables/displacement.h"
#include "linear_solvers/linear_solver.h"


#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

namespace mfem {

  // Mixed FEM Sequential Quadratic Program
  template <int DIM>
  class MixedSQPPDOptimizer : public Optimizer<DIM> {

    typedef Optimizer<DIM> Base;

  public:
    
    MixedSQPPDOptimizer(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<SimConfig> config) : Optimizer<DIM>(mesh, config) {}

    static std::string name() {
      return "SQP-PD";
    }

    void step() override;
    void reset() override;

  private:

    using Base::mesh_;
    using Base::data_;
    using Base::config_;

    // Update gradients, LHS, RHS for a new configuration
    void update_system();

    void substep(double& decrement);

    // linear system left hand side
    Eigen::SparseMatrix<double, Eigen::RowMajor> lhs_; 

    // linear system right hand side
    Eigen::VectorXd rhs_;       

    std::shared_ptr<Stretch<DIM>> svar_;
    std::shared_ptr<Displacement<DIM>> xvar_;
    std::shared_ptr<LinearSolver<double, Eigen::RowMajor>> solver_;
  };
}
