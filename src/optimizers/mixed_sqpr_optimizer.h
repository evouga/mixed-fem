#pragma once

#include "optimizers/mixed_sqp_optimizer.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

namespace mfem {

  // Mixed FEM Sequential Quadratic Program
  class MixedSQPROptimizer : public MixedSQPOptimizer {
  public:
    
    MixedSQPROptimizer(std::shared_ptr<SimObject> object,
        std::shared_ptr<SimConfig> config) : MixedSQPOptimizer(object, config) {}
  
  public:

    // Build system left hand side
    virtual void build_lhs() override;

    // Build linear system right hand side
    virtual void build_rhs() override;

    // Simulation substep for this object
    // init_guess - whether to initialize guess with a prefactor solve
    // decrement  - newton decrement norm
    virtual void substep(bool init_guess, double& decrement) override;

    Eigen::VectorXd gl_;
  };
}
