#pragma once

#include "optimizers/mixed_admm_optimizer.h"
#include <deque>

namespace mfem {

  // Mixed FEM Augmented Lagrangian method with proximal point method for
  // solving the dual variables.
  class MixedLBFGSOptimizer : public MixedADMMOptimizer {
  public:
    MixedLBFGSOptimizer(std::shared_ptr<SimObject> object,
        std::shared_ptr<SimConfig> config) : MixedADMMOptimizer(object, config) {}

    void reset() override;
    void step() override;
  
  protected:

    // Build system left hand side
    // Build linear system right hand side
    virtual void hessian_s();

    // Simulation substep for this object
    // init_guess - whether to initialize guess with a prefactor solve
    // decrement  - newton decrement norm
    virtual void substep(bool init_guess, double& decrement) override;

    int history_ = 5; // size of history

    // using notation from Alg 2 in
    // https://tiantianliu.cn/papers/liu17quasi/liu17quasi.pdf
    std::deque<Eigen::VectorXd> si_, ti_;
    std::deque<double> rho_;

  };
}