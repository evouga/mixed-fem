#pragma once

#include "optimizers/optimizer.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

namespace mfem {

  // Standard linear FEM 
  class NewtonOptimizer : public Optimizer {
  public:
    
    NewtonOptimizer(std::shared_ptr<SimObject> object,
        std::shared_ptr<SimConfig> config) : Optimizer(object, config) {}

    void reset() override;
    void step() override;
  
  public:

    virtual double energy(const Eigen::VectorXd& x);

    // Build system left hand side
    virtual void build_lhs();

    // Build linear system right hand side
    virtual void build_rhs();

    // Simulation substep for this object
    // init_guess - whether to initialize guess with a prefactor solve
    // decrement  - newton decrement norm
    virtual void substep(bool init_guess, double& decrement);

    // At the end of the timestep, update position, velocity variables,
    // and reset lambda & kappa.
    virtual void update_configuration();

    // Configuration vectors & body forces
    Eigen::VectorXd x_;        // current positions
    Eigen::VectorXd vt_;        // current velocities
    Eigen::VectorXd x0_;        // previous positions
    Eigen::VectorXd dx_;        // current update
    Eigen::VectorXd f_ext_;     // per-node external forces
    Eigen::VectorXd b_;         // coordinates projected out
    Eigen::VectorXd vols_;      // per element volume
    Eigen::VectorXd rhs_;       // linear system right hand side
    Eigen::SparseMatrixd lhs_;  // linear system left hand side

    Eigen::SparseMatrixd M_;          // mass matrix
    Eigen::SparseMatrixd P_;          // pinning constraint (for vertices)
    Eigen::SparseMatrixdRowMajor J_;  // jacobian
    Eigen::MatrixXd dphidX_;

    // Solve used for preconditioner
    #if defined(SIM_USE_CHOLMOD)
    Eigen::CholmodSimplicialLDLT<Eigen::SparseMatrixd> solver_;
    #else
    Eigen::SimplicialLLT<Eigen::SparseMatrixd> solver_;
    #endif

    double E_prev_; // energy from last result of linesearch

    Eigen::VectorXi pinnedV_;
  };
}
