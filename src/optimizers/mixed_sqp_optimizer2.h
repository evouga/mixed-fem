#pragma once

#include "optimizers/optimizer.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

namespace mfem {

  // Mixed FEM Sequential Quadratic Program
  class MixedSQPOptimizer2 : public Optimizer {
  public:
    
    MixedSQPOptimizer2(std::shared_ptr<SimObject> object,
        std::shared_ptr<SimConfig> config) : Optimizer(object, config) {}

    void reset() override;
    void step() override;
  
  public:

    // Evaluated augmented lagrangian energy
    virtual double energy(const Eigen::VectorXd& x, const Eigen::VectorXd& s,
        const Eigen::VectorXd& la);

    // Build system left hand side
    virtual void build_lhs();

    // Build linear system right hand side
    virtual void build_rhs();

    // For a new set of positions, update the rotations and their
    // derivatives
    virtual void update_rotations();

    // Update gradients, LHS, RHS for a new configuration
    virtual void update_system();

    // Linesearch over positions
    // x  - initial positions. Output of linesearch updates this variable
    // dx - direction we perform linesearch on
    virtual bool linesearch(Eigen::VectorXd& x, const Eigen::VectorXd& dx);

    // Simulation substep for this object
    // init_guess - whether to initialize guess with a prefactor solve
    // decrement  - newton decrement norm
    virtual void substep(bool init_guess, double& decrement);

    // Warm start the timestep with a explicit euler prediction of 
    // the positions
    virtual void warm_start();

    // At the end of the timestep, update position, velocity variables,
    // and reset lambda & kappa.
    virtual void update_configuration();

    // TODO this is terrible
    virtual Eigen::VectorXd collision_force();

    // Configuration vectors & body forces
    Eigen::VectorXd xt_;        // current positions
    Eigen::VectorXd vt_;        // current velocities
    Eigen::VectorXd x0_;        // previous positions
    Eigen::VectorXd dx_;        // current update
    Eigen::VectorXd f_ext_;     // per-node external forces
    Eigen::VectorXd la_;        // lambdas
    Eigen::VectorXd ds_;        // deformation updates
    Eigen::VectorXd s_;         // deformation variables
    Eigen::VectorXd b_;         // coordinates projected out
    Eigen::VectorXd vols_;      // per element volume
    Eigen::VectorXd rhs_;       // linear system right hand side
    Eigen::SparseMatrixd lhs_;  // linear system left hand side

    std::vector<Eigen::Matrix3d> R_;  // Per-element rotations
    std::vector<Eigen::Vector6d> S_;    // Per-element deformation
    std::vector<Eigen::Matrix6d> Hinv_; // Elemental hessians w.r.t dS
    std::vector<Eigen::Vector6d> g_;    // Elemental gradients w.r.t dS
    std::vector<Eigen::Matrix<double,9,6>> dS_;

    Eigen::SparseMatrixd M_;        // mass matrix
    Eigen::SparseMatrixd P_;        // pinning constraint (for vertices)
    SparseMatrixdRowMajor J_;       // jacobian
    SparseMatrixdRowMajor Jw_;      // integrated (weighted) jacobian
    Eigen::SparseMatrixd J2_;
    Eigen::SparseMatrixd J_tilde_;
    SparseMatrixdRowMajor Ws_;      // integrated (weighted) jacobian
    Eigen::SparseMatrixd W_;
    Eigen::SparseMatrixd A_;        
    Eigen::SparseMatrixd G_;
    Eigen::SparseMatrixd D_;
    Eigen::SparseMatrixd L_;
    Eigen::SparseMatrixd C_;
    Eigen::SparseMatrixd JC_;
    Eigen::SparseMatrixd Hx_;
    Eigen::SparseMatrixd MinvC_;
    Eigen::SparseMatrixd Minv_;
    Eigen::VectorXd gx_;
    Eigen::VectorXd gs_;

    // CG temp variables
    Eigen::VectorXd tmp_r_;
    Eigen::VectorXd tmp_z_;
    Eigen::VectorXd tmp_p_;
    Eigen::VectorXd tmp_Ap_;

    // // Solve used for preconditioner
    // #if defined(SIM_USE_CHOLMOD)
    // Eigen::CholmodSimplicialLDLT<Eigen::SparseMatrixd> solver_;
    // #else
    // Eigen::SimplicialLLT<Eigen::SparseMatrixd> solver_;
    // #endif
    Eigen::SparseLU<Eigen::SparseMatrixd> solver_;

    int nelem_;     // number of elements
    double E_prev_; // energy from last result of linesearch

    Eigen::VectorXi pinnedV_;
  };
}