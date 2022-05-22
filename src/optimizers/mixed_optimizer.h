#pragma once

#include "optimizers/optimizer.h"


#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

namespace mfem {

  // Mixed FEM Optimizer Base Class
  class MixedOptimizer : public Optimizer {
  public:
    
    MixedOptimizer(std::shared_ptr<Mesh> object,
        std::shared_ptr<SimConfig> config) : Optimizer(object, config) {}

    void reset() override;
    void step() override;
    virtual void update_vertices(const Eigen::MatrixXd& V) override;
    virtual void set_state(const Eigen::VectorXd& x,
        const Eigen::VectorXd& v) override;
  public:

    virtual double primal_energy(const Eigen::VectorXd& x, const Eigen::VectorXd& s, 
      Eigen::VectorXd& gx, Eigen::VectorXd& gs);

    // Evaluated mixed energy
    virtual double energy(const Eigen::VectorXd& x, const Eigen::VectorXd& s,
        const Eigen::VectorXd& la) = 0;

    // Mixed energy gradient
    virtual void gradient(Eigen::VectorXd& g, const Eigen::VectorXd& x,
        const Eigen::VectorXd& s, const Eigen::VectorXd& la) = 0;

    // Build system left hand side
    virtual void build_lhs() = 0;

    // Build linear system right hand side
    virtual void build_rhs() = 0;

    // Prepare for next solver iterate
    // Update gradients, LHS, RHS for a new configuration
    virtual void update_system() = 0;

    // Simulation substep for this object
    // step      - current substep
    // decrement - newton decrement norm
    virtual void substep(int step, double& decrement) = 0;

    // For a new set of positions, update R,S and their derivatives
    virtual void update_rotations();

    // Linesearch over positions
    // x  - initial positions. Output of linesearch updates this variable
    // dx - direction we perform linesearch on
    virtual bool linesearch_x(Eigen::VectorXd& x, const Eigen::VectorXd& dx);
    virtual bool linesearch_s(Eigen::VectorXd& s, const Eigen::VectorXd& ds);
    virtual bool linesearch_s_local(Eigen::VectorXd& s,
        const Eigen::VectorXd& ds);
    virtual bool linesearch(Eigen::VectorXd& x, const Eigen::VectorXd& dx,
        Eigen::VectorXd& s, const Eigen::VectorXd& ds);

    // At the end of the timestep, update position, velocity variables,
    // and reset lambda & kappa.
    virtual void update_configuration();

    // Configuration vectors & body forces
    Eigen::VectorXd q_;
    Eigen::VectorXd x_;        // current positions
    Eigen::VectorXd vt_;        // current velocities
    Eigen::VectorXd x0_;        // previous positions
    Eigen::VectorXd x1_;        // previous previous positions positions
    Eigen::VectorXd x2_;        // previous previous positions positions
    Eigen::VectorXd dx_;        // current update
    Eigen::VectorXd f_ext_;     // per-node external forces
    Eigen::VectorXd la_;        // lambdas
    Eigen::VectorXd ds_;        // deformation updates
    Eigen::VectorXd s_;         // deformation variables
    Eigen::VectorXd b_;         // coordinates projected out
    Eigen::VectorXd vols_;      // per element volume
    Eigen::VectorXd rhs_;       // linear system right hand side
    Eigen::SparseMatrixdRowMajor lhs_; // linear system left hand side

    std::vector<Eigen::Matrix3d> R_;  // Per-element rotations
    std::vector<Eigen::Vector6d> S_;    // Per-element deformation
    std::vector<Eigen::Matrix6d> H_; // Elemental hessians w.r.t dS
    std::vector<Eigen::Matrix6d> Hinv_; // Elemental hessians w.r.t dS
    std::vector<Eigen::Vector6d> g_;    // Elemental gradients w.r.t dS
    std::vector<Eigen::Matrix<double,9,6>> dS_;

    Eigen::SparseMatrixdRowMajor M_;     // projected mass matrix
    Eigen::SparseMatrixdRowMajor Mfull_; // mass matrix
    Eigen::SparseMatrixdRowMajor J_;  // jacobian
    Eigen::SparseMatrixd W_;
    Eigen::VectorXd grad_;

    double E_prev_; // energy from last result of linesearch

    //weights for bdf1 or bdf2
   

  };
}
