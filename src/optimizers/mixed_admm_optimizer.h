#pragma once

#include "optimizers/mixed_optimizer.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

namespace mfem {

  // Mixed FEM Augmented Lagrangian method with proximal point method for
  // solving the dual variables.
  class MixedADMMOptimizer : public MixedOptimizer {
  public:
    MixedADMMOptimizer(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<SimConfig> config) : MixedOptimizer(mesh, config) {}

    static std::string name() {
      return "ADMM";
    }

    void reset() override;
    void step() override;
  
  protected:

    // Evaluated augmented lagrangian energy
    virtual double energy(const Eigen::VectorXd& x, const Eigen::VectorXd& s,
        const Eigen::VectorXd& la) override;

    virtual void gradient(Eigen::VectorXd& g, const Eigen::VectorXd& x,
        const Eigen::VectorXd& s, const Eigen::VectorXd& la) override;

    virtual void build_lhs() override;
    virtual void build_rhs() override {
      std::cout << "build_rhs unused" << std::endl;
    }

    virtual void gradient_x();
    virtual void gradient_s();

    // For a new set of positions, update the rotations and their
    // derivatives
    virtual void update_rotations() override;

    // Update gradients, LHS, RHS for a new configuration
    virtual void update_system() override;

    // Simulation substep
    // init_guess - whether to initialize guess with a prefactor solve
    // decrement  - newton decrement norm
    virtual void substep(int step, double& decrement) override;

    // Warm start the timestep with a explicit euler prediction of 
    // the positions
    virtual void warm_start();

    // Update lagrange multipliers and kappa value
    virtual void update_constraints(double residual);

    // Configuration vectors & body forces
    Eigen::VectorXd gx_;
    Eigen::VectorXd gs_;

    std::vector<Eigen::Matrix6d> Hs_;   // Elemental hessians w.r.t dS
    std::vector<Eigen::Matrix9d> dRS_;           // dRS/dF
    std::vector<Eigen::Matrix<double,9,6>> dRL_; // dRL/dF
    std::vector<Eigen::Matrix<double,9,6>> dRe_; // dR(RS-F)/dF

    Eigen::SparseMatrixd J2_;
    Eigen::SparseMatrixd J_tilde_;
    Eigen::SparseMatrixd A_;        
    Eigen::SparseMatrixd G_;
    Eigen::SparseMatrixd Hx_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> L_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> Ws_; // integrated (weighted) jacobian
    Eigen::SparseMatrix<double, Eigen::RowMajor> Jw_; // integrated (weighted) jacobian

    // Rotation matrices assembled into sparse matrices
    Eigen::SparseMatrixd WhatL_;
    Eigen::SparseMatrixd WhatS_;
    Eigen::SparseMatrixd Whate_;

    // CG temp variables
    Eigen::VectorXd tmp_r_;
    Eigen::VectorXd tmp_z_;
    Eigen::VectorXd tmp_p_;
    Eigen::VectorXd tmp_Ap_;

    // Solve used for preconditioner
    #if defined(SIM_USE_CHOLMOD)
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrixd> solver_;
    #else
    Eigen::SimplicialLLT<Eigen::SparseMatrixd> solver_;
    #endif

    int nelem_;     // number of elements
    double E_prev_; // energy from last result of linesearch

    Eigen::VectorXi pinnedV_;
  };
}
