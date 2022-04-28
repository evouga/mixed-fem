#pragma once

#include <Eigen/Dense>
#include <EigenTypes.h>
#include <memory>

#include "materials/material_model.h"
#include "config.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

namespace mfem {

  using SparseMatrixdRowMajor = Eigen::SparseMatrix<double, Eigen::RowMajor>;

  static Eigen::Vector6d I_vec = (Eigen::Vector6d() <<
      1, 1, 1, 0, 0, 0).finished();


  // Class to maintain the state and perform physics updates on an object,
  // which has a particular discretization, material, and material config
  class SimObject {
  public:

    SimObject(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        std::shared_ptr<SimConfig> config,
        std::shared_ptr<MaterialModel> material,
        std::shared_ptr<MaterialConfig> material_config)
        : V_(V), V0_(V), T_(T), config_(config), material_(material),
          material_config_(material_config) {
    }
    
    void reset_variables();
    virtual void volumes(Eigen::VectorXd& vol) = 0;
    virtual void mass_matrix(Eigen::SparseMatrixd& M) = 0;
    virtual void jacobian(SparseMatrixdRowMajor& J, bool weighted) = 0;
    virtual void jacobian_regularized(SparseMatrixdRowMajor& J,
        bool weighted) = 0;
    virtual void jacobian_rotational(SparseMatrixdRowMajor& J,
        bool weighted) {}    
    virtual void massmatrix_rotational(Eigen::SparseMatrixd& J) {}  
    double energy(Eigen::VectorXd x, std::vector<Eigen::Vector6d> s, Eigen::VectorXd la);
    void init();

    // Build the KKT lhs (just initializes it). Still have to update the
    // compliance blocks each timestep.
    virtual void build_lhs();

    // Build the KKT right hand side
    virtual void build_rhs();
    
    // Update per-element S, symmetric deformation, and R, rotation matrices
    virtual void fit_rotations(Eigen::VectorXd& dq, Eigen::VectorXd& la);
    virtual void update_rotations();
    virtual void update_s(std::vector<Eigen::Vector6d>& s, const Eigen::VectorXd& q);
    virtual void linesearch(Eigen::VectorXd& q, const Eigen::VectorXd& dq);
    virtual void linesearch();

    // Recompute per-element gradient and hessians using new
    // S and R matrices.
    void update_gradients();

    // Simulation substep for this object
    // init_guess - whether to initialize guess with a prefactor solve
    void substep(bool init_guess);
  
    void warm_start();
    void update_lambdas(int t);
    void update_positions();

    virtual Eigen::VectorXd collision_force();

    Eigen::MatrixXd vertices() {
      return V_;
    }

  protected:

    std::shared_ptr<SimConfig> config_;
    std::shared_ptr<MaterialModel> material_;
    std::shared_ptr<MaterialConfig> material_config_;

    std::vector<Eigen::Matrix3d> R_;    // Per-element rotations
    std::vector<Eigen::Vector6d> S_;    // Per-element deformation
    std::vector<Eigen::Matrix6d> Hinv_; // Elemental hessians w.r.t dS
    std::vector<Eigen::Vector6d> g_;    // Elemental gradients w.r.t dS
    std::vector<Eigen::Matrix9d> dRS_;  // dRS/dF where each row is dRS/dF_ij
    std::vector<Eigen::Matrix<double,9,6>> dRL_;
    std::vector<Eigen::Matrix<double,9,6>> dRe_;

    Eigen::SparseMatrixd M_;        // mass matrix
    Eigen::SparseMatrixd P_;        // pinning constraint (for vertices)
    Eigen::SparseMatrixd P_kkt_;    // pinning constraint (for kkt matrix)
    SparseMatrixdRowMajor J_;       // jacobian
    SparseMatrixdRowMajor Jw_;      // integrated (weighted) jacobian
    SparseMatrixdRowMajor J_tilde_;  // jacobian for regularizer
    SparseMatrixdRowMajor Jw_tilde_; // jacobian for regularizer
    SparseMatrixdRowMajor Jw_rot_; // jacobian for regularizer
    SparseMatrixdRowMajor Ws_;      // integrated (weighted) jacobian
    SparseMatrixdRowMajor WhatS_;
    SparseMatrixdRowMajor Whate_;
    SparseMatrixdRowMajor WhatL_;

    Eigen::VectorXi pinnedV_;

    // Configuration vectors & body forces
    Eigen::VectorXd dq_ds_; // q & Lambda update stacked
    Eigen::VectorXd qt_;    // current positions
    Eigen::VectorXd vt_;    // current velocities
    Eigen::VectorXd q0_;    // previous positions
    //Eigen::VectorXd q1_;    // previous^2 positions
    Eigen::VectorXd dq_;    // current update
    Eigen::VectorXd f_ext_; // per-node external forces
    //Eigen::VectorXd f_ext0_;// per-node external forces (not integrated)
    //Eigen::VectorXd f_ext1_;// per-node external forces (not integrated)
    Eigen::VectorXd la_;    // lambdas
    Eigen::VectorXd ds_;    // lambdas
    Eigen::VectorXd b_;     // coordinates projected out
    Eigen::VectorXd vols_;  // per element volume
    Eigen::VectorXd rhs_;
    Eigen::SparseMatrixd lhs_;
    Eigen::SparseMatrixd lhs_reg_; // regularized mass matrix
    Eigen::SparseMatrixd lhs_rot_; // regularized mass matrix
    double E_prev = 0;



    #if defined(SIM_USE_CHOLMOD)
    Eigen::CholmodSimplicialLDLT<Eigen::SparseMatrixd> solver_;
    #else
    Eigen::SimplicialLDLT<Eigen::SparseMatrixd> solver_;
    #endif

    // CG temp variables
    Eigen::VectorXd tmp_r_;
    Eigen::VectorXd tmp_z_;
    Eigen::VectorXd tmp_p_;
    Eigen::VectorXd tmp_Ap_;

    Eigen::MatrixXd V_;
    Eigen::MatrixXd V0_;
    Eigen::MatrixXi T_;

    double ibeta_;

    // Debug timing variables
    double t_coll = 0;
    double t_asm = 0;
    double t_precond = 0;
    double t_rhs = 0;
    double t_solve = 0;
    double t_SR = 0; 
  };
}

// Add discretizations
#include "objects/tet_object.h"
#include "objects/tri_object.h"
#include "objects/rod_object.h"
