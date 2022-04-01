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

  class SimObject {
  public:

    void init();

    // Build the KKT right hand side
    void build_rhs();
    
    // Update per-element S, symmetric deformation, and R, rotation matrices
    void update_SR();

    // Recompute per-element gradient and hessians using new
    // S and R matrices.
    void update_gradients();

    // Simulation substep for this object
    // init_guess - whether to initialize guess with a prefactor solve
    void substep(bool init_guess);

    Eigen::VectorXd collision_force();


  private:

    SimConfig config_;
    std::shared_ptr<MaterialModel> material_;
    std::shared_ptr<MaterialConfig> material_config_;

    std::vector<Eigen::Matrix3d> R_;    // Per-element rotations
    std::vector<Eigen::Vector6d> S_;    // Per-element deformation
    std::vector<Eigen::Vector6d> dS_;   // Per-element deformation update
    std::vector<Eigen::Matrix6d> Hinv_; // Elemental hessians w.r.t dS
    std::vector<Eigen::Vector6d> g_;    // Elemental gradients w.r.t dS

    Eigen::SparseMatrixd M_;     // mass matrix
    Eigen::SparseMatrixd P_;     // pinning constraint (for vertices)
    Eigen::SparseMatrixd P_kkt_; // pinning constraint (for kkt matrix)
    SparseMatrixdRowMajor Jw_;   // integrated (weighted) jacobian
    SparseMatrixdRowMajor J_;    // jacobian
    Eigen::MatrixXd dphidX_; 
    Eigen::VectorXi pinnedV_;

    // Configuration vectors & body forces
    Eigen::VectorXd dq_la_; // q & Lambda update stacked
    Eigen::VectorXd qt_;    // current positions
    Eigen::VectorXd q0_;    // previous positions
    Eigen::VectorXd q1_;    // previous^2 positions
    Eigen::VectorXd dq_;    // current update
    Eigen::VectorXd f_ext_; // per-node external forces
    Eigen::VectorXd f_ext0_;// per-node external forces (not integrated)
    Eigen::VectorXd la_;    // lambdas
    Eigen::VectorXd b_;     // coordinates projected out
    Eigen::VectorXd vols_;  // per element volume
    Eigen::VectorXd rhs_;

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