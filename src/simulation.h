#pragma once

#include <Eigen/Dense>
#include <EigenTypes.h>

namespace mfem {

  using SparseMatrixdRowMajor = Eigen::SparseMatrix<double, Eigen::RowMajor>;


  static Eigen::Vector6d I_vec = (Eigen::Vector6d() <<
      1, 1, 1, 0, 0, 0).finished();

  class Simulation {
  public:

    void init();
    void build_rhs();

  private:
    std::vector<Eigen::Matrix3d> R_;    // Per-element rotations
    std::vector<Eigen::Vector6d> S_;    // Per-element deformation
    std::vector<Eigen::Vector6d> dS_;   // Per-element deformation update
    std::vector<Eigen::Matrix6d> Hinv_; // Elemental hessians w.r.t dS
    std::vector<Eigen::Vector6d> g_;    // Elemental gradients w.r.t dS

    Eigen::SparseMatrixd M;          // mass matrix
    Eigen::SparseMatrixd P;          // pinning constraint (for vertices)
    Eigen::SparseMatrixd P_kkt;      // pinning constraint (for kkt matrix)
    SparseMatrixdRowMajor Jw; // integrated (weighted) jacobian
    SparseMatrixdRowMajor J;  // jacobian
    Eigen::MatrixXd dphidX; 
    Eigen::VectorXi pinnedV;

    // Configuration vectors & body forces
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


    // CG temp variables
    Eigen::VectorXd tmp_r_;
    Eigen::VectorXd tmp_z_;
    Eigen::VectorXd tmp_p_;
    Eigen::VectorXd tmp_Ap_;

    Eigen::MatrixXd V_;
    Eigen::MatrixXi T_;


  };
}