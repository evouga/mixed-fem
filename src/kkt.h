#pragma once

#include <EigenTypes.h>

namespace mfem {
  // Initializes the LHS for KKT system. Compliance block initialized with
  // diagonal 1/mu entries.
  //
  // M      - mass matrix
  // Jw     - volume weighted jacobian
  // vols   - per-element volumes
  // mu     - lame parameter
  // ih2    - inverse squared timestep
  // trips  - output triplets for matrix assembly
  //
  // Outputs into vector for A,B,B^T,C blocks for the kkt of form [A B^T; B C]
  void kkt_lhs(const Eigen::SparseMatrixd& M,
      const Eigen::SparseMatrix<double,Eigen::RowMajor>& Jw,
      double ih2, std::vector<Eigen::Triplet<double>>& trips);

  void diagonal_compliance(const Eigen::VectorXd& vols, double mu, int offset,
      std::vector<Eigen::Triplet<double>>& trips);

  void init_compliance_blocks(int N, int offset,
      std::vector<Eigen::Triplet<double>>& trips);
}