#pragma once

// Functions for KKT lhs & rhs assembly for tetrahedral simulation
#include <EigenTypes.h>

// Assembles *most* of the LHS for the tetmesh KKT
// Leaves out the entries for the compliance matrix as these
// may be non-const. Leave it to caller to add these entries.
//
// M: mass matrix
// Jw: volume weighted jacobian
// ih2: inverse squared timestep
// trips: output triplets for matrix assembly
//
// Outputs into vector for A,B,B^T blocks for the kkt of form [A B^T; B C]
void tet_kkt_lhs(const Eigen::SparseMatrixd& M,
        const Eigen::SparseMatrix<double,Eigen::RowMajor>& Jw,
        double ih2, std::vector<Eigen::Triplet<double>>& trips);

// Builds jacobian matrix
// size:  9|F| x 3|V|
Eigen::SparseMatrix<double,Eigen::RowMajor> tet_jacobian(const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& T, const Eigen::VectorXd& vols, bool weighted);


