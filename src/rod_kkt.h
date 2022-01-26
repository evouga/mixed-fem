#pragma once

// Functions for KKT lhs & rhs assembly for triangle simulation
#include <EigenTypes.h>

// Builds jacobian matrix
// size:  9|F| x 3|V|
Eigen::SparseMatrix<double,Eigen::RowMajor> rod_jacobian(
    const Eigen::MatrixXd& V, const Eigen::MatrixXi& E,
    const Eigen::VectorXd& vols, bool weighted);

Eigen::SparseMatrixd rod_massmatrix(const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E, const Eigen::VectorXd& vols);
