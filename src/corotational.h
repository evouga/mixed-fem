#pragma once

#include <EigenTypes.h>

Eigen::Vector9d corotational_ds(const Eigen::Matrix3d& R, 
        Eigen::Vector6d& S, Eigen::Vector9d& L, double mu, double la);
Eigen::Matrix9d corotational_WHinvW(const Eigen::Matrix3d& R,
        double mu, double la);
Eigen::Vector9d corotational_rhs(const Eigen::Matrix3d& R,
        Eigen::Vector6d& S, double mu, double la);

// Utility for workin with corotational
//
// Compliance matrix entries for KKT system (bottom right block)
// - Appends entries onto the provided vector of triplets
void corotational_compliance(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    std::vector<Eigen::Matrix3d>& R, const Eigen::VectorXd& vols, double mu,
    double la, std::vector<Eigen::Triplet<double>>& trips);
