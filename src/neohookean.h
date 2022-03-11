#pragma once

#include <EigenTypes.h>

// Utility for workin with corotational
//
// Compliance matrix entries for KKT system (bottom right block)
// - Appends entries onto the provided vector of triplets
void neohookean_compliance(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    std::vector<Eigen::Matrix3d>& R, std::vector<Eigen::Matrix6d>& Hinv,
    const Eigen::VectorXd& vols, double mu,
    double la, std::vector<Eigen::Triplet<double>>& trips);
void update_neohookean_compliance(int n, int m,
    std::vector<Eigen::Matrix3d>& R, std::vector<Eigen::Matrix6d>& Hinv,
    const Eigen::VectorXd& vols,
    double mu, double la, Eigen::SparseMatrixd& mat);

Eigen::Vector6d neohookean_ds(const Eigen::Matrix3d& R, 
        const Eigen::Vector6d& S, const Eigen::Vector9d& L,
        const Eigen::Matrix6d& Hinv, double mu, double la);

Eigen::Matrix9d neohookean_WHinvW(const Eigen::Matrix3d& R,
        const Eigen::Matrix6d& Hinv);
Eigen::Vector9d neohookean_rhs(const Eigen::Matrix3d& R,
       const Eigen::Vector6d& S, const Eigen::Matrix6d& Hinv,
       double mu, double la);
Eigen::Vector9d neohookean_rhs(const Eigen::Matrix3d& R,
       const Eigen::Vector6d& S, const Eigen::Matrix6d& Hinv,
       const Eigen::Vector6d& g);

double neohookean_psi(const Eigen::Vector6d& S, double mu, double la); 

Eigen::Vector6d neohookean_g(const Eigen::Matrix3d& R, const Eigen::Vector6d& S,
    double mu, double la);
Eigen::Matrix6d neohookean_hinv(const Eigen::Matrix3d& R, const Eigen::Vector6d& S,
        double mu, double la);
