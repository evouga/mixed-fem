#pragma once

#include <EigenTypes.h>

// Utility for workin with arap energy. We love this guy

// Compliance matrix entries for KKT system (bottom right block)
// - Appends entries onto the provided vector of triplets
void arap_compliance(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    const Eigen::VectorXd& vols, double alpha,
    std::vector<Eigen::Triplet<double>>& trips);
void arap_compliance2(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    const std::vector<Eigen::Matrix3d>& R, const Eigen::VectorXd& vols,
    double mu, double la, std::vector<Eigen::Triplet<double>>& trips);

Eigen::Vector9d arap_rhs(const Eigen::Matrix3d& R);
Eigen::Vector6d arap_ds(const Eigen::Matrix3d& R, const Eigen::Vector6d& S,
        const Eigen::Vector9d& L, double mu, double la);
Eigen::Matrix9d arap_WHinvW(const Eigen::Matrix3d& R, double mu, double la);

//Eigen::Vector9d arap_ds(const Eigen::Matrix3d& R, const Eigen::Vector6d& S,1 
//        Eigen::Vector6d& S, Eigen::Vector9d& L, double mu, double la) {
//
//  Matrix<double,9,6> W;
//  W <<
//    R[i](0,0), 0,         0,         0,         R[i](0,2), R[i](0,1),
//    0,         R[i](0,1), 0,         R[i](0,2), 0,         R[i](0,0),
//    0,         0,         R[i](0,2), R[i](0,1), R[i](0,0), 0, 
//    R[i](1,0), 0,         0,         0,         R[i](1,2), R[i](1,1),
//    0,         R[i](1,1), 0,         R[i](1,2), 0,         R[i](1,0),
//    0,         0,         R[i](1,2), R[i](1,1), R[i](1,0), 0,  
//    R[i](2,0), 0,         0,         0,         R[i](2,2), R[i](2,1),
//    0,         R[i](2,1), 0,         R[i](2,2), 0        , R[i](2,0),
//    0,         0,         R[i](2,2), R[i](2,1), R[i](2,0), 0;
//
//  // H^-1 * g = s^i - I
//  return He_inv*W.transpose()*la.segment(9*i,9) -(S[i]-I_vec); 
//
//}
