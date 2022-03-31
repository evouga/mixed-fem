#include "materials/neohookean_model.h"

using namespace Eigen;
using namespace mfem;


void StableNeohookean::update_compliance(int n, int m, 
    const std::vector<Eigen::Matrix3d>& R,
    const std::vector<Eigen::Matrix6d>& Hinv,
    const Eigen::VectorXd& vols, Eigen::SparseMatrixd& mat) {

}

double StableNeohookean::energy(const Eigen::Vector6d& S) {

}

Eigen::Vector6d StableNeohookean::gradient(const Eigen::Matrix3d& R,
    const Eigen::Vector6d& S) {

}

Eigen::Matrix6d StableNeohookean::hessian_inv(const Eigen::Matrix3d& R,
    const Eigen::Vector6d& S) {
  
  double mu = config_->mu;
  double la = config_->la;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  Matrix6d H;
  H(0,0) = mu+la*pow(S2*S3-S6*S6,2.0);
  H(0,1) = -S3*mu-S3*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S3-S5*S5)*(S2*S3-S6*S6);
  H(0,2) = -S2*mu-S2*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S2-S4*S4)*(S2*S3-S6*S6);
  H(0,3) = -la*(S3*S4*2.0-S5*S6*2.0)*(S2*S3-S6*S6);
  H(0,4) = -la*(S2*S5*2.0-S4*S6*2.0)*(S2*S3-S6*S6);
  H(0,5) = S6*mu*2.0+S6*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S1*S6*2.0-S4*S5*2.0)*(S2*S3-S6*S6);
  H(1,0) = -S3*mu-S3*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S3-S5*S5)*(S2*S3-S6*S6);
  H(1,1) = mu+la*pow(S1*S3-S5*S5,2.0);
  H(1,2) = -S1*mu-S1*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S2-S4*S4)*(S1*S3-S5*S5);
  H(1,3) = -la*(S3*S4*2.0-S5*S6*2.0)*(S1*S3-S5*S5);
  H(1,4) = S5*mu*2.0+S5*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S2*S5*2.0-S4*S6*2.0)*(S1*S3-S5*S5);
  H(1,5) = -la*(S1*S6*2.0-S4*S5*2.0)*(S1*S3-S5*S5);
  H(2,0) = -S2*mu-S2*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S2-S4*S4)*(S2*S3-S6*S6);
  H(2,1) = -S1*mu-S1*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S2-S4*S4)*(S1*S3-S5*S5);
  H(2,2) = mu+la*pow(S1*S2-S4*S4,2.0);
  H(2,3) = S4*mu*2.0+S4*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S3*S4*2.0-S5*S6*2.0)*(S1*S2-S4*S4);
  H(2,4) = -la*(S2*S5*2.0-S4*S6*2.0)*(S1*S2-S4*S4);
  H(2,5) = -la*(S1*S6*2.0-S4*S5*2.0)*(S1*S2-S4*S4);
  H(3,0) = -la*(S3*S4*2.0-S5*S6*2.0)*(S2*S3-S6*S6);
  H(3,1) = -la*(S3*S4*2.0-S5*S6*2.0)*(S1*S3-S5*S5);
  H(3,2) = S4*mu*2.0+S4*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S3*S4*2.0-S5*S6*2.0)*(S1*S2-S4*S4);
  H(3,3) = mu*2.0+S3*mu*2.0+la*pow(S3*S4*2.0-S5*S6*2.0,2.0)+S3*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0;
  H(3,4) = S6*mu*-2.0-S6*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0+la*(S2*S5*2.0-S4*S6*2.0)*(S3*S4*2.0-S5*S6*2.0);
  H(3,5) = S5*mu*-2.0-S5*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0+la*(S1*S6*2.0-S4*S5*2.0)*(S3*S4*2.0-S5*S6*2.0);
  H(4,0) = -la*(S2*S5*2.0-S4*S6*2.0)*(S2*S3-S6*S6);
  H(4,1) = S5*mu*2.0+S5*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S2*S5*2.0-S4*S6*2.0)*(S1*S3-S5*S5);
  H(4,2) = -la*(S2*S5*2.0-S4*S6*2.0)*(S1*S2-S4*S4);
  H(4,3) = S6*mu*-2.0-S6*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0+la*(S2*S5*2.0-S4*S6*2.0)*(S3*S4*2.0-S5*S6*2.0);
  H(4,4) = mu*2.0+S2*mu*2.0+la*pow(S2*S5*2.0-S4*S6*2.0,2.0)+S2*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0;
  H(4,5) = S4*mu*-2.0-S4*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0+la*(S1*S6*2.0-S4*S5*2.0)*(S2*S5*2.0-S4*S6*2.0);
  H(5,0) = S6*mu*2.0+S6*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S1*S6*2.0-S4*S5*2.0)*(S2*S3-S6*S6);
  H(5,1) = -la*(S1*S6*2.0-S4*S5*2.0)*(S1*S3-S5*S5);
  H(5,2) = -la*(S1*S6*2.0-S4*S5*2.0)*(S1*S2-S4*S4);
  H(5,3) = S5*mu*-2.0-S5*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0+la*(S1*S6*2.0-S4*S5*2.0)*(S3*S4*2.0-S5*S6*2.0);
  H(5,4) = S4*mu*-2.0-S4*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0+la*(S1*S6*2.0-S4*S5*2.0)*(S2*S5*2.0-S4*S6*2.0);
  H(5,5) = mu*2.0+S1*mu*2.0+la*pow(S1*S6*2.0-S4*S5*2.0,2.0)+S1*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0;

  //sim::simple_psd_fix(H);

  return H.inverse();
      
}

Eigen::Matrix9d StableNeohookean::WHinvW(const Eigen::Matrix3d& R,
    const Eigen::Matrix6d& Hinv) {

}

Eigen::Vector9d StableNeohookean::rhs(const Eigen::Matrix3d& R,
    const Eigen::Vector6d& S, const Eigen::Matrix6d& Hinv,
    const Eigen::Vector6d& g) {
      
}

Eigen::Vector6d StableNeohookean::dS(const Eigen::Matrix3d& R, 
    const Eigen::Vector6d& S, const Eigen::Vector9d& L,
    const Eigen::Matrix6d& Hinv) {
      
}
    