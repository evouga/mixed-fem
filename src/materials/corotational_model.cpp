#include "materials/corotational_model.h"

using namespace Eigen;
using namespace mfem;

double CorotationalModel::energy(const Vector6d& S) {
    
  double mu = config_->mu;
  double la = config_->la;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  return (la*pow(S1+S2+S3-3.0,2.0))/2.0+mu*(pow(S1-1.0,2.0)+pow(S2-1.0,2.0)
      +pow(S3-1.0,2.0)+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0);
}

Vector6d CorotationalModel::gradient(const Matrix3d& R,
    const Vector6d& S) {
  
  double mu = config_->mu;
  double la = config_->la;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  Vector6d g;
  g(0) = (la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S1*2.0-2.0);
  g(1) = (la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S2*2.0-2.0);
  g(2) = (la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S3*2.0-2.0);
  g(3) = S4*mu*4.0;
  g(4) = S5*mu*4.0;
  g(5) = S6*mu*4.0;
  return g;

}

Matrix6d CorotationalModel::hessian_inv(const Matrix3d& R,
    const Vector6d& S) {
  
  double mu = config_->mu;
  double la = config_->la;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  Matrix6d Hinv;
  Hinv.setZero();
  Hinv(0,0) = (la+mu)/(la*mu*3.0+(mu*mu)*2.0);
  Hinv(0,1) = (la*(-1.0/2.0))/(la*mu*3.0+(mu*mu)*2.0);
  Hinv(0,2) = (la*(-1.0/2.0))/(la*mu*3.0+(mu*mu)*2.0);
  Hinv(1,0) = (la*(-1.0/2.0))/(la*mu*3.0+(mu*mu)*2.0);
  Hinv(1,1) = (la+mu)/(la*mu*3.0+(mu*mu)*2.0);
  Hinv(1,2) = (la*(-1.0/2.0))/(la*mu*3.0+(mu*mu)*2.0);
  Hinv(2,0) = (la*(-1.0/2.0))/(la*mu*3.0+(mu*mu)*2.0);
  Hinv(2,1) = (la*(-1.0/2.0))/(la*mu*3.0+(mu*mu)*2.0);
  Hinv(2,2) = (la+mu)/(la*mu*3.0+(mu*mu)*2.0);
  Hinv(3,3) = 1.0/(mu*4.0);
  Hinv(4,4) = 1.0/(mu*4.0);
  Hinv(5,5) = 1.0/(mu*4.0);
  return Hinv;
     
}

Matrix6d CorotationalModel::hessian_inv(const Matrix3d& R,
    const Vector6d& S, double kappa) {

  std::cerr << "hessian_inv with regularizer not implemented!" << std::endl;
  return hessian_inv(R,S);
}
