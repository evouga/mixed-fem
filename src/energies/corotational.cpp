#include "energies/corotational.h"
#include "simple_psd_fix.h"
#include "config.h"

using namespace Eigen;
using namespace mfem;

double Corotational::energy(const Vector6d& S) {
    
  double mu = config_->mu;
  double la = config_->la;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  return 0.5*la*(S1+S2+S3-3.0)*(S1+S2+S3-3.0)+mu*((S1-1.0)*(S1-1.0)+(S2-1.0)*(S2-1.0)+(S3-1.0)*(S3-1.0)+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0);
}

Vector6d Corotational::gradient(const Vector6d& S) {
  
  double mu = config_->mu;
  double la = config_->la;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  Vector6d g;
  g(0) = (la*(S1+S2+S3-3.0))+mu*(S1*2.0-2.0);
  g(1) = (la*(S1+S2+S3-3.0))+mu*(S2*2.0-2.0);
  g(2) = (la*(S1+S2+S3-3.0))+mu*(S3*2.0-2.0);
  g(3) = S4*mu*4.0;
  g(4) = S5*mu*4.0;
  g(5) = S6*mu*4.0;
  return g;

}

Matrix6d Corotational::hessian_inv(const Vector6d& S) {
  
  exit(1);
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

Matrix6d Corotational::hessian(const Vector6d& S, bool psd_fix) {
  Matrix6d H;
  H.setZero();
  double mu = config_->mu;
  double la = config_->la;
  H(0,0) = la+mu*2.0;
  H(0,1) = la;
  H(0,2) = la;
  H(1,0) = la;
  H(1,1) = la+mu*2.0;
  H(1,2) = la;
  H(2,0) = la;
  H(2,1) = la;
  H(2,2) = la+mu*2.0;
  H(3,3) = mu*4.0;
  H(4,4) = mu*4.0;
  H(5,5) = mu*4.0;
  return H;
}
