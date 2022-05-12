#include "materials/neohookean_model.h"
#include "simple_psd_fix.h"

using namespace Eigen;
using namespace mfem;

double StableNeohookean::energy(const Vector6d& S) {
    
  double mu = config_->mu;
  double la = config_->la;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  return mu*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)
    +(la*pow(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0,2.0))
    /2.0+(mu*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0;
}

Vector6d StableNeohookean::gradient(const Vector6d& S) {
  
  double mu = config_->mu;
  double la = config_->la;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  Vector6d g;
  g(0) = S1*mu-mu*(S2*S3-S6*S6)-la*(S2*S3-S6*S6)*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0);
  g(1) = S2*mu-mu*(S1*S3-S5*S5)-la*(S1*S3-S5*S5)*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0);
  g(2) = S3*mu-mu*(S1*S2-S4*S4)-la*(S1*S2-S4*S4)*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0);
  g(3) = S4*mu*2.0+mu*(S3*S4*2.0-S5*S6*2.0)+la*(S3*S4*2.0-S5*S6*2.0)*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0);
  g(4) = S5*mu*2.0+mu*(S2*S5*2.0-S4*S6*2.0)+la*(S2*S5*2.0-S4*S6*2.0)*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0);
  g(5) = S6*mu*2.0+mu*(S1*S6*2.0-S4*S5*2.0)+la*(S1*S6*2.0-S4*S5*2.0)*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0);
  return g;

}

Matrix6d StableNeohookean::hessian(const Eigen::Vector6d& S) {
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
  sim::simple_psd_fix(H);
  return H;
}

double StableNeohookean::energy(const Vector9d& F) {
  double mu = config_->mu;
  double la = config_->la;
  double F1_1 = F(0);
  double F2_1 = F(1);
  double F3_1 = F(2);
  double F1_2 = F(3);
  double F2_2 = F(4);
  double F3_2 = F(5);
  double F1_3 = F(6);
  double F2_3 = F(7);
  double F3_3 = F(8);
  return mu*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0)+(la*pow(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0,2.0))/2.0+(mu*(F1_1*F1_1+F1_2*F1_2+F1_3*F1_3+F2_1*F2_1+F2_2*F2_2+F2_3*F2_3+F3_1*F3_1+F3_2*F3_2+F3_3*F3_3-3.0))/2.0;

}

Vector9d StableNeohookean::gradient(const Vector9d& F) {
  double mu = config_->mu;
  double la = config_->la;
  double F1_1 = F(0);
  double F2_1 = F(1);
  double F3_1 = F(2);
  double F1_2 = F(3);
  double F2_2 = F(4);
  double F3_2 = F(5);
  double F1_3 = F(6);
  double F2_3 = F(7);
  double F3_3 = F(8);
  Vector9d g;
  g(0,0) = -mu*(F2_2*F3_3-F2_3*F3_2)+F1_1*mu-la*(F2_2*F3_3-F2_3*F3_2)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  g(1,0) = mu*(F1_2*F3_3-F1_3*F3_2)+F2_1*mu+la*(F1_2*F3_3-F1_3*F3_2)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  g(2,0) = -mu*(F1_2*F2_3-F1_3*F2_2)+F3_1*mu-la*(F1_2*F2_3-F1_3*F2_2)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  g(3,0) = mu*(F2_1*F3_3-F2_3*F3_1)+F1_2*mu+la*(F2_1*F3_3-F2_3*F3_1)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  g(4,0) = -mu*(F1_1*F3_3-F1_3*F3_1)+F2_2*mu-la*(F1_1*F3_3-F1_3*F3_1)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  g(5,0) = mu*(F1_1*F2_3-F1_3*F2_1)+F3_2*mu+la*(F1_1*F2_3-F1_3*F2_1)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  g(6,0) = -mu*(F2_1*F3_2-F2_2*F3_1)+F1_3*mu-la*(F2_1*F3_2-F2_2*F3_1)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  g(7,0) = mu*(F1_1*F3_2-F1_2*F3_1)+F2_3*mu+la*(F1_1*F3_2-F1_2*F3_1)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  g(8,0) = -mu*(F1_1*F2_2-F1_2*F2_1)+F3_3*mu-la*(F1_1*F2_2-F1_2*F2_1)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  return g;
}

Matrix9d StableNeohookean::hessian(const Vector9d& F) {
  double mu = config_->mu;
  double la = config_->la;
  double F1_1 = F(0);
  double F2_1 = F(1);
  double F3_1 = F(2);
  double F1_2 = F(3);
  double F2_2 = F(4);
  double F3_2 = F(5);
  double F1_3 = F(6);
  double F2_3 = F(7);
  double F3_3 = F(8);
  Matrix9d H;
  H(0,0) = mu+la*pow(F2_2*F3_3-F2_3*F3_2,2.0);
  H(0,1) = -la*(F1_2*F3_3-F1_3*F3_2)*(F2_2*F3_3-F2_3*F3_2);
  H(0,2) = la*(F1_2*F2_3-F1_3*F2_2)*(F2_2*F3_3-F2_3*F3_2);
  H(0,3) = -la*(F2_1*F3_3-F2_3*F3_1)*(F2_2*F3_3-F2_3*F3_2);
  H(0,4) = -F3_3*mu+la*(F1_1*F3_3-F1_3*F3_1)*(F2_2*F3_3-F2_3*F3_2)-F3_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(0,5) = F2_3*mu-la*(F1_1*F2_3-F1_3*F2_1)*(F2_2*F3_3-F2_3*F3_2)+F2_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(0,6) = la*(F2_1*F3_2-F2_2*F3_1)*(F2_2*F3_3-F2_3*F3_2);
  H(0,7) = F3_2*mu-la*(F1_1*F3_2-F1_2*F3_1)*(F2_2*F3_3-F2_3*F3_2)+F3_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(0,8) = -F2_2*mu+la*(F1_1*F2_2-F1_2*F2_1)*(F2_2*F3_3-F2_3*F3_2)-F2_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(1,0) = -la*(F1_2*F3_3-F1_3*F3_2)*(F2_2*F3_3-F2_3*F3_2);
  H(1,1) = mu+la*pow(F1_2*F3_3-F1_3*F3_2,2.0);
  H(1,2) = -la*(F1_2*F2_3-F1_3*F2_2)*(F1_2*F3_3-F1_3*F3_2);
  H(1,3) = F3_3*mu+la*(F1_2*F3_3-F1_3*F3_2)*(F2_1*F3_3-F2_3*F3_1)+F3_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(1,4) = -la*(F1_1*F3_3-F1_3*F3_1)*(F1_2*F3_3-F1_3*F3_2);
  H(1,5) = -F1_3*mu+la*(F1_1*F2_3-F1_3*F2_1)*(F1_2*F3_3-F1_3*F3_2)-F1_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(1,6) = -F3_2*mu-la*(F1_2*F3_3-F1_3*F3_2)*(F2_1*F3_2-F2_2*F3_1)-F3_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(1,7) = la*(F1_1*F3_2-F1_2*F3_1)*(F1_2*F3_3-F1_3*F3_2);
  H(1,8) = F1_2*mu-la*(F1_1*F2_2-F1_2*F2_1)*(F1_2*F3_3-F1_3*F3_2)+F1_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(2,0) = la*(F1_2*F2_3-F1_3*F2_2)*(F2_2*F3_3-F2_3*F3_2);
  H(2,1) = -la*(F1_2*F2_3-F1_3*F2_2)*(F1_2*F3_3-F1_3*F3_2);
  H(2,2) = mu+la*pow(F1_2*F2_3-F1_3*F2_2,2.0);
  H(2,3) = -F2_3*mu-la*(F1_2*F2_3-F1_3*F2_2)*(F2_1*F3_3-F2_3*F3_1)-F2_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(2,4) = F1_3*mu+la*(F1_2*F2_3-F1_3*F2_2)*(F1_1*F3_3-F1_3*F3_1)+F1_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(2,5) = -la*(F1_1*F2_3-F1_3*F2_1)*(F1_2*F2_3-F1_3*F2_2);
  H(2,6) = F2_2*mu+la*(F1_2*F2_3-F1_3*F2_2)*(F2_1*F3_2-F2_2*F3_1)+F2_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(2,7) = -F1_2*mu-la*(F1_2*F2_3-F1_3*F2_2)*(F1_1*F3_2-F1_2*F3_1)-F1_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(2,8) = la*(F1_1*F2_2-F1_2*F2_1)*(F1_2*F2_3-F1_3*F2_2);
  H(3,0) = -la*(F2_1*F3_3-F2_3*F3_1)*(F2_2*F3_3-F2_3*F3_2);
  H(3,1) = F3_3*mu+la*(F1_2*F3_3-F1_3*F3_2)*(F2_1*F3_3-F2_3*F3_1)+F3_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(3,2) = -F2_3*mu-la*(F1_2*F2_3-F1_3*F2_2)*(F2_1*F3_3-F2_3*F3_1)-F2_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(3,3) = mu+la*pow(F2_1*F3_3-F2_3*F3_1,2.0);
  H(3,4) = -la*(F1_1*F3_3-F1_3*F3_1)*(F2_1*F3_3-F2_3*F3_1);
  H(3,5) = la*(F1_1*F2_3-F1_3*F2_1)*(F2_1*F3_3-F2_3*F3_1);
  H(3,6) = -la*(F2_1*F3_2-F2_2*F3_1)*(F2_1*F3_3-F2_3*F3_1);
  H(3,7) = -F3_1*mu+la*(F1_1*F3_2-F1_2*F3_1)*(F2_1*F3_3-F2_3*F3_1)-F3_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(3,8) = F2_1*mu-la*(F1_1*F2_2-F1_2*F2_1)*(F2_1*F3_3-F2_3*F3_1)+F2_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(4,0) = -F3_3*mu+la*(F1_1*F3_3-F1_3*F3_1)*(F2_2*F3_3-F2_3*F3_2)-F3_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(4,1) = -la*(F1_1*F3_3-F1_3*F3_1)*(F1_2*F3_3-F1_3*F3_2);
  H(4,2) = F1_3*mu+la*(F1_2*F2_3-F1_3*F2_2)*(F1_1*F3_3-F1_3*F3_1)+F1_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(4,3) = -la*(F1_1*F3_3-F1_3*F3_1)*(F2_1*F3_3-F2_3*F3_1);
  H(4,4) = mu+la*pow(F1_1*F3_3-F1_3*F3_1,2.0);
  H(4,5) = -la*(F1_1*F2_3-F1_3*F2_1)*(F1_1*F3_3-F1_3*F3_1);
  H(4,6) = F3_1*mu+la*(F1_1*F3_3-F1_3*F3_1)*(F2_1*F3_2-F2_2*F3_1)+F3_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(4,7) = -la*(F1_1*F3_2-F1_2*F3_1)*(F1_1*F3_3-F1_3*F3_1);
  H(4,8) = -F1_1*mu+la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F3_3-F1_3*F3_1)-F1_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(5,0) = F2_3*mu-la*(F1_1*F2_3-F1_3*F2_1)*(F2_2*F3_3-F2_3*F3_2)+F2_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(5,1) = -F1_3*mu+la*(F1_1*F2_3-F1_3*F2_1)*(F1_2*F3_3-F1_3*F3_2)-F1_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(5,2) = -la*(F1_1*F2_3-F1_3*F2_1)*(F1_2*F2_3-F1_3*F2_2);
  H(5,3) = la*(F1_1*F2_3-F1_3*F2_1)*(F2_1*F3_3-F2_3*F3_1);
  H(5,4) = -la*(F1_1*F2_3-F1_3*F2_1)*(F1_1*F3_3-F1_3*F3_1);
  H(5,5) = mu+la*pow(F1_1*F2_3-F1_3*F2_1,2.0);
  H(5,6) = -F2_1*mu-la*(F1_1*F2_3-F1_3*F2_1)*(F2_1*F3_2-F2_2*F3_1)-F2_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(5,7) = F1_1*mu+la*(F1_1*F2_3-F1_3*F2_1)*(F1_1*F3_2-F1_2*F3_1)+F1_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(5,8) = -la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F2_3-F1_3*F2_1);
  H(6,0) = la*(F2_1*F3_2-F2_2*F3_1)*(F2_2*F3_3-F2_3*F3_2);
  H(6,1) = -F3_2*mu-la*(F1_2*F3_3-F1_3*F3_2)*(F2_1*F3_2-F2_2*F3_1)-F3_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(6,2) = F2_2*mu+la*(F1_2*F2_3-F1_3*F2_2)*(F2_1*F3_2-F2_2*F3_1)+F2_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(6,3) = -la*(F2_1*F3_2-F2_2*F3_1)*(F2_1*F3_3-F2_3*F3_1);
  H(6,4) = F3_1*mu+la*(F1_1*F3_3-F1_3*F3_1)*(F2_1*F3_2-F2_2*F3_1)+F3_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(6,5) = -F2_1*mu-la*(F1_1*F2_3-F1_3*F2_1)*(F2_1*F3_2-F2_2*F3_1)-F2_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(6,6) = mu+la*pow(F2_1*F3_2-F2_2*F3_1,2.0);
  H(6,7) = -la*(F1_1*F3_2-F1_2*F3_1)*(F2_1*F3_2-F2_2*F3_1);
  H(6,8) = la*(F1_1*F2_2-F1_2*F2_1)*(F2_1*F3_2-F2_2*F3_1);
  H(7,0) = F3_2*mu-la*(F1_1*F3_2-F1_2*F3_1)*(F2_2*F3_3-F2_3*F3_2)+F3_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(7,1) = la*(F1_1*F3_2-F1_2*F3_1)*(F1_2*F3_3-F1_3*F3_2);
  H(7,2) = -F1_2*mu-la*(F1_2*F2_3-F1_3*F2_2)*(F1_1*F3_2-F1_2*F3_1)-F1_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(7,3) = -F3_1*mu+la*(F1_1*F3_2-F1_2*F3_1)*(F2_1*F3_3-F2_3*F3_1)-F3_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(7,4) = -la*(F1_1*F3_2-F1_2*F3_1)*(F1_1*F3_3-F1_3*F3_1);
  H(7,5) = F1_1*mu+la*(F1_1*F2_3-F1_3*F2_1)*(F1_1*F3_2-F1_2*F3_1)+F1_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(7,6) = -la*(F1_1*F3_2-F1_2*F3_1)*(F2_1*F3_2-F2_2*F3_1);
  H(7,7) = mu+la*pow(F1_1*F3_2-F1_2*F3_1,2.0);
  H(7,8) = -la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F3_2-F1_2*F3_1);
  H(8,0) = -F2_2*mu+la*(F1_1*F2_2-F1_2*F2_1)*(F2_2*F3_3-F2_3*F3_2)-F2_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(8,1) = F1_2*mu-la*(F1_1*F2_2-F1_2*F2_1)*(F1_2*F3_3-F1_3*F3_2)+F1_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(8,2) = la*(F1_1*F2_2-F1_2*F2_1)*(F1_2*F2_3-F1_3*F2_2);
  H(8,3) = F2_1*mu-la*(F1_1*F2_2-F1_2*F2_1)*(F2_1*F3_3-F2_3*F3_1)+F2_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(8,4) = -F1_1*mu+la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F3_3-F1_3*F3_1)-F1_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(8,5) = -la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F2_3-F1_3*F2_1);
  H(8,6) = la*(F1_1*F2_2-F1_2*F2_1)*(F2_1*F3_2-F2_2*F3_1);
  H(8,7) = -la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F3_2-F1_2*F3_1);
  H(8,8) = mu+la*pow(F1_1*F2_2-F1_2*F2_1,2.0);
  sim::simple_psd_fix(H);
  return H;
}
