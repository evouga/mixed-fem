#include "corotational.h"

using namespace Eigen;

Vector9d corotational_ds(const Matrix3d& R, Vector6d& S, Vector9d& L,
    double mu, double la) {

  Vector9d ds;
  double R1_1 = R(0,0);
  double R1_2 = R(0,1);
  double R1_3 = R(0,2);
  double R2_1 = R(1,0);
  double R2_2 = R(1,1);
  double R2_3 = R(1,2);
  double R3_1 = R(2,0);
  double R3_2 = R(2,1);
  double R3_3 = R(2,2);
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  double L1 = L(0);
  double L2 = L(1);
  double L3 = L(2);
  double L4 = L(3);
  double L5 = L(4);
  double L6 = L(5);
  double L7 = L(6);
  double L8 = L(7);
  double L9 = L(8);
  ds(0) = (la*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-(mu*(S2*2.0-2.0))/2.0+L2*R1_2+L5*R2_2+L8*R3_2))/(la*mu*3.0+mu*mu)+(la*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-(mu*(S3*2.0-2.0))/2.0+L3*R1_3+L6*R2_3+L9*R3_3))/(la*mu*3.0+mu*mu)-((la*2.0+mu)*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-(mu*(S1*2.0-2.0))/2.0+L1*R1_1+L4*R2_1+L7*R3_1))/(la*mu*3.0+mu*mu);
  ds(1) = (la*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-(mu*(S1*2.0-2.0))/2.0+L1*R1_1+L4*R2_1+L7*R3_1))/(la*mu*3.0+mu*mu)+(la*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-(mu*(S3*2.0-2.0))/2.0+L3*R1_3+L6*R2_3+L9*R3_3))/(la*mu*3.0+mu*mu)-((la*2.0+mu)*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-(mu*(S2*2.0-2.0))/2.0+L2*R1_2+L5*R2_2+L8*R3_2))/(la*mu*3.0+mu*mu);
  ds(2) = (la*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-(mu*(S1*2.0-2.0))/2.0+L1*R1_1+L4*R2_1+L7*R3_1))/(la*mu*3.0+mu*mu)+(la*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-(mu*(S2*2.0-2.0))/2.0+L2*R1_2+L5*R2_2+L8*R3_2))/(la*mu*3.0+mu*mu)-((la*2.0+mu)*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-(mu*(S3*2.0-2.0))/2.0+L3*R1_3+L6*R2_3+L9*R3_3))/(la*mu*3.0+mu*mu);
  ds(3) = ((S4*mu*-2.0+L2*R1_3+L3*R1_2+L5*R2_3+L6*R2_2+L8*R3_3+L9*R3_2)*(-1.0/2.0))/mu;
  ds(4) = ((S5*mu*-2.0+L1*R1_3+L3*R1_1+L4*R2_3+L6*R2_1+L7*R3_3+L9*R3_1)*(-1.0/2.0))/mu;
  ds(5) = ((S6*mu*-2.0+L1*R1_2+L2*R1_1+L4*R2_2+L5*R2_1+L7*R3_2+L8*R3_1)*(-1.0/2.0))/mu;
  return ds;
}

Vector9d corotational_rhs(const Matrix3d& R, Vector6d& S,
    double mu, double la) {

  double R1_1 = R(0,0);
  double R1_2 = R(0,1);
  double R1_3 = R(0,2);
  double R2_1 = R(1,0);
  double R2_2 = R(1,1);
  double R2_3 = R(1,2);
  double R3_1 = R(2,0);
  double R3_2 = R(2,1);
  double R3_3 = R(2,2);
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);

  Vector9d g;
  g(0) = R1_1*(S1+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S2*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S3*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)-(((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S1*2.0-2.0))/2.0)*(la*2.0+mu))/(la*mu*3.0+mu*mu));
  g(1) = R1_2*(S2+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S1*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S3*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)-(((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S2*2.0-2.0))/2.0)*(la*2.0+mu))/(la*mu*3.0+mu*mu));
  g(2) = R1_3*(S3+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S1*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S2*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)-(((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S3*2.0-2.0))/2.0)*(la*2.0+mu))/(la*mu*3.0+mu*mu));
  g(3) = R2_1*(S1+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S2*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S3*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)-(((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S1*2.0-2.0))/2.0)*(la*2.0+mu))/(la*mu*3.0+mu*mu));
  g(4) = R2_2*(S2+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S1*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S3*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)-(((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S2*2.0-2.0))/2.0)*(la*2.0+mu))/(la*mu*3.0+mu*mu));
  g(5) = R2_3*(S3+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S1*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S2*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)-(((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S3*2.0-2.0))/2.0)*(la*2.0+mu))/(la*mu*3.0+mu*mu));
  g(6) = R3_1*(S1+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S2*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S3*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)-(((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S1*2.0-2.0))/2.0)*(la*2.0+mu))/(la*mu*3.0+mu*mu));
  g(7) = R3_2*(S2+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S1*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S3*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)-(((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S2*2.0-2.0))/2.0)*(la*2.0+mu))/(la*mu*3.0+mu*mu));
  g(8) = R3_3*(S3+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S1*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S2*2.0-2.0))/2.0))/(la*mu*3.0+mu*mu)-(((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+(mu*(S3*2.0-2.0))/2.0)*(la*2.0+mu))/(la*mu*3.0+mu*mu));
  return g;
}

Matrix9d corotational_WHinvW(const Matrix3d& R, double mu, double la) {
  double R1_1 = R(0,0);
  double R1_2 = R(0,1);
  double R1_3 = R(0,2);
  double R2_1 = R(1,0);
  double R2_2 = R(1,1);
  double R2_3 = R(1,2);
  double R3_1 = R(2,0);
  double R3_2 = R(2,1);
  double R3_3 = R(2,2);
  Matrix9d ret;
  ret(0,0) = (R1_2*R1_2)/(mu*2.0)+(R1_3*R1_3)/(mu*2.0)+((R1_1*R1_1)*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(0,1) = (R1_1*R1_2)/(mu*2.0)-(R1_1*R1_2*la)/(la*mu*3.0+mu*mu);
  ret(0,2) = (R1_1*R1_3)/(mu*2.0)-(R1_1*R1_3*la)/(la*mu*3.0+mu*mu);
  ret(0,3) = (R1_2*R2_2)/(mu*2.0)+(R1_3*R2_3)/(mu*2.0)+(R1_1*R2_1*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(0,4) = (R1_2*R2_1)/(mu*2.0)-(R1_1*R2_2*la)/(la*mu*3.0+mu*mu);
  ret(0,5) = (R1_3*R2_1)/(mu*2.0)-(R1_1*R2_3*la)/(la*mu*3.0+mu*mu);
  ret(0,6) = (R1_2*R3_2)/(mu*2.0)+(R1_3*R3_3)/(mu*2.0)+(R1_1*R3_1*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(0,7) = (R1_2*R3_1)/(mu*2.0)-(R1_1*R3_2*la)/(la*mu*3.0+mu*mu);
  ret(0,8) = (R1_3*R3_1)/(mu*2.0)-(R1_1*R3_3*la)/(la*mu*3.0+mu*mu);
  ret(1,0) = (R1_1*R1_2)/(mu*2.0)-(R1_1*R1_2*la)/(la*mu*3.0+mu*mu);
  ret(1,1) = (R1_1*R1_1)/(mu*2.0)+(R1_3*R1_3)/(mu*2.0)+((R1_2*R1_2)*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(1,2) = (R1_2*R1_3)/(mu*2.0)-(R1_2*R1_3*la)/(la*mu*3.0+mu*mu);
  ret(1,3) = (R1_1*R2_2)/(mu*2.0)-(R1_2*R2_1*la)/(la*mu*3.0+mu*mu);
  ret(1,4) = (R1_1*R2_1)/(mu*2.0)+(R1_3*R2_3)/(mu*2.0)+(R1_2*R2_2*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(1,5) = (R1_3*R2_2)/(mu*2.0)-(R1_2*R2_3*la)/(la*mu*3.0+mu*mu);
  ret(1,6) = (R1_1*R3_2)/(mu*2.0)-(R1_2*R3_1*la)/(la*mu*3.0+mu*mu);
  ret(1,7) = (R1_1*R3_1)/(mu*2.0)+(R1_3*R3_3)/(mu*2.0)+(R1_2*R3_2*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(1,8) = (R1_3*R3_2)/(mu*2.0)-(R1_2*R3_3*la)/(la*mu*3.0+mu*mu);
  ret(2,0) = (R1_1*R1_3)/(mu*2.0)-(R1_1*R1_3*la)/(la*mu*3.0+mu*mu);
  ret(2,1) = (R1_2*R1_3)/(mu*2.0)-(R1_2*R1_3*la)/(la*mu*3.0+mu*mu);
  ret(2,2) = (R1_1*R1_1)/(mu*2.0)+(R1_2*R1_2)/(mu*2.0)+((R1_3*R1_3)*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(2,3) = (R1_1*R2_3)/(mu*2.0)-(R1_3*R2_1*la)/(la*mu*3.0+mu*mu);
  ret(2,4) = (R1_2*R2_3)/(mu*2.0)-(R1_3*R2_2*la)/(la*mu*3.0+mu*mu);
  ret(2,5) = (R1_1*R2_1)/(mu*2.0)+(R1_2*R2_2)/(mu*2.0)+(R1_3*R2_3*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(2,6) = (R1_1*R3_3)/(mu*2.0)-(R1_3*R3_1*la)/(la*mu*3.0+mu*mu);
  ret(2,7) = (R1_2*R3_3)/(mu*2.0)-(R1_3*R3_2*la)/(la*mu*3.0+mu*mu);
  ret(2,8) = (R1_1*R3_1)/(mu*2.0)+(R1_2*R3_2)/(mu*2.0)+(R1_3*R3_3*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(3,0) = (R1_2*R2_2)/(mu*2.0)+(R1_3*R2_3)/(mu*2.0)+(R1_1*R2_1*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(3,1) = (R1_1*R2_2)/(mu*2.0)-(R1_2*R2_1*la)/(la*mu*3.0+mu*mu);
  ret(3,2) = (R1_1*R2_3)/(mu*2.0)-(R1_3*R2_1*la)/(la*mu*3.0+mu*mu);
  ret(3,3) = (R2_2*R2_2)/(mu*2.0)+(R2_3*R2_3)/(mu*2.0)+((R2_1*R2_1)*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(3,4) = (R2_1*R2_2)/(mu*2.0)-(R2_1*R2_2*la)/(la*mu*3.0+mu*mu);
  ret(3,5) = (R2_1*R2_3)/(mu*2.0)-(R2_1*R2_3*la)/(la*mu*3.0+mu*mu);
  ret(3,6) = (R2_2*R3_2)/(mu*2.0)+(R2_3*R3_3)/(mu*2.0)+(R2_1*R3_1*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(3,7) = (R2_2*R3_1)/(mu*2.0)-(R2_1*R3_2*la)/(la*mu*3.0+mu*mu);
  ret(3,8) = (R2_3*R3_1)/(mu*2.0)-(R2_1*R3_3*la)/(la*mu*3.0+mu*mu);
  ret(4,0) = (R1_2*R2_1)/(mu*2.0)-(R1_1*R2_2*la)/(la*mu*3.0+mu*mu);
  ret(4,1) = (R1_1*R2_1)/(mu*2.0)+(R1_3*R2_3)/(mu*2.0)+(R1_2*R2_2*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(4,2) = (R1_2*R2_3)/(mu*2.0)-(R1_3*R2_2*la)/(la*mu*3.0+mu*mu);
  ret(4,3) = (R2_1*R2_2)/(mu*2.0)-(R2_1*R2_2*la)/(la*mu*3.0+mu*mu);
  ret(4,4) = (R2_1*R2_1)/(mu*2.0)+(R2_3*R2_3)/(mu*2.0)+((R2_2*R2_2)*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(4,5) = (R2_2*R2_3)/(mu*2.0)-(R2_2*R2_3*la)/(la*mu*3.0+mu*mu);
  ret(4,6) = (R2_1*R3_2)/(mu*2.0)-(R2_2*R3_1*la)/(la*mu*3.0+mu*mu);
  ret(4,7) = (R2_1*R3_1)/(mu*2.0)+(R2_3*R3_3)/(mu*2.0)+(R2_2*R3_2*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(4,8) = (R2_3*R3_2)/(mu*2.0)-(R2_2*R3_3*la)/(la*mu*3.0+mu*mu);
  ret(5,0) = (R1_3*R2_1)/(mu*2.0)-(R1_1*R2_3*la)/(la*mu*3.0+mu*mu);
  ret(5,1) = (R1_3*R2_2)/(mu*2.0)-(R1_2*R2_3*la)/(la*mu*3.0+mu*mu);
  ret(5,2) = (R1_1*R2_1)/(mu*2.0)+(R1_2*R2_2)/(mu*2.0)+(R1_3*R2_3*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(5,3) = (R2_1*R2_3)/(mu*2.0)-(R2_1*R2_3*la)/(la*mu*3.0+mu*mu);
  ret(5,4) = (R2_2*R2_3)/(mu*2.0)-(R2_2*R2_3*la)/(la*mu*3.0+mu*mu);
  ret(5,5) = (R2_1*R2_1)/(mu*2.0)+(R2_2*R2_2)/(mu*2.0)+((R2_3*R2_3)*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(5,6) = (R2_1*R3_3)/(mu*2.0)-(R2_3*R3_1*la)/(la*mu*3.0+mu*mu);
  ret(5,7) = (R2_2*R3_3)/(mu*2.0)-(R2_3*R3_2*la)/(la*mu*3.0+mu*mu);
  ret(5,8) = (R2_1*R3_1)/(mu*2.0)+(R2_2*R3_2)/(mu*2.0)+(R2_3*R3_3*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(6,0) = (R1_2*R3_2)/(mu*2.0)+(R1_3*R3_3)/(mu*2.0)+(R1_1*R3_1*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(6,1) = (R1_1*R3_2)/(mu*2.0)-(R1_2*R3_1*la)/(la*mu*3.0+mu*mu);
  ret(6,2) = (R1_1*R3_3)/(mu*2.0)-(R1_3*R3_1*la)/(la*mu*3.0+mu*mu);
  ret(6,3) = (R2_2*R3_2)/(mu*2.0)+(R2_3*R3_3)/(mu*2.0)+(R2_1*R3_1*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(6,4) = (R2_1*R3_2)/(mu*2.0)-(R2_2*R3_1*la)/(la*mu*3.0+mu*mu);
  ret(6,5) = (R2_1*R3_3)/(mu*2.0)-(R2_3*R3_1*la)/(la*mu*3.0+mu*mu);
  ret(6,6) = (R3_2*R3_2)/(mu*2.0)+(R3_3*R3_3)/(mu*2.0)+((R3_1*R3_1)*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(6,7) = (R3_1*R3_2)/(mu*2.0)-(R3_1*R3_2*la)/(la*mu*3.0+mu*mu);
  ret(6,8) = (R3_1*R3_3)/(mu*2.0)-(R3_1*R3_3*la)/(la*mu*3.0+mu*mu);
  ret(7,0) = (R1_2*R3_1)/(mu*2.0)-(R1_1*R3_2*la)/(la*mu*3.0+mu*mu);
  ret(7,1) = (R1_1*R3_1)/(mu*2.0)+(R1_3*R3_3)/(mu*2.0)+(R1_2*R3_2*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(7,2) = (R1_2*R3_3)/(mu*2.0)-(R1_3*R3_2*la)/(la*mu*3.0+mu*mu);
  ret(7,3) = (R2_2*R3_1)/(mu*2.0)-(R2_1*R3_2*la)/(la*mu*3.0+mu*mu);
  ret(7,4) = (R2_1*R3_1)/(mu*2.0)+(R2_3*R3_3)/(mu*2.0)+(R2_2*R3_2*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(7,5) = (R2_2*R3_3)/(mu*2.0)-(R2_3*R3_2*la)/(la*mu*3.0+mu*mu);
  ret(7,6) = (R3_1*R3_2)/(mu*2.0)-(R3_1*R3_2*la)/(la*mu*3.0+mu*mu);
  ret(7,7) = (R3_1*R3_1)/(mu*2.0)+(R3_3*R3_3)/(mu*2.0)+((R3_2*R3_2)*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(7,8) = (R3_2*R3_3)/(mu*2.0)-(R3_2*R3_3*la)/(la*mu*3.0+mu*mu);
  ret(8,0) = (R1_3*R3_1)/(mu*2.0)-(R1_1*R3_3*la)/(la*mu*3.0+mu*mu);
  ret(8,1) = (R1_3*R3_2)/(mu*2.0)-(R1_2*R3_3*la)/(la*mu*3.0+mu*mu);
  ret(8,2) = (R1_1*R3_1)/(mu*2.0)+(R1_2*R3_2)/(mu*2.0)+(R1_3*R3_3*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(8,3) = (R2_3*R3_1)/(mu*2.0)-(R2_1*R3_3*la)/(la*mu*3.0+mu*mu);
  ret(8,4) = (R2_3*R3_2)/(mu*2.0)-(R2_2*R3_3*la)/(la*mu*3.0+mu*mu);
  ret(8,5) = (R2_1*R3_1)/(mu*2.0)+(R2_2*R3_2)/(mu*2.0)+(R2_3*R3_3*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  ret(8,6) = (R3_1*R3_3)/(mu*2.0)-(R3_1*R3_3*la)/(la*mu*3.0+mu*mu);
  ret(8,7) = (R3_2*R3_3)/(mu*2.0)-(R3_2*R3_3*la)/(la*mu*3.0+mu*mu);
  ret(8,8) = (R3_1*R3_1)/(mu*2.0)+(R3_2*R3_2)/(mu*2.0)+((R3_3*R3_3)*(la*2.0+mu))/(la*mu*3.0+mu*mu);
  return ret;
}

void corotational_compliance(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    std::vector<Eigen::Matrix3d>& R, const Eigen::VectorXd& vols, double mu,
    double la, std::vector<Eigen::Triplet<double>>& trips) {

  double offset = V.size();
  for (int i = 0; i < T.rows(); ++i) {

    Eigen::Matrix9d WHiW = corotational_WHinvW(R[i], mu, la);

    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 9; ++k) {
        trips.push_back(Eigen::Triplet<double>(
                    offset+9*i+j,offset+9*i+k, -vols(i)*WHiW(j,k)));
      }
    }
  }
}
