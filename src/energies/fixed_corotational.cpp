#include "energies/fixed_corotational.h"
#include "simple_psd_fix.h"
#include "config.h"
#include "svd/svd_eigen.h"
#include "svd/iARAP.h"

using namespace Eigen;
using namespace mfem;

double FixedCorotational::energy(const Vector6d& S) {
    
  double mu = config_->mu;
  double la = config_->la;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  return (la*pow(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0,2.0))/2.0+mu*(pow(S1-1.0,2.0)+pow(S2-1.0,2.0)+pow(S3-1.0,2.0)+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0);
}

Vector6d FixedCorotational::gradient(const Vector6d& S) {
  
  double mu = config_->mu;
  double la = config_->la;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  Vector6d g;
  g(0) = mu*(S1*2.0-2.0)-la*(S2*S3-S6*S6)*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0);
  g(1) = mu*(S2*2.0-2.0)-la*(S1*S3-S5*S5)*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0);
  g(2) = mu*(S3*2.0-2.0)-la*(S1*S2-S4*S4)*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0);
  g(3) = S4*mu*4.0+la*(S3*S4*2.0-S5*S6*2.0)*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0);
  g(4) = S5*mu*4.0+la*(S2*S5*2.0-S4*S6*2.0)*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0);
  g(5) = S6*mu*4.0+la*(S1*S6*2.0-S4*S5*2.0)*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0);

  return g;

}

Matrix6d FixedCorotational::hessian(const Vector6d& S, bool psd_fix) {
  Matrix6d H;
  H.setZero();
  double mu = config_->mu;
  double la = config_->la;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  H(0,0) = mu*2.0+la*pow(S2*S3-S6*S6,2.0);
  H(0,1) = -S3*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S3-S5*S5)*(S2*S3-S6*S6);
  H(0,2) = -S2*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S2-S4*S4)*(S2*S3-S6*S6);
  H(0,3) = -la*(S3*S4*2.0-S5*S6*2.0)*(S2*S3-S6*S6);
  H(0,4) = -la*(S2*S5*2.0-S4*S6*2.0)*(S2*S3-S6*S6);
  H(0,5) = S6*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S1*S6*2.0-S4*S5*2.0)*(S2*S3-S6*S6);
  H(1,0) = -S3*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S3-S5*S5)*(S2*S3-S6*S6);
  H(1,1) = mu*2.0+la*pow(S1*S3-S5*S5,2.0);
  H(1,2) = -S1*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S2-S4*S4)*(S1*S3-S5*S5);
  H(1,3) = -la*(S3*S4*2.0-S5*S6*2.0)*(S1*S3-S5*S5);
  H(1,4) = S5*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S2*S5*2.0-S4*S6*2.0)*(S1*S3-S5*S5);
  H(1,5) = -la*(S1*S6*2.0-S4*S5*2.0)*(S1*S3-S5*S5);
  H(2,0) = -S2*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S2-S4*S4)*(S2*S3-S6*S6);
  H(2,1) = -S1*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S2-S4*S4)*(S1*S3-S5*S5);
  H(2,2) = mu*2.0+la*pow(S1*S2-S4*S4,2.0);
  H(2,3) = S4*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S3*S4*2.0-S5*S6*2.0)*(S1*S2-S4*S4);
  H(2,4) = -la*(S2*S5*2.0-S4*S6*2.0)*(S1*S2-S4*S4);
  H(2,5) = -la*(S1*S6*2.0-S4*S5*2.0)*(S1*S2-S4*S4);
  H(3,0) = -la*(S3*S4*2.0-S5*S6*2.0)*(S2*S3-S6*S6);
  H(3,1) = -la*(S3*S4*2.0-S5*S6*2.0)*(S1*S3-S5*S5);
  H(3,2) = S4*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S3*S4*2.0-S5*S6*2.0)*(S1*S2-S4*S4);
  H(3,3) = mu*4.0+la*pow(S3*S4*2.0-S5*S6*2.0,2.0)+S3*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0;
  H(3,4) = S6*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*-2.0+la*(S2*S5*2.0-S4*S6*2.0)*(S3*S4*2.0-S5*S6*2.0);
  H(3,5) = S5*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*-2.0+la*(S1*S6*2.0-S4*S5*2.0)*(S3*S4*2.0-S5*S6*2.0);
  H(4,0) = -la*(S2*S5*2.0-S4*S6*2.0)*(S2*S3-S6*S6);
  H(4,1) = S5*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S2*S5*2.0-S4*S6*2.0)*(S1*S3-S5*S5);
  H(4,2) = -la*(S2*S5*2.0-S4*S6*2.0)*(S1*S2-S4*S4);
  H(4,3) = S6*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*-2.0+la*(S2*S5*2.0-S4*S6*2.0)*(S3*S4*2.0-S5*S6*2.0);
  H(4,4) = mu*4.0+la*pow(S2*S5*2.0-S4*S6*2.0,2.0)+S2*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0;
  H(4,5) = S4*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*-2.0+la*(S1*S6*2.0-S4*S5*2.0)*(S2*S5*2.0-S4*S6*2.0);
  H(5,0) = S6*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S1*S6*2.0-S4*S5*2.0)*(S2*S3-S6*S6);
  H(5,1) = -la*(S1*S6*2.0-S4*S5*2.0)*(S1*S3-S5*S5);
  H(5,2) = -la*(S1*S6*2.0-S4*S5*2.0)*(S1*S2-S4*S4);
  H(5,3) = S5*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*-2.0+la*(S1*S6*2.0-S4*S5*2.0)*(S3*S4*2.0-S5*S6*2.0);
  H(5,4) = S4*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*-2.0+la*(S1*S6*2.0-S4*S5*2.0)*(S2*S5*2.0-S4*S6*2.0);
  H(5,5) = mu*4.0+la*pow(S1*S6*2.0-S4*S5*2.0,2.0)+S1*la*(S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0;
  sim::simple_psd_fix(H);
  return H;
}

double FixedCorotational::energy(const Eigen::Vector3d& s) {
 
  double mu = config_->mu;
  double la = config_->la;
  double S1 = s(0);
  double S2 = s(1);
  double S3 = s(2);
  return (la*pow(-S1*S2+S3*S3+1.0,2.0))/2.0+mu*(pow(S1-1.0,2.0)+pow(S2-1.0,2.0)+(S3*S3)*2.0);
}
Vector3d FixedCorotational::gradient(const Vector3d& s) {
  double mu = config_->mu;
  double la = config_->la;
  double S1 = s(0);
  double S2 = s(1);
  double S3 = s(2);
  Vector3d g;
  g(0) = mu*(S1*2.0-2.0)-S2*la*(-S1*S2+S3*S3+1.0);
  g(1) = mu*(S2*2.0-2.0)-S1*la*(-S1*S2+S3*S3+1.0);
  g(2) = S3*mu*4.0+S3*la*(-S1*S2+S3*S3+1.0)*2.0;
  return g;
}
Matrix3d FixedCorotational::hessian(const Vector3d& s) {
  double mu = config_->mu;
  double la = config_->la;
  double S1 = s(0);
  double S2 = s(1);
  double S3 = s(2);
  Matrix3d H;
  H.setZero();
  H(0,0) = mu*2.0+(S2*S2)*la;
  H(0,1) = -la*(-S1*S2+S3*S3+1.0)+S1*S2*la;
  H(0,2) = S2*S3*la*-2.0;
  H(1,0) = -la*(-S1*S2+S3*S3+1.0)+S1*S2*la;
  H(1,1) = mu*2.0+(S1*S1)*la;
  H(1,2) = S1*S3*la*-2.0;
  H(2,0) = S2*S3*la*-2.0;
  H(2,1) = S1*S3*la*-2.0;
  H(2,2) = mu*4.0+(S3*S3)*la*4.0+la*(-S1*S2+S3*S3+1.0)*2.0;
  sim::simple_psd_fix(H);
  return H;
}

double FixedCorotational::energy(const Vector4d& F) {
  double mu = config_->mu;
  double la = config_->la;

  // Compute SVD and rotation
  Matrix2d A,U,V;
  A << F(0), F(2),
       F(1), F(3);
  Vector2d sval;
  mfem::svd<double,2>(A, sval, U, V);
  Matrix2d R = U * V.transpose();

  double I3 = A.determinant();

  return mu*(A-R).squaredNorm() + la*0.5*std::pow(I3-1,2.0);
}

Vector4d FixedCorotational::gradient(const Vector4d& F) {
  double mu = config_->mu;
  double la = config_->la;

  // Compute SVD and rotation
  Matrix2d A,U,V;
  A << F(0), F(2),
       F(1), F(3);
  Vector2d sval;
  mfem::svd<double,2>(A, sval, U, V);

  double J = A.determinant();
  Matrix2d dJdF;
  dJdF << A(1,1), -A(1,0),
         -A(0,1), A(0,0);

  Matrix2d G = mu * 2.0 * (A - U * V.transpose()) + la*(J-1)*dJdF;
  Vector4d g = Map<Vector4d>(G.data());

  // P = 2 * mu * (F - R) + la * (J-1) * dJ/dF
  return g;
}

Matrix4d FixedCorotational::hessian(const Vector4d& F) {
  double mu = config_->mu;
  double la = config_->la;

  // Compute SVD and rotation
  Matrix2d A,U,V;
  A << F(0), F(2),
       F(1), F(3);
  Vector2d sval;
  mfem::svd<double,2>(A, sval, U, V);
  Matrix2d R = U * V.transpose();

  Matrix2d S;
  S << 0, -1,
       1, 0;
  Matrix2d Q0 = (1.0 /sqrt(2.0)) * U * S * V.transpose();
  Vector4d q0 = Map<Vector4d>(Q0.data());
  double l0 = 2.0 / (sval(0) + sval(1));
  Matrix4d dRdF = l0 * (q0 * q0.transpose());
  Matrix4d H = 2.0 * mu * (Matrix4d::Identity() - dRdF);

  double J = A.determinant();
  Matrix2d dJdF;
  dJdF << A(1,1), -A(1,0),
         -A(0,1), A(0,0);

  Vector4d gJ = Map<Vector4d>(dJdF.data());
  Matrix4d HJ;
  HJ << 0, 0, 0, 1,
        0, 0, -1, 0,
        0, -1, 0, 0,
        1, 0, 0, 0;

  H += la * ((J-1)*HJ + gJ*gJ.transpose());
  sim::simple_psd_fix(H);
  // dPdF = 2*mu*(I - dRdF) + la * ((J-1) * d2J/dF2 + dJ/dF*dJ/dF
  return H;
}

double FixedCorotational::energy(const Vector9d& F) {
  double mu = config_->mu;
  double la = config_->la;

  Matrix3d A = Map<const Matrix3d>(F.data());
  double f = iARAP::trace_S(A);
  double i1 = F.squaredNorm();
  return mu * 0.5 * (i1 - 2 * f + 3);
}

Vector9d FixedCorotational::gradient(const Vector9d& F) {
  double mu = config_->mu;
  double la = config_->la;

  Matrix3d A = Map<const Matrix3d>(F.data());
  Matrix3d G = mu * (A - iARAP::rotation(A));
  Vector9d g = Map<Vector9d>(G.data());

  // Volume term
  double F1_1 = F(0);
  double F2_1 = F(1);
  double F3_1 = F(2);
  double F1_2 = F(3);
  double F2_2 = F(4);
  double F3_2 = F(5);
  double F1_3 = F(6);
  double F2_3 = F(7);
  double F3_3 = F(8);
  g(0) += -la*(F2_2*F3_3-F2_3*F3_2)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  g(1) += la*(F1_2*F3_3-F1_3*F3_2)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  g(2) += -la*(F1_2*F2_3-F1_3*F2_2)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  g(3) += la*(F2_1*F3_3-F2_3*F3_1)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  g(4) += -la*(F1_1*F3_3-F1_3*F3_1)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  g(5) += la*(F1_1*F2_3-F1_3*F2_1)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  g(6) += -la*(F2_1*F3_2-F2_2*F3_1)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  g(7) += la*(F1_1*F3_2-F1_2*F3_1)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  g(8) += -la*(F1_1*F2_2-F1_2*F2_1)*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  return g;
}


Matrix9d FixedCorotational::hessian(const Vector9d& F) {
  double mu = config_->mu;
  double la = config_->la;

  // Compute SVD and rotation
  Matrix3d A = Map<const Matrix3d>(F.data());

  Matrix9d dRdF;
  Matrix3d R = iARAP::rotation(A, true, dRdF);
  Matrix9d H = mu * (Matrix9d::Identity() - dRdF);

  // Volume term
  double F1_1 = F(0);
  double F2_1 = F(1);
  double F3_1 = F(2);
  double F1_2 = F(3);
  double F2_2 = F(4);
  double F3_2 = F(5);
  double F1_3 = F(6);
  double F2_3 = F(7);
  double F3_3 = F(8);
  H(0,0) += la*pow(F2_2*F3_3-F2_3*F3_2,2.0);
  H(0,1) += -la*(F1_2*F3_3-F1_3*F3_2)*(F2_2*F3_3-F2_3*F3_2);
  H(0,2) += la*(F1_2*F2_3-F1_3*F2_2)*(F2_2*F3_3-F2_3*F3_2);
  H(0,3) += -la*(F2_1*F3_3-F2_3*F3_1)*(F2_2*F3_3-F2_3*F3_2);
  H(0,4) += la*(F1_1*F3_3-F1_3*F3_1)*(F2_2*F3_3-F2_3*F3_2)-F3_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(0,5) += -la*(F1_1*F2_3-F1_3*F2_1)*(F2_2*F3_3-F2_3*F3_2)+F2_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(0,6) += la*(F2_1*F3_2-F2_2*F3_1)*(F2_2*F3_3-F2_3*F3_2);
  H(0,7) += -la*(F1_1*F3_2-F1_2*F3_1)*(F2_2*F3_3-F2_3*F3_2)+F3_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(0,8) += la*(F1_1*F2_2-F1_2*F2_1)*(F2_2*F3_3-F2_3*F3_2)-F2_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(1,0) += -la*(F1_2*F3_3-F1_3*F3_2)*(F2_2*F3_3-F2_3*F3_2);
  H(1,1) += la*pow(F1_2*F3_3-F1_3*F3_2,2.0);
  H(1,2) += -la*(F1_2*F2_3-F1_3*F2_2)*(F1_2*F3_3-F1_3*F3_2);
  H(1,3) += la*(F1_2*F3_3-F1_3*F3_2)*(F2_1*F3_3-F2_3*F3_1)+F3_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(1,4) += -la*(F1_1*F3_3-F1_3*F3_1)*(F1_2*F3_3-F1_3*F3_2);
  H(1,5) += la*(F1_1*F2_3-F1_3*F2_1)*(F1_2*F3_3-F1_3*F3_2)-F1_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(1,6) += -la*(F1_2*F3_3-F1_3*F3_2)*(F2_1*F3_2-F2_2*F3_1)-F3_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(1,7) += la*(F1_1*F3_2-F1_2*F3_1)*(F1_2*F3_3-F1_3*F3_2);
  H(1,8) += -la*(F1_1*F2_2-F1_2*F2_1)*(F1_2*F3_3-F1_3*F3_2)+F1_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(2,0) += la*(F1_2*F2_3-F1_3*F2_2)*(F2_2*F3_3-F2_3*F3_2);
  H(2,1) += -la*(F1_2*F2_3-F1_3*F2_2)*(F1_2*F3_3-F1_3*F3_2);
  H(2,2) += la*pow(F1_2*F2_3-F1_3*F2_2,2.0);
  H(2,3) += -la*(F1_2*F2_3-F1_3*F2_2)*(F2_1*F3_3-F2_3*F3_1)-F2_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(2,4) += la*(F1_2*F2_3-F1_3*F2_2)*(F1_1*F3_3-F1_3*F3_1)+F1_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(2,5) += -la*(F1_1*F2_3-F1_3*F2_1)*(F1_2*F2_3-F1_3*F2_2);
  H(2,6) += la*(F1_2*F2_3-F1_3*F2_2)*(F2_1*F3_2-F2_2*F3_1)+F2_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(2,7) += -la*(F1_2*F2_3-F1_3*F2_2)*(F1_1*F3_2-F1_2*F3_1)-F1_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(2,8) += la*(F1_1*F2_2-F1_2*F2_1)*(F1_2*F2_3-F1_3*F2_2);
  H(3,0) += -la*(F2_1*F3_3-F2_3*F3_1)*(F2_2*F3_3-F2_3*F3_2);
  H(3,1) += la*(F1_2*F3_3-F1_3*F3_2)*(F2_1*F3_3-F2_3*F3_1)+F3_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(3,2) += -la*(F1_2*F2_3-F1_3*F2_2)*(F2_1*F3_3-F2_3*F3_1)-F2_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(3,3) += la*pow(F2_1*F3_3-F2_3*F3_1,2.0);
  H(3,4) += -la*(F1_1*F3_3-F1_3*F3_1)*(F2_1*F3_3-F2_3*F3_1);
  H(3,5) += la*(F1_1*F2_3-F1_3*F2_1)*(F2_1*F3_3-F2_3*F3_1);
  H(3,6) += -la*(F2_1*F3_2-F2_2*F3_1)*(F2_1*F3_3-F2_3*F3_1);
  H(3,7) += la*(F1_1*F3_2-F1_2*F3_1)*(F2_1*F3_3-F2_3*F3_1)-F3_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(3,8) += -la*(F1_1*F2_2-F1_2*F2_1)*(F2_1*F3_3-F2_3*F3_1)+F2_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(4,0) += la*(F1_1*F3_3-F1_3*F3_1)*(F2_2*F3_3-F2_3*F3_2)-F3_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(4,1) += -la*(F1_1*F3_3-F1_3*F3_1)*(F1_2*F3_3-F1_3*F3_2);
  H(4,2) += la*(F1_2*F2_3-F1_3*F2_2)*(F1_1*F3_3-F1_3*F3_1)+F1_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(4,3) += -la*(F1_1*F3_3-F1_3*F3_1)*(F2_1*F3_3-F2_3*F3_1);
  H(4,4) += la*pow(F1_1*F3_3-F1_3*F3_1,2.0);
  H(4,5) += -la*(F1_1*F2_3-F1_3*F2_1)*(F1_1*F3_3-F1_3*F3_1);
  H(4,6) += la*(F1_1*F3_3-F1_3*F3_1)*(F2_1*F3_2-F2_2*F3_1)+F3_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(4,7) += -la*(F1_1*F3_2-F1_2*F3_1)*(F1_1*F3_3-F1_3*F3_1);
  H(4,8) += la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F3_3-F1_3*F3_1)-F1_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(5,0) += -la*(F1_1*F2_3-F1_3*F2_1)*(F2_2*F3_3-F2_3*F3_2)+F2_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(5,1) += la*(F1_1*F2_3-F1_3*F2_1)*(F1_2*F3_3-F1_3*F3_2)-F1_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(5,2) += -la*(F1_1*F2_3-F1_3*F2_1)*(F1_2*F2_3-F1_3*F2_2);
  H(5,3) += la*(F1_1*F2_3-F1_3*F2_1)*(F2_1*F3_3-F2_3*F3_1);
  H(5,4) += -la*(F1_1*F2_3-F1_3*F2_1)*(F1_1*F3_3-F1_3*F3_1);
  H(5,5) += la*pow(F1_1*F2_3-F1_3*F2_1,2.0);
  H(5,6) += -la*(F1_1*F2_3-F1_3*F2_1)*(F2_1*F3_2-F2_2*F3_1)-F2_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(5,7) += la*(F1_1*F2_3-F1_3*F2_1)*(F1_1*F3_2-F1_2*F3_1)+F1_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(5,8) += -la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F2_3-F1_3*F2_1);
  H(6,0) += la*(F2_1*F3_2-F2_2*F3_1)*(F2_2*F3_3-F2_3*F3_2);
  H(6,1) += -la*(F1_2*F3_3-F1_3*F3_2)*(F2_1*F3_2-F2_2*F3_1)-F3_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(6,2) += la*(F1_2*F2_3-F1_3*F2_2)*(F2_1*F3_2-F2_2*F3_1)+F2_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(6,3) += -la*(F2_1*F3_2-F2_2*F3_1)*(F2_1*F3_3-F2_3*F3_1);
  H(6,4) += la*(F1_1*F3_3-F1_3*F3_1)*(F2_1*F3_2-F2_2*F3_1)+F3_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(6,5) += -la*(F1_1*F2_3-F1_3*F2_1)*(F2_1*F3_2-F2_2*F3_1)-F2_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(6,6) += la*pow(F2_1*F3_2-F2_2*F3_1,2.0);
  H(6,7) += -la*(F1_1*F3_2-F1_2*F3_1)*(F2_1*F3_2-F2_2*F3_1);
  H(6,8) += la*(F1_1*F2_2-F1_2*F2_1)*(F2_1*F3_2-F2_2*F3_1);
  H(7,0) += -la*(F1_1*F3_2-F1_2*F3_1)*(F2_2*F3_3-F2_3*F3_2)+F3_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(7,1) += la*(F1_1*F3_2-F1_2*F3_1)*(F1_2*F3_3-F1_3*F3_2);
  H(7,2) += -la*(F1_2*F2_3-F1_3*F2_2)*(F1_1*F3_2-F1_2*F3_1)-F1_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(7,3) += la*(F1_1*F3_2-F1_2*F3_1)*(F2_1*F3_3-F2_3*F3_1)-F3_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(7,4) += -la*(F1_1*F3_2-F1_2*F3_1)*(F1_1*F3_3-F1_3*F3_1);
  H(7,5) += la*(F1_1*F2_3-F1_3*F2_1)*(F1_1*F3_2-F1_2*F3_1)+F1_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(7,6) += -la*(F1_1*F3_2-F1_2*F3_1)*(F2_1*F3_2-F2_2*F3_1);
  H(7,7) += la*pow(F1_1*F3_2-F1_2*F3_1,2.0);
  H(7,8) += -la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F3_2-F1_2*F3_1);
  H(8,0) += la*(F1_1*F2_2-F1_2*F2_1)*(F2_2*F3_3-F2_3*F3_2)-F2_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(8,1) += -la*(F1_1*F2_2-F1_2*F2_1)*(F1_2*F3_3-F1_3*F3_2)+F1_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(8,2) += la*(F1_1*F2_2-F1_2*F2_1)*(F1_2*F2_3-F1_3*F2_2);
  H(8,3) += -la*(F1_1*F2_2-F1_2*F2_1)*(F2_1*F3_3-F2_3*F3_1)+F2_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(8,4) += la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F3_3-F1_3*F3_1)-F1_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
  H(8,5) += -la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F2_3-F1_3*F2_1);
  H(8,6) += la*(F1_1*F2_2-F1_2*F2_1)*(F2_1*F3_2-F2_2*F3_1);
  H(8,7) += -la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F3_2-F1_2*F3_1);
  H(8,8) += la*pow(F1_1*F2_2-F1_2*F2_1,2.0);
  sim::simple_psd_fix(H);

  return H;
}