#include "energies/arap.h"
#include "config.h"
#include "svd/svd_eigen.h"
#include "svd/iARAP.h"

using namespace Eigen;
using namespace mfem;

double ARAP::energy(const Vector6d& S) {
    
  double mu = config_->mu;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  return (mu*(pow(S1-1.0,2.0)+pow(S2-1.0,2.0)+pow(S3-1.0,2.0)+(S4*S4)*2.0
        +(S5*S5)*2.0+(S6*S6)*2.0))/2.0;
}

Vector6d ARAP::gradient(const Vector6d& S) {
  
  double mu = config_->mu;
  double la = config_->la;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  Vector6d g;
  g(0) = (mu*(S1*2.0-2.0))/2.0;
  g(1) = (mu*(S2*2.0-2.0))/2.0;
  g(2) = (mu*(S3*2.0-2.0))/2.0;
  g(3) = S4*mu*2.0;
  g(4) = S5*mu*2.0;
  g(5) = S6*mu*2.0;
  return g;

}

Matrix6d ARAP::hessian_inv(const Vector6d& S) {
  
  double mu = config_->mu;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  Matrix6d Hinv;
  Hinv.setZero();
  Hinv(0,0) = 1. / mu;
  Hinv(1,1) = 1. / mu;
  Hinv(2,2) = 1. / mu;
  Hinv(3,3) = 1. / (2.0 * mu);
  Hinv(4,4) = 1. / (2.0 * mu);
  Hinv(5,5) = 1. / (2.0 * mu);
  return Hinv;
}

Matrix6d ARAP::hessian(const Vector6d& S, bool psd_fix) {
  double mu = config_->mu;
  Vector6d tmp; tmp << 1,1,1,2,2,2;
  return tmp.asDiagonal() * mu;
}

double ARAP::energy(const Vector3d& s) {
  double mu = config_->mu;
  double S1 = s(0);
  double S2 = s(1);
  double S3 = s(2);
  return (mu*(pow(S1-1.0,2.0)+pow(S2-1.0,2.0)+(S3*S3)*2.0))/2.0;;
}

Vector3d ARAP::gradient(const Vector3d& s) {
  double mu = config_->mu;
  double S1 = s(0);
  double S2 = s(1);
  double S3 = s(2);
  Vector3d g;
  g(0) = (mu*(S1*2.0-2.0))/2.0;
  g(1) = (mu*(S2*2.0-2.0))/2.0;
  g(2) = S3*mu*2.0;
  return g;
}

Matrix3d ARAP::hessian(const Vector3d& s) {
  double mu = config_->mu;
  double S1 = s(0);
  double S2 = s(1);
  double S3 = s(2);
  Matrix3d H;
  H.setZero();
  H(0,0) = mu;
  H(1,1) = mu;
  H(2,2) = mu*2.0;
  return H;
}

double ARAP::energy(const Vector4d& F) {
  double mu = config_->mu;

  // Compute SVD and rotation
  Matrix2d A,U,V;
  A << F(0), F(2),
       F(1), F(3);
  Vector2d sval;
  mfem::svd<double,2>(A, sval, U, V);
  Matrix2d R = U * V.transpose();

  return mu*0.5*(A-R).squaredNorm();
}

Vector4d ARAP::gradient(const Vector4d& F) {
  double mu = config_->mu;

  // Compute SVD and rotation
  Matrix2d A,U,V;
  A << F(0), F(2),
       F(1), F(3);
  Vector2d sval;
  mfem::svd<double,2>(A, sval, U, V);

  Matrix2d G = mu * (A - U * V.transpose());
  Vector4d g = Map<Vector4d>(G.data());
  // std::cout << " g: " << g << std::endl;
  return g;
}

Matrix4d ARAP::hessian(const Vector4d& F) {
  double mu = config_->mu;

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
  Matrix4d H = mu * (Matrix4d::Identity() - dRdF);
  return H;
}

double ARAP::energy(const Vector9d& F) {
  double mu = config_->mu;

  Matrix3d A = Map<const Matrix3d>(F.data());
  double f = iARAP::trace_S(A);
  double i1 = F.squaredNorm();
  return mu * 0.5 * (i1 - 2 * f + 3);
}

Vector9d ARAP::gradient(const Vector9d& F) {
  double mu = config_->mu;

  Matrix3d A = Map<const Matrix3d>(F.data());
  Matrix3d G = mu * (A - iARAP::rotation(A));
  Vector9d g = Map<Vector9d>(G.data());
  return g;
}


Matrix9d ARAP::hessian(const Vector9d& F) {
  double mu = config_->mu;

  // Compute SVD and rotation
  Matrix3d A = Map<const Matrix3d>(F.data());

  Matrix9d dRdF;
  Matrix3d R = iARAP::rotation(A, true, dRdF);
  Matrix9d H = mu * (Matrix9d::Identity() - dRdF);
  return H;
}