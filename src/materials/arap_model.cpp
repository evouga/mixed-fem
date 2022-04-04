#include "materials/arap_model.h"

using namespace Eigen;
using namespace mfem;

void ArapModel::update_compliance(int n, int m, 
    const std::vector<Matrix3d>& R,
    const std::vector<Matrix6d>& Hinv,
    const VectorXd& vols, SparseMatrixd& mat) {

  int offset = n;

  double mu = config_->mu;
  double tol = std::min(1e-6, 1./mu);

  #pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    Matrix9d WHiW = WHinvW(R[i], Hinv[i]);
    MaterialModel::fill_compliance_block(offset, i, vols(i), tol, WHiW, mat);
  }

  //write out matrix here
  //bool did_it_write = saveMarket(mat, "./lhs.txt");
  //exit(0);
}

double ArapModel::energy(const Vector6d& S) {
    
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

Vector6d ArapModel::gradient(const Matrix3d& R,
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
  g(0) = (mu*(S1*2.0-2.0))/2.0;
  g(1) = (mu*(S2*2.0-2.0))/2.0;
  g(2) = (mu*(S3*2.0-2.0))/2.0;
  g(3) = S4*mu*2.0;
  g(4) = S5*mu*2.0;
  g(5) = S6*mu*2.0;
  return g;

}

Matrix6d ArapModel::hessian_inv(const Matrix3d& R,
    const Vector6d& S) {
  
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

Matrix9d ArapModel::WHinvW(const Matrix3d& R,
    const Matrix6d& Hinv) {
  Matrix<double,9,6> W;
  W <<
    R(0,0), 0,         0,         0,         R(0,2), R(0,1),
    0,         R(0,1), 0,         R(0,2), 0,         R(0,0),
    0,         0,         R(0,2), R(0,1), R(0,0), 0, 
    R(1,0), 0,         0,         0,         R(1,2), R(1,1),
    0,         R(1,1), 0,         R(1,2), 0,         R(1,0),
    0,         0,         R(1,2), R(1,1), R(1,0)   , 0,  
    R(2,0), 0,         0,         0,         R(2,2), R(2,1),
    0,         R(2,1), 0,         R(2,2), 0        , R(2,0),
    0,         0,         R(2,2), R(2,1), R(2,0)   , 0;
  return W*Hinv*W.transpose();
}

Vector9d ArapModel::rhs(const Matrix3d& R,
    const Vector6d& S, const Matrix6d& Hinv,
    const Vector6d& g) {
  
  Matrix<double,9,6> W;
  W <<
    R(0,0), 0,         0,         0,         R(0,2), R(0,1),
    0,         R(0,1), 0,         R(0,2), 0,         R(0,0),
    0,         0,         R(0,2), R(0,1), R(0,0), 0, 
    R(1,0), 0,         0,         0,         R(1,2), R(1,1),
    0,         R(1,1), 0,         R(1,2), 0,         R(1,0),
    0,         0,         R(1,2), R(1,1), R(1,0)   , 0,  
    R(2,0), 0,         0,         0,         R(2,2), R(2,1),
    0,         R(2,1), 0,         R(2,2), 0        , R(2,0),
    0,         0,         R(2,2), R(2,1), R(2,0)   , 0;
  return W*(S - Hinv*g);    
}

Vector6d ArapModel::dS(const Matrix3d& R, 
    const Vector6d& S, const Vector9d& L,
    const Matrix6d& Hinv) {
  
  Vector6d g = gradient(R,S);
  Matrix<double,9,6> W;
  W <<
    R(0,0), 0,         0,         0,         R(0,2), R(0,1),
    0,         R(0,1), 0,         R(0,2), 0,         R(0,0),
    0,         0,         R(0,2), R(0,1), R(0,0), 0, 
    R(1,0), 0,         0,         0,         R(1,2), R(1,1),
    0,         R(1,1), 0,         R(1,2), 0,         R(1,0),
    0,         0,         R(1,2), R(1,1), R(1,0)   , 0,  
    R(2,0), 0,         0,         0,         R(2,2), R(2,1),
    0,         R(2,1), 0,         R(2,2), 0        , R(2,0),
    0,         0,         R(2,2), R(2,1), R(2,0)   , 0;
  return Hinv*(W.transpose()*L - g); 
}
    
