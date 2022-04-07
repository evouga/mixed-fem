#include "materials/material_model.h"

using namespace Eigen;
using namespace mfem;

void MaterialModel::fill_compliance_block(int offset, int row,
    double vol, double tol, const Eigen::Matrix9d& WHiW,
    Eigen::SparseMatrixd& mat) {

  // Assign to last nine entries of the j-th column for the i-th block
  for (int j = 0; j < 9; ++j) {
    int offset_j = offset + row*9 + j;
    int colsize = (mat.outerIndexPtr()[offset_j+1] 
      - mat.outerIndexPtr()[offset_j]);
    int row_j = mat.outerIndexPtr()[offset_j] + colsize - 9;

    for (int k = 0; k < 9; ++k) {
      if (k==j) {
        mat.valuePtr()[row_j+k] = -vol*(WHiW(j,k)+tol);
      } else {
        mat.valuePtr()[row_j+k] = -vol*WHiW(j,k);
      }
    }
  }
}

void MaterialModel::update_compliance(int n, int m, 
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

Matrix9d MaterialModel::WHinvW(const Matrix3d& R,
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

Vector9d MaterialModel::rhs(const Matrix3d& R,
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

Vector6d MaterialModel::dS(const Matrix3d& R, 
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
