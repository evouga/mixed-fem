#include "neohookean.h"
#include <string>
#include <fstream>
#include <iostream>
#include <utility>

#include <eigen3/unsupported/Eigen/src/SparseExtra/MarketIO.h>

using namespace Eigen;

void neohookean_compliance(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    std::vector<Eigen::Matrix3d>& R, std::vector<Eigen::Matrix6d>& Hinv,
    const Eigen::VectorXd& vols, double mu,
    double la, std::vector<Eigen::Triplet<double>>& trips) {

  int offset = V.size();
  for (int i = 0; i < T.rows(); ++i) {

    Eigen::Matrix9d WHiW = neohookean_WHinvW(R[i], Hinv[i]);

    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 9; ++k) {
        if (k==j) {
          trips.push_back(Eigen::Triplet<double>(
              offset+9*i+j,offset+9*i+k, -vols(i)*(WHiW(j,k)+1e-6)));
        } else {
          trips.push_back(Eigen::Triplet<double>(
              offset+9*i+j,offset+9*i+k, -vols(i)*WHiW(j,k)));
        }
      }
    }
  }
}

void update_neohookean_compliance(int n, int m,
    std::vector<Eigen::Matrix3d>& R, std::vector<Eigen::Matrix6d>& Hinv,
    const Eigen::VectorXd& vols, double mu, double la,
    SparseMatrixd& mat) {

  int offset = n;
  #pragma omp parallel for
  for (int i = 0; i < m; ++i) {

    Eigen::Matrix9d WHiW = neohookean_WHinvW(R[i], Hinv[i]);

    // Assign to last nine entries of the j-th column for the i-th block
    for (int j = 0; j < 9; ++j) {
      int offset_j = offset + i*9 + j;
      int colsize = (mat.outerIndexPtr()[offset_j+1] 
        - mat.outerIndexPtr()[offset_j]);

      //std::cout << "offset: " << offset <<  " i: " << i << " j " << j << " idx: " << 
      //  mat.outerIndexPtr()[offset + i*9 + j] << std::endl;
      //std::cout << "NNZ: " << colsize << std::endl;
      int row_j = mat.outerIndexPtr()[offset_j] + colsize - 9;
      for (int k = 0; k < 9; ++k) {
        if (k==j) {
          mat.valuePtr()[row_j+k] = -vols(i)*(WHiW(j,k)+1e-5);
        } else {
          mat.valuePtr()[row_j+k] = -vols(i)*WHiW(j,k);
        }
      }
    }
  }

  //right out matrix here
  bool did_it_write = saveMarket(mat, "./lhs.txt");
  exit(0);

}

Eigen::Vector6d neohookean_ds(const Eigen::Matrix3d& R, 
        const Eigen::Vector6d& S, const Eigen::Vector9d& L,
        const Eigen::Matrix6d& Hinv,
        double mu, double la) {

  Vector6d g = neohookean_g(R,S,mu,la);
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

Eigen::Matrix9d neohookean_WHinvW(const Eigen::Matrix3d& R,
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
Eigen::Vector9d neohookean_rhs(const Eigen::Matrix3d& R,
        const Eigen::Vector6d& S, const Matrix6d& Hinv,
        double mu, double la) {
  //Matrix6d H = neohookean_hinv(R,S,mu,la);
  Vector6d g = neohookean_g(R,S,mu,la);
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

Vector6d neohookean_g(const Matrix3d& R, const Vector6d& S, double mu, double la) {
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

Matrix6d neohookean_hinv(const Matrix3d& R, const Vector6d& S, double mu, double la) {
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

  return H.inverse();
}
