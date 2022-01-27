#include "arap.h"

using namespace Eigen;

void arap_compliance(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    const std::vector<Eigen::Matrix3d>& R, const Eigen::VectorXd& vols,
    double mu, double la, std::vector<Eigen::Triplet<double>>& trips) {
  int offset = V.size();

  for (int i = 0; i < T.rows(); ++i) {
    Eigen::Matrix9d WHiW = arap_WHinvW(R[i], mu, la);
    //Eigen::Matrix9d WHiW2 = arap_WHinvW2(R[i], mu, la);
    //if (i == 0) {
    //  std::cout << "WHIW1: \n" << WHiW << std::endl;
    //  std::cout << "WHIW2: \n" << WHiW2 << std::endl;
    //}
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 9; ++k) {
        if (k==j) {
        trips.push_back(Eigen::Triplet<double>(
                    offset+9*i+j,offset+9*i+k, -vols(i)*(WHiW(j,k)+1e-6)));
        } else {
        trips.push_back(Eigen::Triplet<double>(
                    offset+9*i+j,offset+9*i+k, -vols(i)*WHiW(j,k)));
        }
        //trips.push_back(Eigen::Triplet<double>(
        //            offset+9*i+j,offset+9*i+k, -vols(i)*WHiW(j,k)));
      }
    }
  }
}

void update_arap_compliance(int n, int m,
    std::vector<Eigen::Matrix3d>& R,
    const Eigen::VectorXd& vols, double mu, double la,
    SparseMatrixd& mat) {

  int offset = n;
  #pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    Eigen::Matrix9d WHiW = arap_WHinvW(R[i], mu, la);


    // Assign to last nine entries of the j-th column for the i-th block
    for (int j = 0; j < 9; ++j) {
      int offset_j = offset + i*9 + j;
      int colsize = (mat.outerIndexPtr()[offset_j+1] 
        - mat.outerIndexPtr()[offset_j]);

      //std::cout << "offset: " << offset <<  " i: " << i << " j " << j << " idx: " << 
      //  mat.outerIndexPtr()[offset + i*9 + j] << std::endl;
      //std::cout << "NNZ: " << colsize << std::endl;
      int row_j = mat.outerIndexPtr()[offset_j] + colsize - 9;
      // 100 + 10 - 9 
      for (int k = 0; k < 9; ++k) {
        if (k==j) {
          mat.valuePtr()[row_j+k] = -vols(i)*(WHiW(j,k)+1e-6);
        } else {
          mat.valuePtr()[row_j+k] = -vols(i)*WHiW(j,k);
        }
      }
    }
  }
}

Vector6d arap_ds(const Matrix3d& R, const Vector6d& S,
        const Vector9d& L, double mu, double la) {

  Vector6d ds;
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
  ds(0) = (mu*(S1*2.0-2.0)*(-1.0/2.0)+L1*R1_1+L4*R2_1+L7*R3_1)/mu;
  ds(1) = (mu*(S2*2.0-2.0)*(-1.0/2.0)+L2*R1_2+L5*R2_2+L8*R3_2)/mu;
  ds(2) = (mu*(S3*2.0-2.0)*(-1.0/2.0)+L3*R1_3+L6*R2_3+L9*R3_3)/mu;
  ds(3) = (S4*mu*-2.0+L2*R1_3+L3*R1_2+L5*R2_3+L6*R2_2+L8*R3_3+L9*R3_2)/(mu*2.0);
  ds(4) = (S5*mu*-2.0+L1*R1_3+L3*R1_1+L4*R2_3+L6*R2_1+L7*R3_3+L9*R3_1)/(mu*2.0);
  ds(5) = (S6*mu*-2.0+L1*R1_2+L2*R1_1+L4*R2_2+L5*R2_1+L7*R3_2+L8*R3_1)/(mu*2.0);
  return ds;
}

Vector9d arap_rhs(const Matrix3d& R) {
  double R1_1 = R(0,0);
  double R1_2 = R(0,1);
  double R1_3 = R(0,2);
  double R2_1 = R(1,0);
  double R2_2 = R(1,1);
  double R2_3 = R(1,2);
  double R3_1 = R(2,0);
  double R3_2 = R(2,1);
  double R3_3 = R(2,2);
  Vector9d rhs;
  rhs(0) = R1_1;
  rhs(1) = R1_2;
  rhs(2) = R1_3;
  rhs(3) = R2_1;
  rhs(4) = R2_2;
  rhs(5) = R2_3;
  rhs(6) = R3_1;
  rhs(7) = R3_2;
  rhs(8) = R3_3;
  return rhs;
}

Eigen::Matrix9d arap_WHinvW2(const Matrix3d& R, double mu, double la) {
  Matrix<double,9,6> W;
  W <<
    R(0,0), 0,         0,         0,         R(0,2), R(0,1),
    0,         R(0,1), 0,         R(0,2), 0,         R(0,0),
    0,         0,         R(0,2), R(0,1), R(0,0), 0, 
    R(1,0), 0,         0,         0,         R(1,2), R(1,1),
    0,         R(1,1), 0,         R(1,2), 0,         R(1,0),
    0,         0,         R(1,2), R(1,1), R(1,0), 0,  
    R(2,0), 0,         0,         0,         R(2,2), R(2,1),
    0,         R(2,1), 0,         R(2,2), 0        , R(2,0),
    0,         0,         R(2,2), R(2,1), R(2,0), 0;
  // Local hessian (constant for all elements)
  Vector6d He_inv_vec;
  He_inv_vec << 1,1,1,0.5,0.5,0.5;
  He_inv_vec /= mu;
  DiagonalMatrix<double,6> He_inv = He_inv_vec.asDiagonal();
  return W*He_inv*W.transpose();

}

Eigen::Matrix9d arap_WHinvW(const Eigen::Matrix3d& R, double mu, double la) {
  double R1_1 = R(0,0);
  double R1_2 = R(0,1);
  double R1_3 = R(0,2);
  double R2_1 = R(1,0);
  double R2_2 = R(1,1);
  double R2_3 = R(1,2);
  double R3_1 = R(2,0);
  double R3_2 = R(2,1);
  double R3_3 = R(2,2);
  Matrix9d WHW;
  WHW(0,0) = (R1_1*R1_1)/mu+(R1_2*R1_2)/(mu*2.0)+(R1_3*R1_3)/(mu*2.0);
  WHW(0,1) = (R1_1*R1_2)/(mu*2.0);
  WHW(0,2) = (R1_1*R1_3)/(mu*2.0);
  WHW(0,3) = (R1_1*R2_1)/mu+(R1_2*R2_2)/(mu*2.0)+(R1_3*R2_3)/(mu*2.0);
  WHW(0,4) = (R1_2*R2_1)/(mu*2.0);
  WHW(0,5) = (R1_3*R2_1)/(mu*2.0);
  WHW(0,6) = (R1_1*R3_1)/mu+(R1_2*R3_2)/(mu*2.0)+(R1_3*R3_3)/(mu*2.0);
  WHW(0,7) = (R1_2*R3_1)/(mu*2.0);
  WHW(0,8) = (R1_3*R3_1)/(mu*2.0);
  WHW(1,0) = (R1_1*R1_2)/(mu*2.0);
  WHW(1,1) = (R1_1*R1_1)/(mu*2.0)+(R1_2*R1_2)/mu+(R1_3*R1_3)/(mu*2.0);
  WHW(1,2) = (R1_2*R1_3)/(mu*2.0);
  WHW(1,3) = (R1_1*R2_2)/(mu*2.0);
  WHW(1,4) = (R1_1*R2_1)/(mu*2.0)+(R1_2*R2_2)/mu+(R1_3*R2_3)/(mu*2.0);
  WHW(1,5) = (R1_3*R2_2)/(mu*2.0);
  WHW(1,6) = (R1_1*R3_2)/(mu*2.0);
  WHW(1,7) = (R1_1*R3_1)/(mu*2.0)+(R1_2*R3_2)/mu+(R1_3*R3_3)/(mu*2.0);
  WHW(1,8) = (R1_3*R3_2)/(mu*2.0);
  WHW(2,0) = (R1_1*R1_3)/(mu*2.0);
  WHW(2,1) = (R1_2*R1_3)/(mu*2.0);
  WHW(2,2) = (R1_1*R1_1)/(mu*2.0)+(R1_2*R1_2)/(mu*2.0)+(R1_3*R1_3)/mu;
  WHW(2,3) = (R1_1*R2_3)/(mu*2.0);
  WHW(2,4) = (R1_2*R2_3)/(mu*2.0);
  WHW(2,5) = (R1_1*R2_1)/(mu*2.0)+(R1_2*R2_2)/(mu*2.0)+(R1_3*R2_3)/mu;
  WHW(2,6) = (R1_1*R3_3)/(mu*2.0);
  WHW(2,7) = (R1_2*R3_3)/(mu*2.0);
  WHW(2,8) = (R1_1*R3_1)/(mu*2.0)+(R1_2*R3_2)/(mu*2.0)+(R1_3*R3_3)/mu;
  WHW(3,0) = (R1_1*R2_1)/mu+(R1_2*R2_2)/(mu*2.0)+(R1_3*R2_3)/(mu*2.0);
  WHW(3,1) = (R1_1*R2_2)/(mu*2.0);
  WHW(3,2) = (R1_1*R2_3)/(mu*2.0);
  WHW(3,3) = (R2_1*R2_1)/mu+(R2_2*R2_2)/(mu*2.0)+(R2_3*R2_3)/(mu*2.0);
  WHW(3,4) = (R2_1*R2_2)/(mu*2.0);
  WHW(3,5) = (R2_1*R2_3)/(mu*2.0);
  WHW(3,6) = (R2_1*R3_1)/mu+(R2_2*R3_2)/(mu*2.0)+(R2_3*R3_3)/(mu*2.0);
  WHW(3,7) = (R2_2*R3_1)/(mu*2.0);
  WHW(3,8) = (R2_3*R3_1)/(mu*2.0);
  WHW(4,0) = (R1_2*R2_1)/(mu*2.0);
  WHW(4,1) = (R1_1*R2_1)/(mu*2.0)+(R1_2*R2_2)/mu+(R1_3*R2_3)/(mu*2.0);
  WHW(4,2) = (R1_2*R2_3)/(mu*2.0);
  WHW(4,3) = (R2_1*R2_2)/(mu*2.0);
  WHW(4,4) = (R2_1*R2_1)/(mu*2.0)+(R2_2*R2_2)/mu+(R2_3*R2_3)/(mu*2.0);
  WHW(4,5) = (R2_2*R2_3)/(mu*2.0);
  WHW(4,6) = (R2_1*R3_2)/(mu*2.0);
  WHW(4,7) = (R2_1*R3_1)/(mu*2.0)+(R2_2*R3_2)/mu+(R2_3*R3_3)/(mu*2.0);
  WHW(4,8) = (R2_3*R3_2)/(mu*2.0);
  WHW(5,0) = (R1_3*R2_1)/(mu*2.0);
  WHW(5,1) = (R1_3*R2_2)/(mu*2.0);
  WHW(5,2) = (R1_1*R2_1)/(mu*2.0)+(R1_2*R2_2)/(mu*2.0)+(R1_3*R2_3)/mu;
  WHW(5,3) = (R2_1*R2_3)/(mu*2.0);
  WHW(5,4) = (R2_2*R2_3)/(mu*2.0);
  WHW(5,5) = (R2_1*R2_1)/(mu*2.0)+(R2_2*R2_2)/(mu*2.0)+(R2_3*R2_3)/mu;
  WHW(5,6) = (R2_1*R3_3)/(mu*2.0);
  WHW(5,7) = (R2_2*R3_3)/(mu*2.0);
  WHW(5,8) = (R2_1*R3_1)/(mu*2.0)+(R2_2*R3_2)/(mu*2.0)+(R2_3*R3_3)/mu;
  WHW(6,0) = (R1_1*R3_1)/mu+(R1_2*R3_2)/(mu*2.0)+(R1_3*R3_3)/(mu*2.0);
  WHW(6,1) = (R1_1*R3_2)/(mu*2.0);
  WHW(6,2) = (R1_1*R3_3)/(mu*2.0);
  WHW(6,3) = (R2_1*R3_1)/mu+(R2_2*R3_2)/(mu*2.0)+(R2_3*R3_3)/(mu*2.0);
  WHW(6,4) = (R2_1*R3_2)/(mu*2.0);
  WHW(6,5) = (R2_1*R3_3)/(mu*2.0);
  WHW(6,6) = (R3_1*R3_1)/mu+(R3_2*R3_2)/(mu*2.0)+(R3_3*R3_3)/(mu*2.0);
  WHW(6,7) = (R3_1*R3_2)/(mu*2.0);
  WHW(6,8) = (R3_1*R3_3)/(mu*2.0);
  WHW(7,0) = (R1_2*R3_1)/(mu*2.0);
  WHW(7,1) = (R1_1*R3_1)/(mu*2.0)+(R1_2*R3_2)/mu+(R1_3*R3_3)/(mu*2.0);
  WHW(7,2) = (R1_2*R3_3)/(mu*2.0);
  WHW(7,3) = (R2_2*R3_1)/(mu*2.0);
  WHW(7,4) = (R2_1*R3_1)/(mu*2.0)+(R2_2*R3_2)/mu+(R2_3*R3_3)/(mu*2.0);
  WHW(7,5) = (R2_2*R3_3)/(mu*2.0);
  WHW(7,6) = (R3_1*R3_2)/(mu*2.0);
  WHW(7,7) = (R3_1*R3_1)/(mu*2.0)+(R3_2*R3_2)/mu+(R3_3*R3_3)/(mu*2.0);
  WHW(7,8) = (R3_2*R3_3)/(mu*2.0);
  WHW(8,0) = (R1_3*R3_1)/(mu*2.0);
  WHW(8,1) = (R1_3*R3_2)/(mu*2.0);
  WHW(8,2) = (R1_1*R3_1)/(mu*2.0)+(R1_2*R3_2)/(mu*2.0)+(R1_3*R3_3)/mu;
  WHW(8,3) = (R2_3*R3_1)/(mu*2.0);
  WHW(8,4) = (R2_3*R3_2)/(mu*2.0);
  WHW(8,5) = (R2_1*R3_1)/(mu*2.0)+(R2_2*R3_2)/(mu*2.0)+(R2_3*R3_3)/mu;
  WHW(8,6) = (R3_1*R3_3)/(mu*2.0);
  WHW(8,7) = (R3_2*R3_3)/(mu*2.0);
  WHW(8,8) = (R3_1*R3_1)/(mu*2.0)+(R3_2*R3_2)/(mu*2.0)+(R3_3*R3_3)/mu;
  return WHW;
}
