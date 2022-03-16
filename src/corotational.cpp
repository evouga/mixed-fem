#include "corotational.h"

using namespace Eigen;

void corotational_compliance(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    std::vector<Eigen::Matrix3d>& R, const Eigen::VectorXd& vols, double mu,
    double la, std::vector<Eigen::Triplet<double>>& trips) {

  int offset = V.size();
  for (int i = 0; i < T.rows(); ++i) {

    Eigen::Matrix9d WHiW = corotational_WHinvW(R[i], mu, la);

    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 9; ++k) {
        if (k==j) {
          trips.push_back(Eigen::Triplet<double>(
              offset+9*i+j,offset+9*i+k, -vols(i)*(WHiW(j,k)+1e-5)));
        } else {
          trips.push_back(Eigen::Triplet<double>(
              offset+9*i+j,offset+9*i+k, -vols(i)*WHiW(j,k)));
        }
      }
    }
  }
}

void update_corotational_compliance(int n, int m,
    std::vector<Eigen::Matrix3d>& R,
    const Eigen::VectorXd& vols, double mu, double la,
    SparseMatrixd& mat) {

  int offset = n;
  #pragma omp parallel for
  for (int i = 0; i < m; ++i) {

    Eigen::Matrix9d WHiW = corotational_WHinvW(R[i], mu, la);


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
          mat.valuePtr()[row_j+k] = -vols(i)*(WHiW(j,k)+1e-5);
        } else {
          mat.valuePtr()[row_j+k] = -vols(i)*WHiW(j,k);
        }
      }
    }
  }
}


Vector6d corotational_ds(const Matrix3d& R, const Vector6d& S, const Vector9d& L,
    double mu, double la) {

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
  ds(0) = ((la+mu)*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-mu*(S1*2.0-2.0)+L1*R1_1+L4*R2_1+L7*R3_1))/(la*mu*3.0+(mu*mu)*2.0)-(la*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-mu*(S2*2.0-2.0)+L2*R1_2+L5*R2_2+L8*R3_2))/(la*mu*6.0+(mu*mu)*4.0)-(la*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-mu*(S3*2.0-2.0)+L3*R1_3+L6*R2_3+L9*R3_3))/(la*mu*6.0+(mu*mu)*4.0);
  ds(1) = ((la+mu)*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-mu*(S2*2.0-2.0)+L2*R1_2+L5*R2_2+L8*R3_2))/(la*mu*3.0+(mu*mu)*2.0)-(la*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-mu*(S1*2.0-2.0)+L1*R1_1+L4*R2_1+L7*R3_1))/(la*mu*6.0+(mu*mu)*4.0)-(la*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-mu*(S3*2.0-2.0)+L3*R1_3+L6*R2_3+L9*R3_3))/(la*mu*6.0+(mu*mu)*4.0);
  ds(2) = ((la+mu)*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-mu*(S3*2.0-2.0)+L3*R1_3+L6*R2_3+L9*R3_3))/(la*mu*3.0+(mu*mu)*2.0)-(la*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-mu*(S1*2.0-2.0)+L1*R1_1+L4*R2_1+L7*R3_1))/(la*mu*6.0+(mu*mu)*4.0)-(la*(la*(S1*2.0+S2*2.0+S3*2.0-6.0)*(-1.0/2.0)-mu*(S2*2.0-2.0)+L2*R1_2+L5*R2_2+L8*R3_2))/(la*mu*6.0+(mu*mu)*4.0);
  ds(3) = (S4*mu*-4.0+L2*R1_3+L3*R1_2+L5*R2_3+L6*R2_2+L8*R3_3+L9*R3_2)/(mu*4.0);
  ds(4) = (S5*mu*-4.0+L1*R1_3+L3*R1_1+L4*R2_3+L6*R2_1+L7*R3_3+L9*R3_1)/(mu*4.0);
  ds(5) = (S6*mu*-4.0+L1*R1_2+L2*R1_1+L4*R2_2+L5*R2_1+L7*R3_2+L8*R3_1)/(mu*4.0);
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
g(0) = R1_1*(S1-((la+mu)*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S1*2.0-2.0)))/(la*mu*3.0+(mu*mu)*2.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S2*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S3*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0));
g(1) = R1_2*(S2-((la+mu)*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S2*2.0-2.0)))/(la*mu*3.0+(mu*mu)*2.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S1*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S3*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0));
g(2) = R1_3*(S3-((la+mu)*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S3*2.0-2.0)))/(la*mu*3.0+(mu*mu)*2.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S1*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S2*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0));
g(3)= R2_1*(S1-((la+mu)*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S1*2.0-2.0)))/(la*mu*3.0+(mu*mu)*2.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S2*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S3*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0));
g(4) = R2_2*(S2-((la+mu)*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S2*2.0-2.0)))/(la*mu*3.0+(mu*mu)*2.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S1*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S3*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0));
g(5) = R2_3*(S3-((la+mu)*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S3*2.0-2.0)))/(la*mu*3.0+(mu*mu)*2.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S1*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S2*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0));
g(6) = R3_1*(S1-((la+mu)*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S1*2.0-2.0)))/(la*mu*3.0+(mu*mu)*2.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S2*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S3*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0));
g(7) = R3_2*(S2-((la+mu)*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S2*2.0-2.0)))/(la*mu*3.0+(mu*mu)*2.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S1*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S3*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0));
g(8) = R3_3*(S3-((la+mu)*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S3*2.0-2.0)))/(la*mu*3.0+(mu*mu)*2.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S1*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0)+(la*((la*(S1*2.0+S2*2.0+S3*2.0-6.0))/2.0+mu*(S2*2.0-2.0)))/(la*mu*6.0+(mu*mu)*4.0));
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
  ret(0,0) = (R1_2*R1_2)/(mu*4.0)+(R1_3*R1_3)/(mu*4.0)+((R1_1*R1_1)*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(0,1) = (R1_1*R1_2)/(mu*4.0)-(R1_1*R1_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(0,2) = (R1_1*R1_3)/(mu*4.0)-(R1_1*R1_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(0,3) = (R1_2*R2_2)/(mu*4.0)+(R1_3*R2_3)/(mu*4.0)+(R1_1*R2_1*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(0,4) = (R1_2*R2_1)/(mu*4.0)-(R1_1*R2_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(0,5) = (R1_3*R2_1)/(mu*4.0)-(R1_1*R2_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(0,6) = (R1_2*R3_2)/(mu*4.0)+(R1_3*R3_3)/(mu*4.0)+(R1_1*R3_1*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(0,7) = (R1_2*R3_1)/(mu*4.0)-(R1_1*R3_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(0,8) = (R1_3*R3_1)/(mu*4.0)-(R1_1*R3_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(1,0) = (R1_1*R1_2)/(mu*4.0)-(R1_1*R1_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(1,1) = (R1_1*R1_1)/(mu*4.0)+(R1_3*R1_3)/(mu*4.0)+((R1_2*R1_2)*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(1,2) = (R1_2*R1_3)/(mu*4.0)-(R1_2*R1_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(1,3) = (R1_1*R2_2)/(mu*4.0)-(R1_2*R2_1*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(1,4) = (R1_1*R2_1)/(mu*4.0)+(R1_3*R2_3)/(mu*4.0)+(R1_2*R2_2*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(1,5) = (R1_3*R2_2)/(mu*4.0)-(R1_2*R2_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(1,6) = (R1_1*R3_2)/(mu*4.0)-(R1_2*R3_1*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(1,7) = (R1_1*R3_1)/(mu*4.0)+(R1_3*R3_3)/(mu*4.0)+(R1_2*R3_2*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(1,8) = (R1_3*R3_2)/(mu*4.0)-(R1_2*R3_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(2,0) = (R1_1*R1_3)/(mu*4.0)-(R1_1*R1_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(2,1) = (R1_2*R1_3)/(mu*4.0)-(R1_2*R1_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(2,2) = (R1_1*R1_1)/(mu*4.0)+(R1_2*R1_2)/(mu*4.0)+((R1_3*R1_3)*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(2,3) = (R1_1*R2_3)/(mu*4.0)-(R1_3*R2_1*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(2,4) = (R1_2*R2_3)/(mu*4.0)-(R1_3*R2_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(2,5) = (R1_1*R2_1)/(mu*4.0)+(R1_2*R2_2)/(mu*4.0)+(R1_3*R2_3*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(2,6) = (R1_1*R3_3)/(mu*4.0)-(R1_3*R3_1*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(2,7) = (R1_2*R3_3)/(mu*4.0)-(R1_3*R3_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(2,8) = (R1_1*R3_1)/(mu*4.0)+(R1_2*R3_2)/(mu*4.0)+(R1_3*R3_3*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(3,0) = (R1_2*R2_2)/(mu*4.0)+(R1_3*R2_3)/(mu*4.0)+(R1_1*R2_1*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(3,1) = (R1_1*R2_2)/(mu*4.0)-(R1_2*R2_1*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(3,2) = (R1_1*R2_3)/(mu*4.0)-(R1_3*R2_1*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(3,3) = (R2_2*R2_2)/(mu*4.0)+(R2_3*R2_3)/(mu*4.0)+((R2_1*R2_1)*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(3,4) = (R2_1*R2_2)/(mu*4.0)-(R2_1*R2_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(3,5) = (R2_1*R2_3)/(mu*4.0)-(R2_1*R2_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(3,6) = (R2_2*R3_2)/(mu*4.0)+(R2_3*R3_3)/(mu*4.0)+(R2_1*R3_1*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(3,7) = (R2_2*R3_1)/(mu*4.0)-(R2_1*R3_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(3,8) = (R2_3*R3_1)/(mu*4.0)-(R2_1*R3_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(4,0) = (R1_2*R2_1)/(mu*4.0)-(R1_1*R2_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(4,1) = (R1_1*R2_1)/(mu*4.0)+(R1_3*R2_3)/(mu*4.0)+(R1_2*R2_2*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(4,2) = (R1_2*R2_3)/(mu*4.0)-(R1_3*R2_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(4,3) = (R2_1*R2_2)/(mu*4.0)-(R2_1*R2_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(4,4) = (R2_1*R2_1)/(mu*4.0)+(R2_3*R2_3)/(mu*4.0)+((R2_2*R2_2)*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(4,5) = (R2_2*R2_3)/(mu*4.0)-(R2_2*R2_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(4,6) = (R2_1*R3_2)/(mu*4.0)-(R2_2*R3_1*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(4,7) = (R2_1*R3_1)/(mu*4.0)+(R2_3*R3_3)/(mu*4.0)+(R2_2*R3_2*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(4,8) = (R2_3*R3_2)/(mu*4.0)-(R2_2*R3_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(5,0) = (R1_3*R2_1)/(mu*4.0)-(R1_1*R2_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(5,1) = (R1_3*R2_2)/(mu*4.0)-(R1_2*R2_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(5,2) = (R1_1*R2_1)/(mu*4.0)+(R1_2*R2_2)/(mu*4.0)+(R1_3*R2_3*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(5,3) = (R2_1*R2_3)/(mu*4.0)-(R2_1*R2_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(5,4) = (R2_2*R2_3)/(mu*4.0)-(R2_2*R2_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(5,5) = (R2_1*R2_1)/(mu*4.0)+(R2_2*R2_2)/(mu*4.0)+((R2_3*R2_3)*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(5,6) = (R2_1*R3_3)/(mu*4.0)-(R2_3*R3_1*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(5,7) = (R2_2*R3_3)/(mu*4.0)-(R2_3*R3_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(5,8) = (R2_1*R3_1)/(mu*4.0)+(R2_2*R3_2)/(mu*4.0)+(R2_3*R3_3*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(6,0) = (R1_2*R3_2)/(mu*4.0)+(R1_3*R3_3)/(mu*4.0)+(R1_1*R3_1*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(6,1) = (R1_1*R3_2)/(mu*4.0)-(R1_2*R3_1*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(6,2) = (R1_1*R3_3)/(mu*4.0)-(R1_3*R3_1*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(6,3) = (R2_2*R3_2)/(mu*4.0)+(R2_3*R3_3)/(mu*4.0)+(R2_1*R3_1*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(6,4) = (R2_1*R3_2)/(mu*4.0)-(R2_2*R3_1*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(6,5) = (R2_1*R3_3)/(mu*4.0)-(R2_3*R3_1*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(6,6) = (R3_2*R3_2)/(mu*4.0)+(R3_3*R3_3)/(mu*4.0)+((R3_1*R3_1)*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(6,7) = (R3_1*R3_2)/(mu*4.0)-(R3_1*R3_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(6,8) = (R3_1*R3_3)/(mu*4.0)-(R3_1*R3_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(7,0) = (R1_2*R3_1)/(mu*4.0)-(R1_1*R3_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(7,1) = (R1_1*R3_1)/(mu*4.0)+(R1_3*R3_3)/(mu*4.0)+(R1_2*R3_2*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(7,2) = (R1_2*R3_3)/(mu*4.0)-(R1_3*R3_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(7,3) = (R2_2*R3_1)/(mu*4.0)-(R2_1*R3_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(7,4) = (R2_1*R3_1)/(mu*4.0)+(R2_3*R3_3)/(mu*4.0)+(R2_2*R3_2*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(7,5) = (R2_2*R3_3)/(mu*4.0)-(R2_3*R3_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(7,6) = (R3_1*R3_2)/(mu*4.0)-(R3_1*R3_2*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(7,7) = (R3_1*R3_1)/(mu*4.0)+(R3_3*R3_3)/(mu*4.0)+((R3_2*R3_2)*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(7,8) = (R3_2*R3_3)/(mu*4.0)-(R3_2*R3_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(8,0) = (R1_3*R3_1)/(mu*4.0)-(R1_1*R3_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(8,1) = (R1_3*R3_2)/(mu*4.0)-(R1_2*R3_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(8,2) = (R1_1*R3_1)/(mu*4.0)+(R1_2*R3_2)/(mu*4.0)+(R1_3*R3_3*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(8,3) = (R2_3*R3_1)/(mu*4.0)-(R2_1*R3_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(8,4) = (R2_3*R3_2)/(mu*4.0)-(R2_2*R3_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(8,5) = (R2_1*R3_1)/(mu*4.0)+(R2_2*R3_2)/(mu*4.0)+(R2_3*R3_3*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  ret(8,6) = (R3_1*R3_3)/(mu*4.0)-(R3_1*R3_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(8,7) = (R3_2*R3_3)/(mu*4.0)-(R3_2*R3_3*la)/(la*mu*6.0+(mu*mu)*4.0);
  ret(8,8) = (R3_1*R3_1)/(mu*4.0)+(R3_2*R3_2)/(mu*4.0)+((R3_3*R3_3)*(la+mu))/(la*mu*3.0+(mu*mu)*2.0);
  return ret;
}
