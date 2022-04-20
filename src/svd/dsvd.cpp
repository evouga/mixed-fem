#include "svd/dsvd.h"

using namespace Eigen;

void dsvd(Tensor3333d &dU, Tensor333d  &dS,
    Tensor3333d &dV, Ref<const Matrix3d> Fin) {

  Matrix3d UVT, tmp, U,V;
  Matrix3d lambda;
  Matrix3d F;
  Vector3d S; 
  //get the SVD 
  F = Fin;
  JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
  U = svd.matrixU();
  V = svd.matrixV();
  S = svd.singularValues();
  
  //crappy hack for now
  double tol = 1e-5;

  if(std::fabs(S[0] - S[1]) < tol || std::fabs(S[1] - S[2]) < tol || std::fabs(S[0] - S[2]) < tol) {
    F += Matrix3d::Random()*tol;
    JacobiSVD<Matrix3d> svd2(F, ComputeFullU | ComputeFullV);
    U = svd2.matrixU();
    V = svd2.matrixV();
    S = svd2.singularValues();
  }

  double w01, w02, w12;
  double d01, d02, d12;
  
  d01 = S(1)*S(1)-S(0)*S(0);
  d02 = S(2)*S(2)-S(0)*S(0);
  d12 = S(2)*S(2)-S(1)*S(1);
  
  //corresponds to conservative solution --- if singularity is detected no angular velocity
  d01 = 1.0/(std::abs(d01) < tol ? std::numeric_limits<double>::infinity() : d01);
  d02 = 1.0/(std::abs(d02) < tol ? std::numeric_limits<double>::infinity() : d02);
  d12 = 1.0/(std::abs(d12) < tol ? std::numeric_limits<double>::infinity() : d12);
  
  for(unsigned int r=0; r<3; ++r) {
    for(unsigned int s =0; s <3; ++s) {
        
      UVT = U.row(r).transpose()*V.row(s);
      
      //Compute dS
      dS[r][s] = UVT.diagonal();
      
      UVT -= dS[r][s].asDiagonal();
      
      tmp  = S.asDiagonal()*UVT + UVT.transpose()*S.asDiagonal();
      w01 = tmp(0,1)*d01;
      w02 = tmp(0,2)*d02;
      w12 = tmp(1,2)*d12;
      tmp << 0, w01, w02,
              -w01, 0, w12,
              -w02, -w12, 0;
      
      dV[r][s] = V*tmp;
      
      tmp = UVT*S.asDiagonal() + S.asDiagonal()*UVT.transpose();
      w01 = tmp(0,1)*d01;
      w02 = tmp(0,2)*d02;
      w12 = tmp(1,2)*d12;
      tmp << 0, w01, w02,
      -w01, 0, w12,
      -w02, -w12, 0;
      
      dU[r][s] = U*tmp;
                    
    }
  }
}

void dsvd(Ref<const Matrix3d> Fin, Ref<const Matrix3d> Uin,
    Ref<const Vector3d> Sin, Ref<const Matrix3d> Vtin,
    std::array<Matrix3d, 9>& dR_dF) {

  Matrix3d UVT, tmp, dVt, dU;
  Matrix3d U = Uin;
  Vector3d S = Sin;
  Matrix3d Vt = Vtin;//.transpose();
  Matrix3d F = Fin;
  Vector3d dS;

    // JacobiSVD<Matrix3d> svd1(F, ComputeFullU | ComputeFullV);
  // std::cout << "F: " << F << " \nU: " << U << "\n Usvd: " << svd1.matrixU() << "\n V: "
  //    << Vt << "\n Vsvd: " << svd1.matrixV() << "\n S: " << S 
  //    << " \nSsvd: " << svd1.singularValues() << "\nUSV: " << svd1.matrixU() * svd1.singularValues().asDiagonal() * svd1.matrixV().transpose() << "\nUSVt: " << U*S.asDiagonal()*Vt.transpose() << std::endl;
  
  //crappy hack for now
  double tol = 1e-5;
    // U = svd1.matrixU();
    // Vt = svd1.matrixV();
    // S = svd1.singularValues();
  if(std::fabs(S[0] - S[1]) < tol || std::fabs(S[1] - S[2]) < tol || std::fabs(S[0] - S[2]) < tol) {
    F += Matrix3d::Random()*tol;
    JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
    U = svd.matrixU();
    Vt = svd.matrixV();
    S = svd.singularValues();
    //std::cerr << "does this actually happen" << std::endl;
  }

  double w01, w02, w12;
  double d01, d02, d12;
  
  d01 = S(1)*S(1)-S(0)*S(0);
  d02 = S(2)*S(2)-S(0)*S(0);
  d12 = S(2)*S(2)-S(1)*S(1);
  
  //corresponds to conservative solution --- if singularity is detected no angular velocity
  d01 = 1.0/(std::abs(d01) < tol ? std::numeric_limits<double>::infinity() : d01);
  d02 = 1.0/(std::abs(d02) < tol ? std::numeric_limits<double>::infinity() : d02);
  d12 = 1.0/(std::abs(d12) < tol ? std::numeric_limits<double>::infinity() : d12);
  

  for(unsigned int r=0; r<3; ++r) {
    for(unsigned int s =0; s <3; ++s) {
        
      UVT = U.row(r).transpose()*Vt.row(s);
      
      //Compute dS
      dS = UVT.diagonal();
      
      UVT -= dS.asDiagonal();
      
      tmp  = S.asDiagonal()*UVT + UVT.transpose()*S.asDiagonal();
      w01 = tmp(0,1)*d01;
      w02 = tmp(0,2)*d02;
      w12 = tmp(1,2)*d12;
      tmp << 0, w01, w02,
              -w01, 0, w12,
              -w02, -w12, 0;
      
      dVt = Vt*tmp;
      
      tmp = UVT*S.asDiagonal() + S.asDiagonal()*UVT.transpose();
      w01 = tmp(0,1)*d01;
      w02 = tmp(0,2)*d02;
      w12 = tmp(1,2)*d12;
      tmp << 0, w01, w02,
      -w01, 0, w12,
      -w02, -w12, 0;
      
      dU = U*tmp;

      dR_dF[3*s + r] = dU*Vt + U*dVt;
                    
    }
  }

}