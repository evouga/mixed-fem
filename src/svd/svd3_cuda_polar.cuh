#pragma once

#include "svd3_cuda.cuh"
#include <Eigen/Core>

__device__ 
void svd_polar(const Eigen::Matrix3d& Ad, Eigen::Matrix3f& R) {
	Eigen::Matrix3f A = Ad.cast<float>();
	Eigen::Matrix3f U, V;
	Eigen::Vector3f sigma;

	svd(A(0,0), A(0,1), A(0,2), A(1,0), A(1,1), A(1,2), A(2,0), A(2,1), A(2,2),
			U(0,0), U(0,1), U(0,2), U(1,0), U(1,1), U(1,2), U(2,0), U(2,1), U(2,2),
			sigma(0), sigma(1), sigma(2),
			V(0,0), V(0,1), V(0,2), V(1,0), V(1,1), V(1,2), V(2,0), V(2,1), V(2,2));

	R = U * V.transpose();
	// R = Rf.cast<double>();
	// Check if correct
	// Eigen::Matrix3f A2 = U * sigma.asDiagonal() * V.transpose();
	// Eigen::Matrix3f diff = A2 - A;
	// float error = diff.norm();
	// printf("Error: %f \n",error);
	// // print A2
	// printf("A = %f %f %f %f %f %f %f %f %f\n", A2(0,0), A2(0,1), A2(0,2), A2(1,0), A2(1,1), A2(1,2), A2(2,0), A2(2,1), A2(2,2));

}

__device__ 
void svd_deriv_S(const Eigen::Matrix3d& Ad, Eigen::Matrix3f& R,
		Eigen::Matrix<float, 6, 9>& dSdF) {
	Eigen::Matrix3f A = Ad.cast<float>();
	Eigen::Matrix3f U, V;
	Eigen::Vector3f S;

	svd(A(0,0), A(0,1), A(0,2), A(1,0), A(1,1), A(1,2), A(2,0), A(2,1), A(2,2),
			U(0,0), U(0,1), U(0,2), U(1,0), U(1,1), U(1,2), U(2,0), U(2,1), U(2,2),
			S(0), S(1), S(2),
			V(0,0), V(0,1), V(0,2), V(1,0), V(1,1), V(1,2), V(2,0), V(2,1), V(2,2));

	R = U * V.transpose();

	Eigen::Matrix3f UVT, tmp, dV, dU;
  Eigen::Vector3f dS;

  //crappy hack for now
  float tol = 1e-5;
  float w01, w02, w12;
  float d01, d02, d12;
  
  d01 = S(1)*S(1)-S(0)*S(0);
  d02 = S(2)*S(2)-S(0)*S(0);
  d12 = S(2)*S(2)-S(1)*S(1);
  
  //corresponds to conservative solution --- if singularity is detected no angular velocity
	if (abs(d01) < tol) {
		d01 = 0;
	} else {
		d01 = 1.0 / d01;
	} 
	if (abs(d02) < tol) {
		d02 = 0;
	} else {
		d02 = 1.0 / d02;
	}
	if (abs(d12) < tol) {
		d12 = 0;
	} else {
		d12 = 1.0 / d12;
	}

	Eigen::Matrix3f dR_dF[9];

  for(unsigned int r=0; r<3; ++r) {
    for(unsigned int s =0; s <3; ++s) {
        
      UVT = U.row(r).transpose()*V.row(s);
      
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
      
      dV = V*tmp;
      
      tmp = UVT*S.asDiagonal() + S.asDiagonal()*UVT.transpose();
      w01 = tmp(0,1)*d01;
      w02 = tmp(0,2)*d02;
      w12 = tmp(1,2)*d12;
      tmp << 0, w01, w02,
      -w01, 0, w12,
      -w02, -w12, 0;
      
      dU = U*tmp;

      dR_dF[3*s + r] = dU*V.transpose() + U*dV.transpose();
                   
    }
  }

  Eigen::Matrix<float,9,9> J = Eigen::Matrix<float,9,9>::Zero();
  J.block<3,3>(0,0) = R.transpose();
  J.block<3,3>(3,3) = R.transpose();
  J.block<3,3>(6,6) = R.transpose();
	Eigen::Matrix3f Sym = R.transpose() * A;
  for (int i = 0; i < 9; ++i) {
    Eigen::Matrix3f Ji = R.transpose() * (dR_dF[i]*Sym);
    J.col(i) -= Eigen::Matrix<float,9,1>(Ji.data());
  }

  dSdF.row(0) = J.row(0);
  dSdF.row(1) = J.row(4);
  dSdF.row(2) = J.row(8);
  dSdF.row(3) = J.row(1);
  dSdF.row(4) = J.row(2);
  dSdF.row(5) = J.row(5);

	// print row of dsfd
	// printf("dSdF1 = %f %f %f %f %f %f %f %f %f \n", dSdF(0,0), dSdF(0,1), dSdF(0,2), dSdF(0,3), dSdF(0,4), dSdF(0,5), dSdF(0,6), dSdF(0,7), dSdF(0,8));
	//next row
	// printf("dSdF2 = %f %f %f %f %f %f %f %f %f \n", dSdF(1,0), dSdF(1,1), dSdF(1,2), dSdF(1,3), dSdF(1,4), dSdF(1,5), dSdF(1,6), dSdF(1,7), dSdF(1,8));
}