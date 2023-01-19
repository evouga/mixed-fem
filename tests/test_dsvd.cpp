#define CATCH_CONFIG_MAIN

#include "catch2/catch.hpp"
//#include "test_common.h"
#include "svd/dsvd.h"
#include "svd/newton_procrustes.h"
#include "finitediff.hpp"
#include "svd/iARAP.h"

using namespace Eigen;
using namespace fd;

TEST_CASE("dsvd - dS/dF procrustes") {

  //generate a random rotation
  Matrix3d F;
  F << 1.0, 0.1, 0.2,
       0.1, 2.0, 0.4,
       0.3, 0.4, 0.5;

         JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
  Matrix3d U = svd.matrixU();
  Matrix3d V = svd.matrixV();

  Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 9,9> dRdF; 
  Eigen::Matrix3d R = U*V.transpose();
  newton_procrustes(R, S, F, true, dRdF, 1e-8, 1000);
  std::cout << "newton_procrustes DRDF:\n" << dRdF << std::endl;
  S = R.transpose() * F;
  // Tensor3333d dU, dV;
  // Tensor333d dS;
  // dsvd(dU, dS, dV, F);

  // Matrix3d V = svd.matrixV();
  // Matrix3d S = svd.singularValues().asDiagonal();
  // std::array<Matrix3d, 9> dS_dF;

  // for (int r = 0; r < 3; ++r) {
  //   for (int c = 0; c < 3; ++c) {
  //     dS_dF[3*c + r] = dV[r][c]*S*V.transpose() + V*dS[r][c].asDiagonal()*V.transpose()
  //         + V*S*dV[r][c].transpose();
  //   }
  // }

  // Matrix<double, 9, 9> J;
  // for (int i = 0; i < 9; ++i) {
  //   J.col(i) = Vector9d(dS_dF[i].data());
  // }
  // Matrix9d J = sim::flatten_multiply<Matrix3d>(R.transpose()) *
  //       (Matrix9d::Identity() - sim::flatten_multiply_right<Matrix3d>(S)*dRdF);

  Matrix9d J = Matrix9d::Zero();
  J.block<3,3>(0,0) = R.transpose();
  J.block<3,3>(3,3) = R.transpose();
  J.block<3,3>(6,6) = R.transpose();
  for (int i = 0; i < 9; ++i) {
    Vector9d flat = dRdF.row(i).transpose();
    Matrix3d dRdFi = Matrix3d(flat.data());
    std::cout << "dRdFi: " << i << " \n" << dRdFi << std::endl;
    Matrix3d Ji = R.transpose() * (dRdFi*S);
    J.col(i) -= Vector9d(Ji.data());
  }
  std::cout << "R \n" << R << " \n S \n" << S << std::endl;

  Matrix<double, 6, 9> Js;
  Js.row(0) = J.row(0);
  Js.row(1) = J.row(4);
  Js.row(2) = J.row(8);
  Js.row(3) = J.row(1);
  Js.row(4) = J.row(2);
  Js.row(5) = J.row(5);


  // Vecotr function for finite differences
  auto E = [&](const VectorXd& vecF)-> VectorXd {

    F = Matrix3d(vecF.data());
    JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
    Matrix3d S = svd.matrixV() * svd.singularValues().asDiagonal()
        * svd.matrixV().transpose();
    Vector6d vecS; vecS << S(0,0), S(1,1), S(2,2), S(1,0), S(2,0), S(2,1);
    return vecS;
  };

  // Finite difference gradient
  MatrixXd fgrad;
  VectorXd vecF = Vector9d(F.data());
  finite_jacobian(vecF, E, fgrad, SECOND);

  std::cout << "PROCRUSTES" << std::endl;
  // std::cout << "J: " << J << std::endl;
  std::cout << "fgrad: \n" << fgrad << std::endl;
  std::cout << "grad: \n" << Js << std::endl;
  CHECK(compare_jacobian(Js, fgrad));
}

TEST_CASE("dsvd - dS/dF") {

  Matrix3d F;
  F << 1.0, 0.1, 0.2,
       0.1, 2.0, 0.4,
       0.3, 0.4, 0.5;

  JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
  Matrix3d U = svd.matrixU();
  Matrix3d V = svd.matrixV();
  Matrix3d R = U*V.transpose();
  Matrix3d S = V * svd.singularValues().asDiagonal() * V.transpose();
  std::array<Matrix3d, 9> dRdF;
  dsvd(F, U, svd.singularValues(), V, dRdF);

  Matrix<double, 9, 9> dRdFVec;
  for (int i = 0; i < 9; ++i) {
    dRdFVec.row(i) = Vector9d(dRdF[i].data()).transpose();
  }
  std::cout << "R \n" << R << " \n S \n" << S << std::endl;

  std::cout << "SVD DRDF:\n" << dRdFVec << std::endl;


  // Matrix<double, 9, 9> J;
  // for (int i = 0; i < 9; ++i) {
  //   J.row(i) = Vector9d(dRdF[i].data()).transpose();
  // }

  Matrix9d J = sim::flatten_multiply<Matrix3d>(R.transpose()) *
      (Matrix9d::Identity() - sim::flatten_multiply_right<Matrix3d>(S)*dRdFVec);


  Matrix9d J2 = Matrix9d::Zero();
  J2.block<3,3>(0,0) = R.transpose();
  J2.block<3,3>(3,3) = R.transpose();
  J2.block<3,3>(6,6) = R.transpose();
  for (int i = 0; i < 9; ++i) {
    Matrix3d dRdFi = dRdF[i];
    std::cout << "dRdFi: " << i << " \n" << dRdFi << std::endl;

    Matrix3d Ji = R.transpose() * (dRdFi*S);
    J2.col(i) -= Vector9d(Ji.data());
  }

  std::cout << " DIFF: " << (J - J2).norm() << std::endl;


  Matrix<double, 6, 9> Js;
  Js.row(0) = J.row(0);
  Js.row(1) = J.row(4);
  Js.row(2) = J.row(8);
  Js.row(3) = J.row(1);
  Js.row(4) = J.row(2);
  Js.row(5) = J.row(5);

  // Tensor3333d dU, dV;
  // Tensor333d dS;
  // dsvd(dU, dS, dV, F);

  // Matrix3d V = svd.matrixV();
  // Matrix3d S = svd.singularValues().asDiagonal();
  // std::array<Matrix3d, 9> dS_dF;

  // for (int r = 0; r < 3; ++r) {
  //   for (int c = 0; c < 3; ++c) {
  //     dS_dF[3*c + r] = dV[r][c]*S*V.transpose() + V*dS[r][c].asDiagonal()*V.transpose()
  //         + V*S*dV[r][c].transpose();
  //   }
  // }

  // Matrix<double, 9, 9> J;
  // for (int i = 0; i < 9; ++i) {
  //   J.col(i) = Vector9d(dS_dF[i].data());
  // }

  // Vecotr function for finite differences
  auto E = [&](const VectorXd& vecF)-> VectorXd {

    F = Matrix3d(vecF.data());
    JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
    Matrix3d S = svd.matrixV() * svd.singularValues().asDiagonal()
        * svd.matrixV().transpose();
    Vector6d vecS; vecS << S(0,0), S(1,1), S(2,2), S(1,0), S(2,0), S(2,1);
    return vecS;
  };

  // Finite difference gradient
  MatrixXd fgrad;
  VectorXd vecF = Vector9d(F.data());
  finite_jacobian(vecF, E, fgrad, SECOND);

  std::cout << "SVD" << std::endl;
  // std::cout << "J: " << J << std::endl;
  // std::cout << "J2: " << J2 << std::endl;
  std::cout << "fgrad: \n" << fgrad << std::endl;
  std::cout << "grad: \n" << Js << std::endl;
  CHECK(compare_jacobian(Js, fgrad));
  
}

TEST_CASE("dsvd - dR/dF") {

  Matrix3d F;
  F << 1.0, 0.1, 0.2,
       0.1, 2.0, 0.4,
       0.3, 0.4, 0.5;

  JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
  Matrix3d U = svd.matrixU();
  Matrix3d V = svd.matrixV();
  std::array<Matrix3d, 9> dRdF;
  dsvd(F, U, svd.singularValues(), V, dRdF);

  Matrix<double, 9, 9> J;
  for (int i = 0; i < 9; ++i) {
    J.row(i) = Vector9d(dRdF[i].data()).transpose();
  }

  // Vecotr function for finite differences
  auto E = [&](const VectorXd& vecF)-> VectorXd {

    F = Matrix3d(vecF.data());
    JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
    Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    Vector9d vecR(R.data());
    return vecR;
  };

  // Finite difference gradient
  MatrixXd fgrad;
  VectorXd vecF = Vector9d(F.data());
  finite_jacobian(vecF, E, fgrad, SECOND);

  // std::cout << "fgrad: \n" << fgrad << std::endl;
  // std::cout << "grad: \n" << J << std::endl;
  CHECK(compare_jacobian(J, fgrad));
}

TEST_CASE("dRdF") {

  //generate a random rotation
  Eigen::Matrix3d F = 2.0*Eigen::Matrix3d::Random();
  Eigen::Matrix3d S = Eigen::Matrix3d::Random();
  Eigen::Matrix<double, 9,9> dRdF_fd;
  Eigen::Matrix<double, 3,3>  tmpR0, tmpR1;
  Eigen::Matrix<double, 9,9> dRdF; 
  double alpha = 1e-5;

  Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
  
  Eigen::Matrix3d perturb;

  //Finite Difference approximation
  for(unsigned int ii=0; ii< 3; ++ii) {
    for(unsigned int jj=0; jj< 3; ++jj) {
      perturb.setZero();
      perturb(ii,jj) = alpha;
      tmpR0 = tmpR1 = R0;
      newton_procrustes(tmpR0, S, F+perturb);
      newton_procrustes(tmpR1, S, F-perturb);
      dRdF_fd.col(ii + 3*jj) = sim::flatten(((tmpR0 - tmpR1).array()/(2.*alpha)).matrix());
    }

  }
  
  newton_procrustes(R0, S, F, true, dRdF, 1e-8, 200);

  //error 
  double error = (dRdF_fd - dRdF).norm();

  std::cout<<"************* procrustes::dRdF error: "<<error<<" ************* \n";
  CHECK(error <= alpha);
}

// TEST_CASE("R_IARAP") {

//   Matrix3d F;
//   F.setRandom();
//   JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
//   Matrix3d Rgt = svd.matrixU() * svd.matrixV().transpose();
//   Matrix3d R = iARAP::rotation(F);
//   std::cout << "R0: \n " << R << std::endl;
//   std::cout << "Rgt: \n" << Rgt << std::endl;
//   std::cout << " asdf: " << (R - Rgt).squaredNorm()  << std::endl;
//   std::cout << " asdf: " << (R - Rgt) << std::endl;
//   std::cout << "S: \n" <<  svd.matrixV() 
//       * svd.singularValues().asDiagonal()
//       * svd.matrixV().transpose() << std::endl;
//   CHECK((R - Rgt).squaredNorm() < 1e-8);
//   // R = U
// }
// TEST_CASE("dRdF_IARAP") {

//   //generate a random rotation
//   Matrix3d F;
//   F << 1.0, 0.1, 0.2,
//        0.1, 2.0, 0.4,
//        0.3, 0.4, 0.5;
//       //  F.setIdentity();
//   Eigen::Matrix<double, 9,9> dRdF; 
//   double alpha = 1e-5;

//   iARAP::rotation(F, true, dRdF);

//   // function for finite differences
//   auto E = [&](const VectorXd& vecF)-> VectorXd {
//     Matrix3d Ftmp = Matrix3d(vecF.data());
//     JacobiSVD<Matrix3d> svd(Ftmp, ComputeFullU | ComputeFullV);
//     Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
//     Vector9d vecR(R.data());
//     return vecR;
//   };

//   // Finite difference gradient
//   MatrixXd fgrad;
//   VectorXd vecF = Vector9d(F.data());
//   finite_jacobian(vecF, E, fgrad, FOURTH);

//   std::cout << "dRdF_fd: \n" << fgrad << std::endl;
//   std::cout << "drDf: \n" << dRdF << std::endl;
//   //error 
//   double error = (fgrad - dRdF).norm();

//   std::cout<<"************* iARAP::dRdF error: "<<error<<" ************* \n";
//   CHECK(error <= alpha);
// }