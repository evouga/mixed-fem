#include "catch2/catch.hpp"
#include "test_common.h"
#include "svd/dsvd.h"
using namespace Test;

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

TEST_CASE("dsvd - dWs/dF") {

  Matrix3d F;
  F << 1.0, 0.1, 0.2,
       0.1, 2.0, 0.4,
       0.3, 0.4, 0.5;

  Vector6d s;
  s << 1.11, 1.2, 1.3, 0.2, 0.3, 0.4;

  JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
  Matrix3d U = svd.matrixU();
  Matrix3d V = svd.matrixV();
  std::array<Matrix3d, 9> dRdF;
  dsvd(F, U, svd.singularValues(), V, dRdF);

  Matrix<double, 9, 9> J;
  Matrix<double, 9, 6> What;

  for (int i = 0; i < 9; ++i) {
    Wmat(dRdF[i] , What);
    J.col(i) = (What * s);
  }

  // function for finite differences
  auto E = [&](const VectorXd& vecF)-> VectorXd {

    F = Matrix3d(vecF.data());
    JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
    Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    Matrix<double, 9, 6> W;
    Wmat(R, W);
    return W*s;
  };

  // Finite difference gradient
  MatrixXd fgrad;
  VectorXd vecF = Vector9d(F.data());
  finite_jacobian(vecF, E, fgrad, SECOND);
  CHECK(compare_jacobian(J, fgrad));
}


TEST_CASE("dsvd - dWs/dq") {
  App app;
  std::shared_ptr<SimObject> obj = app.obj;
  int n = obj->J_.cols();
  MatrixXd Jk = obj->J_.block(0,0,9,n);
  obj->qt_ *= 10;

  Vector9d vecF = Jk * (obj->P_.transpose() * obj->qt_ + obj->b_);
  Matrix3d F = Matrix3d(vecF.data());
  
  Vector6d s;
  s << 1.11, 1.2, 1.3, 0.2, 0.3, 0.4;
  s *= 100;

  JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
  Matrix3d U = svd.matrixU();
  Matrix3d V = svd.matrixV();
  std::array<Matrix3d, 9> dRdF;
  dsvd(F, U, svd.singularValues(), V, dRdF);

  Matrix<double, 9, 9> Whats;
  Matrix<double, 9, 6> What;
  for (int i = 0; i < 9; ++i) {
    Wmat(dRdF[i] , What);
    Whats.row(i) = (What * s).transpose();
  }

  MatrixXd J = obj->P_ * Jk.transpose() * Whats;

  // function for finite differences
  auto E = [&](const VectorXd& x)-> VectorXd {
    Vector9d vecF = Jk * (obj->P_.transpose() * x + obj->b_);
    Matrix3d F = Matrix3d(vecF.data());
    JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
    Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    Matrix<double, 9, 6> W;
    Wmat(R, W);
    return W*s;
  };

  // Finite difference gradient
  MatrixXd fgrad;
  VectorXd qt = obj->qt_;
  finite_jacobian(qt, E, fgrad, SIXTH, 1e-6);
  // std::cout << "frad: \n" << fgrad << std::endl;
  // std::cout << "grad: \n" << J.transpose() << std::endl;
  CHECK(compare_jacobian(J.transpose(), fgrad));
}