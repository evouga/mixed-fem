#include "catch2/catch.hpp"
#include "test_common.h"
#include "svd/dsvd.h"
using namespace Test;

TEST_CASE("Penalty Energy Gradient - dER/ds") {

  App app;
  std::shared_ptr<SimObject> obj = app.obj;

  VectorXd s(6*obj->T_.rows());
  for (int i = 0; i < obj->T_.rows(); ++i) {
    s.segment(6*i,6) = obj->S_[i];
  }

  VectorXd def_grad = obj->J_*(obj->P_.transpose()*obj->qt_+obj->b_);
  double kappa = 100;
  
  // Energy function for finite differences
  auto E = [&](const VectorXd& st)-> double {
    double h = app.config->h;
    double Ela = 0;

    for (int i = 0; i < obj->T_.rows(); ++i) {
      Matrix<double,9,6> W;
      Wmat(obj->R_[i],W);
      Vector9d diff = W*st.segment(6*i,6) - def_grad.segment(9*i,9);
      Ela += 0.5*kappa*diff.dot(diff) * obj->vols_[i];
    }
    return Ela;
  };

  // Compute gradient
  VectorXd grad(6*obj->T_.rows());
  for (int i = 0; i < obj->T_.rows(); ++i) {
    Matrix<double,9,6> W;
    Wmat(obj->R_[i],W);
    Vector9d diff = W*obj->S_[i] - def_grad.segment(9*i,9);
    Vector6d g = W.transpose() * diff;
    grad.segment(6*i,6) = kappa * obj->vols_[i] * g;
  }

  // Finite difference gradient
  VectorXd fgrad;
  finite_gradient(s, E, fgrad, SECOND); 
  CHECK(compare_gradient(grad, fgrad));
}

TEST_CASE("Penalty Energy Gradient - dER/dx") {

  App app;
  std::shared_ptr<SimObject> obj = app.obj;
  double kappa = 100;

  // Energy function for finite differences
  auto E = [&](const VectorXd& x)-> double {
    double Ela = 0;
    VectorXd def_grad = obj->J_*(obj->P_.transpose()*x+obj->b_);

    for (int i = 0; i < obj->T_.rows(); ++i) {
      Matrix3d F = Map<Matrix3d>(def_grad.segment(9*i,9).data());
      JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
      Matrix3d R = svd.matrixU() *  svd.matrixV().transpose();
      Matrix<double,9,6> W;
      Wmat(R,W);

      Vector9d diff = W*obj->S_[i] - def_grad.segment(9*i,9);
      Ela += 0.5*kappa*diff.dot(diff) * obj->vols_[i];
    }
    return Ela;
  };

  // Compute gradient
  VectorXd def_grad = obj->J_*(obj->P_.transpose()*obj->qt_+obj->b_);
  VectorXd diff(9*obj->T_.rows());
  for (int i = 0; i < obj->T_.rows(); ++i) {
    Matrix<double,9,6> W;
    Wmat(obj->R_[i],W);
    diff.segment(9*i,9) = (W*obj->S_[i] - def_grad.segment(9*i,9));
  }

  VectorXd grad = kappa * obj->P_ * (obj->Jw_.transpose() 
      * obj->WhatS_ - obj->Jw_.transpose()) * diff;

  // Finite difference gradient
  VectorXd fgrad;
  finite_gradient(obj->qt_, E, fgrad, SECOND);
  CHECK(compare_gradient(grad, fgrad));
}

TEST_CASE("Penalty Energy Hessian - d2EL/ds2") {
  App app;
  std::shared_ptr<SimObject> obj = app.obj;

  VectorXd s(6*obj->T_.rows());
  for (int i = 0; i < obj->T_.rows(); ++i) {
    s.segment(6*i,6) = obj->S_[i];
  }

  VectorXd def_grad = obj->J_*(obj->P_.transpose()*obj->qt_+obj->b_);
  double kappa = 100;
  
  // Energy function for finite differences
  auto E = [&](const VectorXd& st)-> double {
    double h = app.config->h;
    double Ela = 0;

    for (int i = 0; i < obj->T_.rows(); ++i) {
      Matrix<double,9,6> W;
      Wmat(obj->R_[i],W);
      Vector9d diff = W*st.segment(6*i,6) - def_grad.segment(9*i,9);
      Ela += 0.5*kappa*diff.dot(diff) * obj->vols_[i];
    }
    return Ela;
  };

  // Compute gradient
  MatrixXd grad(6*obj->T_.rows(),6*obj->T_.rows());
  grad.setZero();
  for (int i = 0; i < obj->T_.rows(); ++i) {
    Matrix<double,9,6> W;
    Wmat(obj->R_[i],W);
    grad.block(6*i,6*i,6,6) = kappa * obj->vols_[i] * W.transpose() * W;
  }

  // Finite difference gradient
  MatrixXd fgrad;
  finite_hessian(s, E, fgrad, SECOND); 
  CHECK(compare_hessian(grad, fgrad));
}



TEST_CASE("Penalty Energy Gradient - d2EL/dxds") {

  App app;
  std::shared_ptr<SimObject> obj = app.obj;

  VectorXd s(6*obj->T_.rows());
  for (int i = 0; i < obj->T_.rows(); ++i) {
    s.segment(6*i,6) = obj->S_[i];
  }

  double kappa = 1e2;

  // Compute jacobian
  obj->update_gradients();

  // Compute gradient
  VectorXd def_grad = obj->J_*(obj->P_.transpose()*obj->qt_+obj->b_);
  MatrixXd Wk(9*obj->T_.rows(), 6*obj->T_.rows());
  Wk.setZero();
  for (int i = 0; i < obj->T_.rows(); ++i) {
    Matrix<double,9,6> W;
    Wmat(obj->R_[i],W);
    Wk.block(9*i,6*i,9,6) = W;
  }

  MatrixXd grad = kappa * obj->P_ * (obj->Jw_.transpose() 
      * obj->WhatS_ - obj->Jw_.transpose()) * Wk;
  grad += kappa * obj->P_ * (obj->Jw_.transpose() * obj->Whate_.transpose());

  // Energy function for finite differences
  auto Ex = [&](const VectorXd& x)-> double {
    double Ela = 0;
    VectorXd def_grad = obj->J_*(obj->P_.transpose()*x+obj->b_);

    for (int i = 0; i < obj->T_.rows(); ++i) {
      Matrix3d F = Map<Matrix3d>(def_grad.segment(9*i,9).data());
      JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
      Matrix3d R = svd.matrixU() *  svd.matrixV().transpose();
      Matrix<double,9,6> W;
      Wmat(R,W);

      Vector9d diff = W*obj->S_[i] - def_grad.segment(9*i,9);
      Ela += 0.5*kappa*diff.dot(diff) * obj->vols_[i];
    }
    return Ela;
  };

  // Vecotr function for finite differences
  auto E = [&](const VectorXd& s)-> VectorXd {
    for (int i = 0; i < obj->T_.rows(); ++i) {
      obj->S_[i] = s.segment(6*i,6);
    }
    // Compute gradient
    VectorXd g;
    VectorXd xt = obj->qt_;
    finite_gradient(xt,Ex,g,FOURTH,1e-4);
    return g;
  };

  // Finite difference gradient
  MatrixXd fgrad;
  finite_jacobian(s, E, fgrad, FOURTH, 1e-4);

  // std::cout << "fgrad: \n" << fgrad << std::endl;
  // std::cout << "grad: \n" << grad << std::endl;
  // std::cout << "diff: \n" << (fgrad-grad).norm() << std::endl;
  CHECK(compare_jacobian(grad, fgrad,1e-4));
}
