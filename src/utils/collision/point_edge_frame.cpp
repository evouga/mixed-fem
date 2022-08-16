#include "point_edge_frame.h"

using namespace mfem;
using namespace Eigen;

template<int DIM>
double PointEdgeFrame<DIM>::distance(const VectorXd& x) {
  using VecD = Vector<double,DIM>;
  const VecD& a = x.segment<DIM>(DIM*E_(0));
  const VecD& b = x.segment<DIM>(DIM*E_(1));
  const VecD& p = x.segment<DIM>(DIM*E_(2));
  VecD v = b - a;
  VecD w = p - a;
  double c = w.dot(v) / v.dot(v);
  return (p - (a + c*v)).norm();
}

template<>
VectorXd PointEdgeFrame<2>::gradient(const VectorXd& x) {
  Eigen::Vector6d q;
  double q1 = x(2*E_(0));
  double q2 = x(2*E_(0) + 1);
  double q3 = x(2*E_(1));
  double q4 = x(2*E_(1) + 1);
  double q5 = x(2*E_(2));
  double q6 = x(2*E_(2) + 1);

  Vector6d g;
  g(0) = (((-q1+q5+((q1-q3)*((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6)))/(pow(q1-q3,2.0)+pow(q2-q4,2.0)))*(-((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6))/(pow(q1-q3,2.0)+pow(q2-q4,2.0))+((q1-q3)*(q1*-2.0+q3+q5))/(pow(q1-q3,2.0)+pow(q2-q4,2.0))+(q1*2.0-q3*2.0)*1.0/pow(pow(q1-q3,2.0)+pow(q2-q4,2.0),2.0)*(q1-q3)*((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6))+1.0)*2.0+(((q2-q4)*(q1*-2.0+q3+q5))/(pow(q1-q3,2.0)+pow(q2-q4,2.0))+(q1*2.0-q3*2.0)*1.0/pow(pow(q1-q3,2.0)+pow(q2-q4,2.0),2.0)*(q2-q4)*((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6)))*(-q2+q6+((q2-q4)*((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6)))/(pow(q1-q3,2.0)+pow(q2-q4,2.0)))*2.0)*1.0/sqrt(1.0/(q1*q3*-2.0-q2*q4*2.0+q1*q1+q2*q2+q3*q3+q4*q4))*(-1.0/2.0))/fabs(q1*q4-q2*q3-q1*q6+q2*q5+q3*q6-q4*q5);
  g(1) = (((-q2+q6+((q2-q4)*((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6)))/(pow(q1-q3,2.0)+pow(q2-q4,2.0)))*(-((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6))/(pow(q1-q3,2.0)+pow(q2-q4,2.0))+((q2-q4)*(q2*-2.0+q4+q6))/(pow(q1-q3,2.0)+pow(q2-q4,2.0))+(q2*2.0-q4*2.0)*1.0/pow(pow(q1-q3,2.0)+pow(q2-q4,2.0),2.0)*(q2-q4)*((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6))+1.0)*2.0+(((q1-q3)*(q2*-2.0+q4+q6))/(pow(q1-q3,2.0)+pow(q2-q4,2.0))+(q2*2.0-q4*2.0)*1.0/pow(pow(q1-q3,2.0)+pow(q2-q4,2.0),2.0)*(q1-q3)*((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6)))*(-q1+q5+((q1-q3)*((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6)))/(pow(q1-q3,2.0)+pow(q2-q4,2.0)))*2.0)*1.0/sqrt(1.0/(q1*q3*-2.0-q2*q4*2.0+q1*q1+q2*q2+q3*q3+q4*q4))*(-1.0/2.0))/fabs(q1*q4-q2*q3-q1*q6+q2*q5+q3*q6-q4*q5);
  g(2) = (((-q1+q5+((q1-q3)*((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6)))/(pow(q1-q3,2.0)+pow(q2-q4,2.0)))*(((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6))/(pow(q1-q3,2.0)+pow(q2-q4,2.0))+((q1-q3)*(q1-q5))/(pow(q1-q3,2.0)+pow(q2-q4,2.0))-(q1*2.0-q3*2.0)*1.0/pow(pow(q1-q3,2.0)+pow(q2-q4,2.0),2.0)*(q1-q3)*((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6)))*2.0+(((q1-q5)*(q2-q4))/(pow(q1-q3,2.0)+pow(q2-q4,2.0))-(q1*2.0-q3*2.0)*1.0/pow(pow(q1-q3,2.0)+pow(q2-q4,2.0),2.0)*(q2-q4)*((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6)))*(-q2+q6+((q2-q4)*((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6)))/(pow(q1-q3,2.0)+pow(q2-q4,2.0)))*2.0)*1.0/sqrt(1.0/(q1*q3*-2.0-q2*q4*2.0+q1*q1+q2*q2+q3*q3+q4*q4))*(-1.0/2.0))/fabs(q1*q4-q2*q3-q1*q6+q2*q5+q3*q6-q4*q5);
  g(3) = (((-q2+q6+((q2-q4)*((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6)))/(pow(q1-q3,2.0)+pow(q2-q4,2.0)))*(((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6))/(pow(q1-q3,2.0)+pow(q2-q4,2.0))+((q2-q4)*(q2-q6))/(pow(q1-q3,2.0)+pow(q2-q4,2.0))-(q2*2.0-q4*2.0)*1.0/pow(pow(q1-q3,2.0)+pow(q2-q4,2.0),2.0)*(q2-q4)*((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6)))*2.0+(((q1-q3)*(q2-q6))/(pow(q1-q3,2.0)+pow(q2-q4,2.0))-(q2*2.0-q4*2.0)*1.0/pow(pow(q1-q3,2.0)+pow(q2-q4,2.0),2.0)*(q1-q3)*((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6)))*(-q1+q5+((q1-q3)*((q1-q3)*(q1-q5)+(q2-q4)*(q2-q6)))/(pow(q1-q3,2.0)+pow(q2-q4,2.0)))*2.0)*1.0/sqrt(1.0/(q1*q3*-2.0-q2*q4*2.0+q1*q1+q2*q2+q3*q3+q4*q4))*(-1.0/2.0))/fabs(q1*q4-q2*q3-q1*q6+q2*q5+q3*q6-q4*q5);
  g(4) = (sqrt(1.0/(q1*q3*-2.0-q2*q4*2.0+q1*q1+q2*q2+q3*q3+q4*q4))*(q2-q4)*(q1*q4-q2*q3-q1*q6+q2*q5+q3*q6-q4*q5))/fabs(q1*q4-q2*q3-q1*q6+q2*q5+q3*q6-q4*q5);
  g(5) = -(sqrt(1.0/(q1*q3*-2.0-q2*q4*2.0+q1*q1+q2*q2+q3*q3+q4*q4))*(q1-q3)*(q1*q4-q2*q3-q1*q6+q2*q5+q3*q6-q4*q5))/fabs(q1*q4-q2*q3-q1*q6+q2*q5+q3*q6-q4*q5);
  return g;
}

template class mfem::PointEdgeFrame<2>;
// template class mfem::PointEdgeFrame<3>;

    // Eigen::VectorXd gradient(const Eigen::VectorXd& x) final;
    // Eigen::MatrixXd hessian(const Eigen::VectorXd& x) final;