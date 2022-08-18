#include "point_point_frame.h"

using namespace mfem;
using namespace Eigen;

template<int DIM>
double PointPointFrame<DIM>::distance(const VectorXd& x) {
  using VecD = Vector<double,DIM>;
  const VecD& a = x.segment<DIM>(DIM*E_(0));
  const VecD& b = x.segment<DIM>(DIM*E_(1));
  return (a-b).norm();
}

template<>
VectorXd PointPointFrame<2>::gradient(const VectorXd& x) {
  Eigen::Vector4d q;
  q << x(2*E_(0)), x(2*E_(0) + 1),
       x(2*E_(1)), x(2*E_(1) + 1);
  Vector4d g;
  double t13;
  double t4;
  double t5;
  double t6;
  double t7;
  /* GRADIENT */
  /*     G = GRADIENT(IN1) */
  /*     This function was generated by the Symbolic Math Toolbox version 8.7.
   */
  /*     16-Aug-2022 20:26:13 */
  t4 = q[0] + -q[2];
  t5 = q[1] + -q[3];
  t6 = fabs(t4);
  t7 = fabs(t5);
  if (t4 < 0.0) {
    t4 = -1.0;
  } else if (t4 > 0.0) {
    t4 = 1.0;
  } else if (t4 == 0.0) {
    t4 = 0.0;
  }
  if (t5 < 0.0) {
    t5 = -1.0;
  } else if (t5 > 0.0) {
    t5 = 1.0;
  } else if (t5 == 0.0) {
    t5 = 0.0;
  }
  t13 = 1.0 / sqrt(t6 * t6 + t7 * t7);
  t6 = t6 * t4 * t13;
  t4 = t7 * t5 * t13;
  g[0] = t6;
  g[1] = t4;
  g[2] = -t6;
  g[3] = -t4;
  return g;
}

template<>
MatrixXd PointPointFrame<2>::hessian(const VectorXd& x) {
  Eigen::Vector4d q;
  q << x(2*E_(0)), x(2*E_(0) + 1),
       x(2*E_(1)), x(2*E_(1) + 1);
  double H[16];
  double t12;
  double t13;
  double t14;
  double t15;
  double t17;
  double t18;
  double t19;
  double t20;
  double t23;
  double t24;
  double t4;
  double t5;
  double t6;
  double t7;
  /* HESSIAN */
  /*     H = HESSIAN(IN1) */
  /*     This function was generated by the Symbolic Math Toolbox version 8.7.
   */
  /*     16-Aug-2022 20:26:13 */
  t4 = q[0] + -q[2];
  t5 = q[1] + -q[3];
  t6 = fabs(t4);
  t7 = fabs(t5);
  /* dirac(t4); */
  /* dirac(t5); */
  if (t4 < 0.0) {
    t4 = -1.0;
  } else if (t4 > 0.0) {
    t4 = 1.0;
  } else if (t4 == 0.0) {
    t4 = 0.0;
  }
  if (t5 < 0.0) {
    t5 = -1.0;
  } else if (t5 > 0.0) {
    t5 = 1.0;
  } else if (t5 == 0.0) {
    t5 = 0.0;
  }
  t12 = t6 * t6;
  t13 = t7 * t7;
  t14 = t4 * t4;
  t15 = t5 * t5;
  t17 = 1.0 / sqrt(t12 + t13);
  t18 = pow(t17, 3.0);
  t19 = t14 * t17;
  t20 = t15 * t17;
  t23 = t6 * 0.0 * t17 * 2.0;
  t24 = t7 * 0.0 * t17 * 2.0;
  t12 = t12 * t14 * t18;
  t17 = t13 * t15 * t18;
  t14 = t6 * t7 * t4 * t5 * t18;
  t13 = (-t19 + -t23) + t12;
  t15 = (-t20 + -t24) + t17;
  t12 = (t19 + t23) + -t12;
  t17 = (t20 + t24) + -t17;
  H[0] = t12;
  H[1] = -t14;
  H[2] = t13;
  H[3] = t14;
  H[4] = -t14;
  H[5] = t17;
  H[6] = t14;
  H[7] = t15;
  H[8] = t13;
  H[9] = t14;
  H[10] = t12;
  H[11] = -t14;
  H[12] = t14;
  H[13] = t15;
  H[14] = -t14;
  H[15] = t17;
  return Map<Matrix4d>(H);
}
template class mfem::PointPointFrame<2>;
// template class mfem::PointPointFrame<3>;
