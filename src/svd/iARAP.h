//
// Modification of code from https://github.com/LamWS/iARAP
//

#pragma once

#include <Eigen/Dense>

namespace iARAP {

  void ComputeEigenvector0(double a00, double a01, double a02, double a11,
      double a12, double a22, double eval0, Eigen::Vector3d &evec0);

  void ComputeOrthogonalComplement(Eigen::Vector3d const &W,
                                  Eigen::Vector3d &U, Eigen::Vector3d &V);

  void ComputeEigenvector1(double a00, double a01, double a02, double a11,
      double a12, double a22, Eigen::Vector3d const &evec0,
      double eval1, Eigen::Vector3d &evec1);

  // Compute the tr(F'R) term seen in the ARAP energy
  double trace_S(const Eigen::Matrix3d &F);

  // Compute rotation
  Eigen::Matrix3d rotation(const Eigen::Matrix3d &F);
  Eigen::Matrix3d rotation(const Eigen::Matrix3d &F, bool compute_gradients,
      Eigen::Matrix<double, 9, 9>& dRdF);
}