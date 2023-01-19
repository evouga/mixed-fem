//
// Modification of code from https://github.com/LamWS/iARAP
//

#pragma once

#include <Eigen/Dense>

namespace mfem {

  template<typename Scalar>  __device__
  void ComputeEigenvector0(Scalar a00, Scalar a01, Scalar a02, Scalar a11,
      Scalar a12, Scalar a22, Scalar eval0,
      Eigen::Matrix<Scalar, 3, 1> &evec0);

  template<typename Scalar> __device__
  void ComputeOrthogonalComplement(Eigen::Matrix<Scalar, 3, 1> const &W,
    Eigen::Matrix<Scalar, 3, 1> &U,
    Eigen::Matrix<Scalar, 3, 1> &V);

  template<typename Scalar> __device__
  void ComputeEigenvector1(Scalar a00, Scalar a01, Scalar a02, Scalar a11,
      Scalar a12, Scalar a22, Eigen::Matrix<Scalar, 3, 1> const &evec0,
      Scalar eval1, Eigen::Matrix<Scalar, 3, 1> &evec1);

  // Compute rotation
  template<typename Scalar> __device__
  Eigen::Matrix<Scalar, 3, 3> rotation(const Eigen::Matrix<Scalar, 3, 3> &F);

  template<typename Scalar> __device__
  Eigen::Matrix<Scalar, 3, 3> rotation(const Eigen::Matrix<Scalar, 3, 3> &F,
      bool compute_gradients, Eigen::Matrix<Scalar, 9, 9>& dRdF);

}