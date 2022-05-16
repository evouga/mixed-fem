#pragma once

#include <EigenTypes.h>

namespace mfem {

  template <typename Scalar>
  void svd(const Eigen::Ref<Eigen::Matrix3d> F,
      Eigen::Vector3x<Scalar>& sigma, Eigen::Matrix3x<Scalar>& U,
      Eigen::Matrix3x<Scalar>& V, bool identify_flips = true) {
    
    Eigen::JacobiSVD<Eigen::Matrix3x<Scalar>> svd(F,
        Eigen::ComputeFullU | Eigen::ComputeFullV);
    sigma = svd.singularValues();
    U = svd.matrixU();
    V = svd.matrixV();

    if (identify_flips) {
      Eigen::Matrix3x<Scalar> J = Eigen::Matrix3x<Scalar>::Identity();
      J(2, 2) = -1.0;
      if (U.determinant() < 0.) {
          U = U * J;
          sigma[2] = -sigma[2];
      }
      if (V.determinant() < 0.0) {
          Eigen::Matrix3x<Scalar> Vt = V.transpose();
          Vt = J * Vt;
          V = Vt.transpose();
          sigma[2] = -sigma[2];
      }
    }        
            
  }
}