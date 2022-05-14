#pragma once

#include <EigenTypes.h>

namespace mfem {
  void svd(const Eigen::Ref<Eigen::Matrix3d> F, Eigen::Vector3d& sigma,
    Eigen::Matrix3d& U, Eigen::Matrix3d& V, bool identify_flips = true) {
    
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F,
        Eigen::ComputeFullU | Eigen::ComputeFullV);
    sigma = svd.singularValues();
    U = svd.matrixU();
    V = svd.matrixV();

    if (identify_flips) {
      Eigen::Matrix3d J = Eigen::Matrix3d::Identity();
      J(2, 2) = -1.0;
      if (U.determinant() < 0.) {
          U = U * J;
          sigma[2] = -sigma[2];
      }
      if (V.determinant() < 0.0) {
          Eigen::Matrix3d Vt = V.transpose();
          Vt = J * Vt;
          V = Vt.transpose();
          sigma[2] = -sigma[2];
      }
    }        
  }
}