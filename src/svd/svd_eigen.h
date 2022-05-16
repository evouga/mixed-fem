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

    Eigen::Vector3x<Scalar> stemp;
    stemp[0] = 1;
    stemp[1] = 1;
    stemp[2]  = (svd.matrixU()*svd.matrixV().transpose()).determinant();
   
    sigma = (svd.singularValues().array() *stemp.array()).matrix(); 
    U = svd.matrixU() *stemp.asDiagonal();
    V = svd.matrixV();
            
  }
}