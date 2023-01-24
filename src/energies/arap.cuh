#pragma once

#include "EigenTypes.h"

namespace mfem {

  template <typename Scalar> __device__ __forceinline__
  Scalar local_energy(const Eigen::Matrix<Scalar, 6, 1>& S, Scalar mu, Scalar la) {
    return (mu*(pow(S(0)-1.0,2.0)+pow(S(1)-1.0,2.0)+pow(S(2)-1.0,2.0)
        +(S(3)*S(3))*2.0+(S(4)*S(4))*2.0+(S(5)*S(5))*2.0))/2.0;
  }
  
  template <typename Scalar> __device__ __forceinline__
  Eigen::Matrix<Scalar, 6, 1> local_gradient(
      const Eigen::Matrix<Scalar, 6, 1>& S, Scalar mu, Scalar la) {
    Eigen::Matrix<Scalar, 6, 1> g;
    g(0) = (mu*(S(0)*2.0-2.0))/2.0;
    g(1) = (mu*(S(1)*2.0-2.0))/2.0;
    g(2) = (mu*(S(2)*2.0-2.0))/2.0;
    g(3) = S(3)*mu*2.0;
    g(4) = S(4)*mu*2.0;
    g(5) = S(5)*mu*2.0;
    return g;    
  }

  template <typename Scalar> __device__ __forceinline__
  Eigen::Matrix<Scalar, 6, 6> local_hessian(
      const Eigen::Matrix<Scalar, 6, 1>& S, Scalar mu, Scalar la) {
    Eigen::Matrix<Scalar, 6, 1> tmp; tmp << 1,1,1,2,2,2;
    Eigen::Matrix<Scalar, 6, 6> tmp2; tmp2.setZero();
    tmp2.diagonal() = tmp;
    return tmp2 * mu;
  }
}
