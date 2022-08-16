#pragma once

#include "collision_frame.h"

namespace mfem {
  
  template<int DIM>
  class PointEdgeFrame : CollisionFrame<DIM> {
  
  public:
    PointEdgeFrame(const Eigen::Vector3i& E) : E_(E) {}

    double distance(const Eigen::VectorXd& x) final;
    Eigen::VectorXd gradient(const Eigen::VectorXd& x) final;
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x) final;
    Eigen::VectorXi& E() final {
      return E_;
    }
    
  private:

    Eigen::VectorXi E_;

  };
}