#pragma once

#include "EigenTypes.h"
#include <memory>

namespace mfem {

  enum DistanceQuery {
    POINT_POINT,
    POINT_EDGE,
    POINT_TRIANGLE,
    EDGE_EDGE
  };


  template<int DIM>
  class CollisionFrame {

  public:

    template<typename Derived, DistanceQuery Q>
    static std::unique_ptr<CollisionFrame> make_collision_frame(const Eigen::VectorXd& x,
      const Eigen::MatrixBase<Derived>& E); 

    CollisionFrame() = default;
    virtual ~CollisionFrame() = default;

    virtual double distance(const Eigen::VectorXd& x) = 0;
    virtual Eigen::VectorXd gradient(const Eigen::VectorXd& x) = 0;
    virtual Eigen::MatrixXd hessian(const Eigen::VectorXd& x) = 0;
    virtual Eigen::VectorXi& E() = 0;
  };
}