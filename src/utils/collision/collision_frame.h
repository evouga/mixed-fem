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
    virtual const Eigen::VectorXi& E() const = 0;

  };

  template<int DIM> 
  struct FrameLess
  {
    template<typename T>
    bool operator()(const T& a, const T& b) const {
      const Eigen::VectorXi& E1 = a->E();
      const Eigen::VectorXi& E2 = b->E();

      // If different size sort according to whichever is smaller 
      if (E1.size() != E2.size()) {
        return E1.size() < E2.size();
      } else {
        // Otherwise sort alphabetically 
        for (int i = 0; i < E1.size(); ++i) {
          if (E1(i) < E2(i))
            return true;
          else if (E1(i) > E2(i))
            return false;
        }
        return false;
      }
    }
  };
}
