#pragma once

#include "collision_frame.h"

namespace mfem {
  
  template<int DIM>
  class PointPointFrame : public CollisionFrame<DIM> {
  
  public:
    PointPointFrame(int a, int b) {
      if (b < a) {
        int tmp = a;
        a = b;
        b = tmp;
      }
      E_ = Eigen::Vector2i(a,b);
    }

    double distance(const Eigen::VectorXd& x) final;
    Eigen::VectorXd gradient(const Eigen::VectorXd& x) final;
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x) final;
    const Eigen::VectorXi& E() const final {
      return E_;
    }
    
  private:

    Eigen::VectorXi E_;

  };
}
