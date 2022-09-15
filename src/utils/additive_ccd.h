#pragma once

#include "EigenTypes.h"
#include "ipc/ipc.hpp"

namespace ipc {
  template <int DIM>
  double additive_ccd(const Eigen::VectorXd& x, const Eigen::VectorXd& p,
      const ipc::CollisionMesh& mesh, double dhat);    

}
