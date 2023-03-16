#pragma once

#include "EigenTypes.h"
#include "ipc/ipc.hpp"

namespace ipc {
  // "Swept-Volume" Broadphase Additive CCD
  template <int DIM>
  double additive_ccd_narrowonly(const Eigen::VectorXd& x,
      const Eigen::VectorXd& p,
      const ipc::CollisionMesh& mesh,
      const ipc::Candidates& candidates);    

  template <int DIM>
  double additive_ccd(const Eigen::VectorXd& x, const Eigen::VectorXd& p,
      const ipc::CollisionMesh& mesh, ipc::Candidates& candidates,
      double dhat);
}
