#pragma once

#include "EigenTypes.h"
#include "ipc/ipc.hpp"

namespace ipc {
  template <int DIM>
  double additive_ccd(const Eigen::VectorXd& x,
      const Eigen::VectorXd& p,
      const ipc::CollisionMesh& mesh,
      ipc::Candidates& candidates,
      double dhat);    

  template <int DIM>
  double additive_ccd(
      const Eigen::MatrixXd& V1,
      const Eigen::MatrixXd& V2,
      const ipc::CollisionMesh& mesh,
      ipc::Candidates& candidates,
      double dhat);    
}
