#pragma once

#include "EigenTypes.h"

namespace mfem {

  template <int DIM>
  double max_possible_step(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
      const Eigen::MatrixXi& F);

  // Additive Continuous Collision Detection (ACCD) from Codimensional-IPC
  // paper.
  template <int DIM>
  double additive_ccd(const Eigen::VectorXd& x, const Eigen::VectorXd& p,
      const Eigen::MatrixXi& F);      
}
