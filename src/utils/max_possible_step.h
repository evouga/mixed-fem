#pragma once

#include "EigenTypes.h"

namespace mfem {

  template <int DIM>
  double max_possible_step(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
      const Eigen::MatrixXi& F);
}
