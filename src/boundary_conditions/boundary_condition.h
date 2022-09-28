#pragma once

#include <EigenTypes.h>
#include "config.h"

namespace mfem {

  class BoundaryCondition {
  public:
    BoundaryCondition(Eigen::MatrixXd& V,
        const BoundaryConditionConfig& config);

  private:
    BoundaryConditionConfig config_;
    Eigen::Matrix23x<double> bbox;

  };

}