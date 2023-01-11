#pragma once

#include "factory.h"
#include "mesh/mesh.h"
#include "boundary_conditions/boundary_condition.h"
#include "boundary_conditions/external_force.h"

namespace mfem {
  
  class BoundaryConditionFactory : public Factory<BCScriptType,
      BoundaryCondition, Eigen::MatrixXd, BoundaryConditionConfig> {
  public:
    BoundaryConditionFactory();
  };

  class ExternalForceFactory : public Factory<ExternalForceType,
      ExternalForce, Eigen::MatrixXd, ExternalForceConfig> {
  public:
    ExternalForceFactory();
  };
}
