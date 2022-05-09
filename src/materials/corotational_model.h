#pragma once

#include "materials/material_model.h"

namespace mfem {

  // corotational material model
  class CorotationalModel : public MaterialModel {
  public:
    
    CorotationalModel(const std::shared_ptr<MaterialConfig>& config)
        : MaterialModel("Corotational", config) {}

    double energy(const Eigen::Vector6d& S) override; 

    Eigen::Vector6d gradient(const Eigen::Vector6d& S) override; 

    Eigen::Matrix6d hessian_inv(const Eigen::Matrix3d& R,
        const Eigen::Vector6d& S) override;
    Eigen::Matrix6d hessian_inv(const Eigen::Matrix3d& R,
        const Eigen::Vector6d& S, double kappa) override;

  };


}
