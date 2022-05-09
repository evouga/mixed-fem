#pragma once

#include "materials/material_model.h"

namespace mfem {

  // Stable neohookean material model
  class StableNeohookean : public MaterialModel {
  public:
    
    StableNeohookean(const std::shared_ptr<MaterialConfig>& config)
        : MaterialModel("Stable Neohookean", config) {}

    double energy(const Eigen::Vector6d& S) override; 

    Eigen::Vector6d gradient(const Eigen::Vector6d& S) override; 

    Eigen::Matrix6d hessian_inv(const Eigen::Matrix3d& R,
        const Eigen::Vector6d& S) override;
    Eigen::Matrix6d hessian_inv(const Eigen::Matrix3d& R,
        const Eigen::Vector6d& S, double kappa) override;
    Eigen::Matrix6d hessian(const Eigen::Vector6d& S) override;
  };


}
