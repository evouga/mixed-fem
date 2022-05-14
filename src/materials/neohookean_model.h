#pragma once

#include "materials/material_model.h"

namespace mfem {

  // Standard neohookean material model
  class Neohookean : public MaterialModel {
  public:
    
    Neohookean(const std::shared_ptr<MaterialConfig>& config)
        : MaterialModel("Neohookean", config) {}

    double energy(const Eigen::Vector6d& S) override; 
    Eigen::Vector6d gradient(const Eigen::Vector6d& S) override; 
    Eigen::Matrix6d hessian(const Eigen::Vector6d& S,
        bool psd_fix = true) override;

    double energy(const Eigen::Vector9d& F) override;
    Eigen::Vector9d gradient(const Eigen::Vector9d& F) override;
    Eigen::Matrix9d hessian(const Eigen::Vector9d& F) override;
  };

}
