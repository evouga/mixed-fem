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

    Eigen::Matrix6d hessian(const Eigen::Vector6d& S) override;
  };

  // Standard neohookean material model
  class Neohookean : public MaterialModel {
  public:
    
    Neohookean(const std::shared_ptr<MaterialConfig>& config)
        : MaterialModel("Neohookean", config) {}

    double energy(const Eigen::Vector6d& S) override; 
    Eigen::Vector6d gradient(const Eigen::Vector6d& S) override; 
    Eigen::Matrix6d hessian(const Eigen::Vector6d& S) override;

    double energy(const Eigen::Vector9d& F) override;
    Eigen::Vector9d gradient(const Eigen::Vector9d& F) override;
    Eigen::Matrix9d hessian(const Eigen::Vector9d& F) override;
  };


}
