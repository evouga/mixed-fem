#pragma once

#include "materials/material_model.h"

namespace mfem {

  // as-rigid-as-possible material model
  class ArapModel : public MaterialModel {
  public:
    
    static std::string name() {
      return "ARAP";
    }

    ArapModel(const std::shared_ptr<MaterialConfig>& config)
        : MaterialModel(config) {}

    double energy(const Eigen::Vector6d& S) override; 

    Eigen::Vector6d gradient(const Eigen::Vector6d& S) override; 

    Eigen::Matrix6d hessian_inv(const Eigen::Vector6d& S);
    Eigen::Matrix6d hessian(const Eigen::Vector6d& S,
        bool psd_fix = true) override;
  };


}
