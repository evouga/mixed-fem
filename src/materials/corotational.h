#pragma once

#include "materials/material_model.h"

namespace mfem {

  // corotational material model
  class Corotational : public MaterialModel {
  public:

    static std::string name() {
      return "COROT";
    }
    
    Corotational(const std::shared_ptr<MaterialConfig>& config)
        : MaterialModel(config) {}

    double energy(const Eigen::Vector6d& S) override; 

    Eigen::Vector6d gradient(const Eigen::Vector6d& S) override; 

    Eigen::Matrix6d hessian_inv(const Eigen::Vector6d& S);

    Eigen::Matrix6d hessian(const Eigen::Vector6d& S,
        bool psd_fix = true) override;
  };


}
