#pragma once

#include <EigenTypes.h>
#include <string>
#include <memory>

#include "config.h"

namespace mfem {

  inline void Wmat(const Eigen::Matrix3d& R, Eigen::Matrix<double,9,6>& W) {
    W <<
      R(0,0), 0, 0, R(0,1), R(0,2), 0,
      R(1,0), 0, 0, R(1,1), R(1,2), 0,
      R(2,0), 0, 0, R(2,1), R(2,2), 0,
      0, R(0,1), 0, R(0,0), 0, R(0,2),
      0, R(1,1), 0, R(1,0), 0, R(1,2),
      0, R(2,1), 0, R(2,0), 0, R(2,2),
      0, 0, R(0,2), 0, R(0,0), R(0,1),
      0, 0, R(1,2), 0, R(1,0), R(1,1),
      0, 0, R(2,2), 0, R(2,0), R(2,1);
  }

  // Base pure virtual class for material models
  class MaterialModel {
  public:
    
    static std::string name() {
      return "base";
    }

    MaterialModel(const std::shared_ptr<MaterialConfig>& config) 
        : config_(config) {}

    // Computes psi, the strain energy density value.
    // S - 6x1 symmetric deformation
    virtual double energy(const Eigen::Vector6d& S) = 0; 

    // Gradient with respect to symmetric deformation, S
    // S - 6x1 symmetric deformation
    virtual Eigen::Vector6d gradient(const Eigen::Vector6d& S) = 0;

    // Hessian matrix for symmetric deformation
    // S - 6x1 symmetric deformation
    virtual Eigen::Matrix6d hessian(const Eigen::Vector6d& S,
        bool psd_fix = true) = 0;

    // Optional energy for non-mixed systems
    // Computes psi, the strain energy density value.
    // F - 9x1 deformation gradient flattened (column-major)
    virtual double energy(const Eigen::Vector9d& F) {
      std::cerr << "energy unimplemented for 9x1 input" << std::endl;
      return 0;
    }

    // Non-mixed gradient with respect to deformation gradient
    // F - 9x1 deformation gradient flattened (column-major)
    virtual Eigen::Vector9d gradient(const Eigen::Vector9d& F) {
      Eigen::Vector9d g;
      std::cerr << "gradient unimplemented for 9x1 input" << std::endl;
      return g;
    }

    // Non-mixed hessian 
    // F - 9x1 deformation gradient flattened (column-major)
    virtual Eigen::Matrix9d hessian(const Eigen::Vector9d& F) {
      std::cerr << "gradient unimplemented for 9x1 input" << std::endl;
      Eigen::Matrix9d H;
      return H;
    }

  protected:

    std::shared_ptr<MaterialConfig> config_;     

  };
}

// Add material models
#include "materials/neohookean_model.h"
#include "materials/corotational_model.h"
#include "materials/arap_model.h"
#include "materials/stable_nh_model.h"
#include "materials/fung.h"

