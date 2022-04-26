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
    
    MaterialModel(const std::string& name,
        const std::shared_ptr<MaterialConfig>& config) 
        : name_(name), config_(config) {}

    // Computes psi, the strain energy density value.
    // S - 3x3 symmetric deformation
    virtual double energy(const Eigen::Vector6d& S) = 0; 

    // Gradient with respect to symmetric deformation, S
    // R - 3x3 rotation
    // S - 3x3 symmetric deformation
    virtual Eigen::Vector6d gradient(const Eigen::Matrix3d& R,
        const Eigen::Vector6d& S) = 0;

    // Hessian matrix for symmetric deformation, S
    // R - 3x3 rotation
    // S - 3x3 symmetric deformation
    virtual Eigen::Matrix6d hessian_inv(const Eigen::Matrix3d& R,
        const Eigen::Vector6d& S) = 0;
    virtual Eigen::Matrix6d hessian_inv(const Eigen::Matrix3d& R,
        const Eigen::Vector6d& S, double kappa) = 0;

    // Updates the compliance block entries in the KKT matrix.
    // Assumes the entries already exist and we can just overwite
    // the 3x3 blocks.
    virtual void update_compliance(int n, int m, 
        const std::vector<Eigen::Matrix3d>& R,
        const std::vector<Eigen::Matrix6d>& Hinv,
        const Eigen::VectorXd& vols, Eigen::SparseMatrixd& mat);

    // WHinvW matrix for the compliance block
    // R    - 3x3 rotation
    // Hinv - 6x6 deformation Hessian
    virtual Eigen::Matrix9d WHinvW(const Eigen::Matrix3d& R,
        const Eigen::Matrix6d& Hinv);

    // Right hand side contribution in KKT problem from the material model.
    // R - 3x3 rotation
    // S - 3x3 symmetric deformation
    virtual Eigen::Vector9d rhs(const Eigen::Matrix3d& R,
        const Eigen::Vector6d& S, const Eigen::Matrix6d& Hinv,
        const Eigen::Vector6d& g);

    // Computes S matrix update
    // R    - 3x3 rotation
    // S    - 3x3 symmetric deformation
    // L    - 9x1 lagrange multipliers
    // Hinv - 6x6 symmetric deformation Hessian
    virtual Eigen::Vector6d dS(const Eigen::Matrix3d& R, 
        const Eigen::Vector6d& S, const Eigen::Vector9d& L,
        const Eigen::Matrix6d& Hinv);

    std::string name() {
      return name_;
    }

  protected:

    // Writes the local 9x9 elemental compliance entries to the global
    // compliance block.
    static void fill_compliance_block(int offset, int row, double vol,
        double tol, const Eigen::Matrix9d& WHiW, Eigen::SparseMatrixd& mat);

    std::string name_;
    std::shared_ptr<MaterialConfig> config_;     

  };

}

// Add material models
#include "materials/neohookean_model.h"
#include "materials/corotational_model.h"
#include "materials/arap_model.h"
