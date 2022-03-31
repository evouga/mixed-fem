#pragma once

#include <EigenTypes.h>
#include <string>
#include <memory>

namespace mfem {

  // Base pure virtual class for material models
  class MaterialModel {
  public:
    
    MaterialModel(const std::string& name) : name_(name) {}

    // Updates the compliance block entries in the KKT matrix.
    // Assumes the entries already exist and we can just overwite
    // the 3x3 blocks.
    virtual void update_compliance(int n, int m, 
        const std::vector<Eigen::Matrix3d>& R,
        const std::vector<Eigen::Matrix6d>& Hinv,
        const Eigen::VectorXd& vols, Eigen::SparseMatrixd& mat) = 0;

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

    // WHinvW matrix for the compliance block
    // R    - 3x3 rotation
    // Hinv - 6x6 deformation Hessian
    virtual Eigen::Matrix9d WHinvW(const Eigen::Matrix3d& R,
        const Eigen::Matrix6d& Hinv) = 0;

    // Right hand side contribution in KKT problem from the material model.
    // R - 3x3 rotation
    // S - 3x3 symmetric deformation
    virtual Eigen::Vector9d rhs(const Eigen::Matrix3d& R,
        const Eigen::Vector6d& S, const Eigen::Matrix6d& Hinv,
        const Eigen::Vector6d& g) = 0;

    // Computes S matrix update
    // R    - 3x3 rotation
    // S    - 3x3 symmetric deformation
    // L    - 9x1 lagrange multipliers
    // Hinv - 6x6 symmetric deformation Hessian
    virtual Eigen::Vector6d dS(const Eigen::Matrix3d& R, 
        const Eigen::Vector6d& S, const Eigen::Vector9d& L,
        const Eigen::Matrix6d& Hinv) = 0;

    std::string name() {
      return name_;
    }

  protected:

    // Writes the local 9x9 elemental compliance entries to the global
    // compliance block.
    static void fill_compliance_block(int offset, int row, double vol,
        double tol, const Eigen::Matrix9d& WHiW, Eigen::SparseMatrixd& mat) {

      // Assign to last nine entries of the j-th column for the i-th block
      for (int j = 0; j < 9; ++j) {
        int offset_j = offset + row*9 + j;
        int colsize = (mat.outerIndexPtr()[offset_j+1] 
          - mat.outerIndexPtr()[offset_j]);
        int row_j = mat.outerIndexPtr()[offset_j] + colsize - 9;

        for (int k = 0; k < 9; ++k) {
          if (k==j) {
            mat.valuePtr()[row_j+k] = -vol*(WHiW(j,k)+tol);
          } else {
            mat.valuePtr()[row_j+k] = -vol*WHiW(j,k);
          }
        }
      }
    }

    std::string name_;

  };


}