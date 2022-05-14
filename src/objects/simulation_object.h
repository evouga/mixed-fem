#pragma once

#include <Eigen/Dense>
#include <EigenTypes.h>
#include <memory>

#include "materials/material_model.h"
#include "config.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

namespace mfem {

  using SparseMatrixdRowMajor = Eigen::SparseMatrix<double, Eigen::RowMajor>;

  // Class to maintain the state and perform physics updates on an object,
  // which has a particular discretization, material, and material config
  class SimObject {
  public:

    SimObject(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        std::shared_ptr<MaterialModel> material,
        std::shared_ptr<MaterialConfig> material_config)
        : V_(V), V0_(V), T_(T), material_(material),
          config_(material_config) {
    }
    
    virtual void volumes(Eigen::VectorXd& vol) = 0;
    virtual void mass_matrix(Eigen::SparseMatrixd& M,
        const Eigen::VectorXd& vols) = 0;
    virtual void jacobian(SparseMatrixdRowMajor& J,
        const Eigen::VectorXd& vols, bool weighted) = 0;

    Eigen::MatrixXd vertices() {
      return V_;
    }

  public:
  
    Eigen::MatrixXd V_;
    Eigen::MatrixXd V0_;
    Eigen::MatrixXi T_;

    std::shared_ptr<MaterialModel> material_;
    std::shared_ptr<MaterialConfig> config_;


  };
}

// Add discretizations
#include "objects/tet_object.h"
#include "objects/tri_object.h"
#include "objects/rod_object.h"
