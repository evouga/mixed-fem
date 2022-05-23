#pragma once

#include <Eigen/Dense>
#include <EigenTypes.h>
#include <memory>

#include "materials/material_model.h"
#include "config.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif


namespace Eigen {
  using SparseMatrixdRowMajor = Eigen::SparseMatrix<double, Eigen::RowMajor>;
}

namespace mfem {

  // Class to maintain the state and perform physics updates on an object,
  // which has a particular discretization, material, and material config
  class Mesh {
  public:

    Mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        std::shared_ptr<MaterialModel> material,
        std::shared_ptr<MaterialConfig> material_config);
    
    // Compute per-element volumes. Size of "vol" is reset
    // vol - nelem x 1 per-element volumes
    virtual void volumes(Eigen::VectorXd& vol) = 0;

    // Compute mass matrix
    // M    - sparse matrix
    // vol  - nelem x 1 per-element volumes
    virtual void mass_matrix(Eigen::SparseMatrixdRowMajor& M,
        const Eigen::VectorXd& vols) = 0;

    // Computes dF/dq jacobian matrix where F is the vectorized deformation
    // gradient, and q is the vector of position configuration
    // M        - sparse matrix
    // vol      - nelem x 1 per-element volumes
    // weighted - boolean on whether to apply volume weights to jacobian
    virtual void jacobian(Eigen::SparseMatrixdRowMajor& J,
        const Eigen::VectorXd& vols, bool weighted) = 0;

    // Computes per-element dF/dq jacobian matrix
    // J  - per-element jacobian matrix
    virtual void jacobian(std::vector<Eigen::Matrix<double,9,12>>& J) {
      std::cerr << "jacobian not implemented!" << std::endl;
    }

    Eigen::MatrixXd vertices() {
      return V_;
    }

    void clear_fixed_vertices() {
      fixed_vertices_.clear();
      is_fixed_.setZero();
    }

    void free_vertex(int id) {
      fixed_vertices_.erase(fixed_vertices_.begin() + id);
      is_fixed_(id) = 0;
    }

    void set_fixed(int id) {
      is_fixed_(id) = 1;
      fixed_vertices_.push_back(id);
    }

    void set_fixed(const std::vector<int>& ids) {
      for (size_t i = 0; i < ids.size(); ++i) {
        is_fixed_(ids[i]) = 1;
      }
      fixed_vertices_.insert(fixed_vertices_.end(), ids.begin(), ids.end());
    }

  public:

    std::vector<std::vector<int>> bc_groups_;
    std::vector<int> fixed_vertices_;
    Eigen::VectorXi is_fixed_;
    Eigen::Matrix23x<double> bbox;

  
    Eigen::MatrixXd V_;
    Eigen::MatrixXd V0_;
    Eigen::MatrixXi T_;

    std::shared_ptr<MaterialModel> material_;
    std::shared_ptr<MaterialConfig> config_;
  };
}

// Add discretizations
#include "mesh/tet_mesh.h"
#include "mesh/tri_mesh.h"
#include "mesh/rod_mesh.h"
