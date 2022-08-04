#pragma once

#include <Eigen/Dense>
#include <EigenTypes.h>
#include <memory>

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif


namespace Eigen {
  using SparseMatrixdRowMajor = Eigen::SparseMatrix<double, Eigen::RowMajor>;
}

namespace mfem {

  class MaterialConfig;
  class MaterialModel;

  enum MatrixType {
    FULL,         // Full matrix
    PROJECT_ROWS, // Project BCs from row entries
    PROJECTED     // Project out both row and columns
  };

  // Class to maintain the state and perform physics updates on an object,
  // which has a particular discretization, material, and material config
  class Mesh {
  public:

    Mesh() {}
    Mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        std::shared_ptr<MaterialModel> material,
        std::shared_ptr<MaterialConfig> material_config);
    
    virtual void init();

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
    virtual void jacobian(std::vector<Eigen::MatrixXd>& J) {
      std::cerr << "jacobian not implemented!" << std::endl;
    }

    virtual bool fixed_jacobian() { return true; }

    // Defo gradients
    virtual void deformation_gradient(const Eigen::VectorXd& x,
        Eigen::VectorXd& F) = 0;
    virtual void init_jacobian() {};
    virtual void update_jacobian(const Eigen::VectorXd& x) {}
    virtual const Eigen::SparseMatrixdRowMajor& jacobian() {
      return PJW_;
    }

    virtual const std::vector<Eigen::MatrixXd>& local_jacobians() {
      return Jloc_;
    }

    virtual const Eigen::VectorXd& volumes() const {
      return vols_;
    }

    virtual const Eigen::MatrixXd& vertices() const {
      return V_;
    }

    Eigen::SparseMatrixdRowMajor laplacian() {
      return P_ * (J_.transpose() * W_ * J_) * P_.transpose();
    }

    template<MatrixType T = MatrixType::PROJECTED>
    const Eigen::SparseMatrixdRowMajor& mass_matrix() {
      if constexpr (T == MatrixType::PROJECTED) {
        return PMP_;
      } else if constexpr (T == MatrixType::PROJECT_ROWS) {
        return PM_;
      } else {
        return M_;
      }
    }

    const Eigen::SparseMatrixdRowMajor& projection_matrix() {
      return P_;
    }


    // I don't like this
    void clear_fixed_vertices();

    void free_vertex(int id);

    void set_fixed(int id);

    void set_fixed(const std::vector<int>& ids);

    void update_free_map();

  public:

    // I don't like this
    std::vector<std::vector<int>> bc_groups_;
    std::vector<int> fixed_vertices_;
    Eigen::VectorXi is_fixed_;
    std::vector<int> free_map_;
    Eigen::Matrix23x<double> bbox;
  
    Eigen::MatrixXd V_;
    Eigen::MatrixXd V0_;
    Eigen::MatrixXi T_;
    Eigen::MatrixXi F_;

    std::shared_ptr<MaterialModel> material_;
    std::shared_ptr<MaterialConfig> config_;

  protected:
    // Weighted jacobian matrix with dirichlet BCs projected out
    Eigen::SparseMatrixdRowMajor PJW_;
    Eigen::SparseMatrixdRowMajor J_;   // Shape function jacobian
    Eigen::SparseMatrixdRowMajor PMP_; // Mass matrix (dirichlet BCs projected)
    Eigen::SparseMatrixdRowMajor PM_;  // Mass matrix (rows projected)
    Eigen::SparseMatrixdRowMajor M_;   // Mass matrix (full matrix)
    Eigen::SparseMatrixdRowMajor P_;   // pinning matrix (for dirichlet BCs) 
    //Eigen::SparseMatrixd P_;         // pinning constraint (for vertices)
    Eigen::SparseMatrixd W_;           // weight matrix
    Eigen::VectorXd vols_;
    std::vector<Eigen::MatrixXd> Jloc_;

  };
}

// Add discretizations
// TODO move to factory
#include "mesh/tet_mesh.h"
#include "mesh/tri_mesh.h"
#include "mesh/rod_mesh.h"
