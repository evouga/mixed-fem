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
    virtual void deformation_gradient(const Eigen::VectorXd& x, Eigen::VectorXd& F) = 0;
    virtual void init_jacobian() {};
    virtual void update_jacobian(const Eigen::VectorXd& x) {}
    virtual const Eigen::SparseMatrixdRowMajor& jacobian() {
      return PJW_;
    }

    virtual const std::vector<Eigen::MatrixXd>& local_jacobians() {
      return Jloc_;
    }

    virtual const Eigen::VectorXd& volumes() {
      return vols_;
    }

    Eigen::MatrixXd vertices() {
      return V_;
    }

    void clear_fixed_vertices();

    void free_vertex(int id);

    void set_fixed(int id);

    void set_fixed(const std::vector<int>& ids);

    void update_free_map();

  public:

    std::vector<std::vector<int>> bc_groups_;
    std::vector<int> fixed_vertices_;
    Eigen::VectorXi is_fixed_;
    std::vector<int> free_map_;
    Eigen::Matrix23x<double> bbox;

  
    Eigen::MatrixXd V_;
    Eigen::MatrixXd V0_;
    Eigen::MatrixXi T_;

    std::shared_ptr<MaterialModel> material_;
    std::shared_ptr<MaterialConfig> config_;

  protected:
    // Weighted jacobian matrix with dirichlet BCs projected out
    Eigen::SparseMatrixdRowMajor PJW_;
    Eigen::SparseMatrixdRowMajor J_;
    Eigen::SparseMatrixd P_;          // pinning constraint (for vertices)
    Eigen::SparseMatrixd W_; // weight matrix
    Eigen::VectorXd vols_;
    std::vector<Eigen::MatrixXd> Jloc_;

  };
}

// Add discretizations
// TODO move to factory
#include "mesh/tet_mesh.h"
#include "mesh/tri_mesh.h"
#include "mesh/rod_mesh.h"
