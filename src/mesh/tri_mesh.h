#pragma once

#include "mesh/mesh.h"
#include "config.h"

namespace mfem {

  // Simulation Mesh for triangle mesh
  class TriMesh : public Mesh {
  public:

    TriMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        const Eigen::MatrixXd& N,
        std::shared_ptr<MaterialModel> material,
        std::shared_ptr<MaterialConfig> material_config);

    virtual void volumes(Eigen::VectorXd& vol) override;
    virtual void mass_matrix(Eigen::SparseMatrixdRowMajor& M,
        const Eigen::VectorXd& vols) override;
    virtual void jacobian(Eigen::SparseMatrixdRowMajor& J,
        const Eigen::VectorXd& vols, bool weighted) override;
    virtual void jacobian(std::vector<Eigen::MatrixXd>& J) override;
    virtual bool update_jacobian(std::vector<Eigen::MatrixXd>& J) override;
    virtual bool update_jacobian(Eigen::SparseMatrixdRowMajor& J) override;

    const Eigen::SparseMatrixdRowMajor& J() {
      return J_;
    }

    virtual bool fixed_jacobian() override { 
      return false;
    }

    Eigen::MatrixXd N_;

  private:
    Eigen::MatrixXd dphidX_;

    // Weighted jacobian matrix with dirichlet BCs projected out
    Eigen::SparseMatrixdRowMajor PJW_;
    Eigen::SparseMatrixdRowMajor J_;
    std::vector<Eigen::MatrixXd> Jloc_;


  };
}
