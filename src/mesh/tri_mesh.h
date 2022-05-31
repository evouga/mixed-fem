#pragma once

#include "mesh/mesh.h"

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

    void deformation_gradient(const Eigen::VectorXd& x,
        Eigen::VectorXd& F) override;
    void update_jacobian(const Eigen::VectorXd& x) override;
    void init_jacobian() override;

    virtual bool fixed_jacobian() override { 
      return false;
    }

    Eigen::MatrixXd N_;

  private:
    Eigen::MatrixXd dphidX_;

    // Constant components of the jacobian
    Eigen::SparseMatrixdRowMajor J0_;
    std::vector<Eigen::MatrixXd> Jloc0_;

  };
}
