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
        std::shared_ptr<MaterialConfig> material_config)
        : Mesh(V,T,material,material_config), N_(N) {
      NN_.resize(T_.rows());
      for (int i = 0; i < T_.rows(); ++i) {
        NN_[i] = N.row(i).transpose() * N.row(i);
      }
    }

    virtual void volumes(Eigen::VectorXd& vol) override;
    virtual void mass_matrix(Eigen::SparseMatrixdRowMajor& M,
        const Eigen::VectorXd& vols) override;
    virtual void jacobian(Eigen::SparseMatrixdRowMajor& J,
        const Eigen::VectorXd& vols, bool weighted) override;
    virtual void jacobian(std::vector<Eigen::MatrixXd>& J) override;
    virtual bool update_jacobian(std::vector<Eigen::MatrixXd>& J) override;

  private:
    Eigen::MatrixXd N_;
    std::vector<Eigen::Matrix3d> NN_; // N * N^T (normal outer product)
  };
}
