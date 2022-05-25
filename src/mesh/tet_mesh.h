#pragma once

#include "mesh.h"
#include "config.h"

namespace mfem {

  // Simulation Mesh for tetrahedral mesh
  class TetrahedralMesh : public Mesh {
  public:

    TetrahedralMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        std::shared_ptr<MaterialModel> material,
        std::shared_ptr<MaterialConfig> material_config)
        : Mesh(V,T,material,material_config) {
    }

    virtual void volumes(Eigen::VectorXd& vol) override;
    virtual void mass_matrix(Eigen::SparseMatrixdRowMajor& M,
        const Eigen::VectorXd& vols) override;
    virtual void jacobian(Eigen::SparseMatrixdRowMajor& J,
        const Eigen::VectorXd& vols, bool weighted) override;
    virtual void jacobian(std::vector<Eigen::MatrixXd>& J) override;
    virtual bool update_jacobian(std::vector<Eigen::MatrixXd>& J) override;
  };
}
