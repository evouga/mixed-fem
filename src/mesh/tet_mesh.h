#pragma once

#include "mesh.h"

namespace mfem {

  // Simulation Mesh for tetrahedral mesh
  class TetrahedralMesh : public Mesh {
  public:

    TetrahedralMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        std::shared_ptr<MaterialConfig> config,
        const std::vector<Eigen::VectorXi>& subsets,
        const std::vector<std::shared_ptr<MaterialModel>>& materials)
        : Mesh(V,T,config,subsets,materials) {
    }

    TetrahedralMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        std::shared_ptr<MaterialModel> material,
        std::shared_ptr<MaterialConfig> material_config)
        : Mesh(V,T,material,material_config) {
    }

    void volumes(Eigen::VectorXd& vol) override;
    void mass_matrix(Eigen::SparseMatrixdRowMajor& M,
        const Eigen::VectorXd& vols) override;
    void jacobian(Eigen::SparseMatrixdRowMajor& J,
        const Eigen::VectorXd& vols, bool weighted) override;
    void jacobian(std::vector<Eigen::MatrixXd>& J) override;
    void deformation_gradient(const Eigen::VectorXd& x,
        Eigen::VectorXd& F) override;
    void init_jacobian() override;

  };
}
