#pragma once

#include "objects/simulation_object.h"
#include "config.h"

namespace mfem {

  // Simulation Object for tetrahedral mesh
  class TetrahedralObject : public SimObject {
  public:

    TetrahedralObject(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        std::shared_ptr<MaterialModel> material,
        std::shared_ptr<MaterialConfig> material_config)
        : SimObject(V,T,material,material_config) {
    }

    virtual void volumes(Eigen::VectorXd& vol) override;
    virtual void mass_matrix(Eigen::SparseMatrixd& M,
        const Eigen::VectorXd& vols) override;
    virtual void jacobian(Eigen::SparseMatrixdRowMajor& J,
        const Eigen::VectorXd& vols, bool weighted) override;
    virtual void jacobian(std::vector<Eigen::Matrix<double,9,12>>& J) override;
  };
}
