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

    void volumes(Eigen::VectorXd& vol) override;
    void mass_matrix(Eigen::SparseMatrixd& M) override;
    void jacobian(SparseMatrixdRowMajor& J, bool weighted) override;
  };
}