#pragma once

#include "objects/simulation_object.h"
#include "config.h"

namespace mfem {

  // Simulation Object for triangle mesh
  class TriObject : public SimObject {
  public:

    TriObject(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        const Eigen::MatrixXd& N, std::shared_ptr<SimConfig> config,
        std::shared_ptr<MaterialModel> material,
        std::shared_ptr<MaterialConfig> material_config)
        : SimObject(V,T,config,material,material_config) {
      NN_.resize(T_.rows());
      for (int i = 0; i < T_.rows(); ++i) {
        NN_[i] = N.row(i).transpose() * N.row(i);
      }
    }

    void volumes(Eigen::VectorXd& vol) override;
    void mass_matrix(Eigen::SparseMatrixd& M,
        const Eigen::VectorXd& vols) override;
    void jacobian(SparseMatrixdRowMajor& J, const Eigen::VectorXd& vols,
        bool weighted) override;

  private:
    std::vector<Eigen::Matrix3d> NN_; // N * N^T (normal outer product)
  };
}
