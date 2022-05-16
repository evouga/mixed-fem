#pragma once

#include "objects/simulation_object.h"
#include "config.h"

namespace mfem {

  // Simulation Object for triangle mesh
  class RodObject : public SimObject {
  public:

    RodObject(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        const Eigen::MatrixXd& N, const Eigen::MatrixXd& BN,
        std::shared_ptr<MaterialModel> material,
        std::shared_ptr<MaterialConfig> material_config)
        : SimObject(V,T,material,material_config) {
      
      NN_.resize(T_.rows());
      BN_.resize(T_.rows());
      for (int i = 0; i < T_.rows(); ++i) {
        NN_[i] = N.row(i).transpose() * N.row(i);
        BN_[i] = BN.row(i).transpose() * BN.row(i);
      }
    }

    void volumes(Eigen::VectorXd& vol) override;
    void mass_matrix(Eigen::SparseMatrixd& M,
        const Eigen::VectorXd& vols) override;
    void jacobian(Eigen::SparseMatrixdRowMajor& J, const Eigen::VectorXd& vols,
        bool weighted) override;

    std::vector<Eigen::Matrix3d> NN_; // N * N^T (normal outer product)
    std::vector<Eigen::Matrix3d> BN_; // BN * BN^T (binormal outer product)
  };
}
