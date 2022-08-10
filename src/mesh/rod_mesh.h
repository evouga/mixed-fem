#pragma once

#include "mesh/mesh.h"

namespace mfem {

  // Simulation Mesh for triangle mesh
  class RodMesh : public Mesh {
  public:

    RodMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        const Eigen::MatrixXd& N, const Eigen::MatrixXd& BN,
        std::shared_ptr<MaterialModel> material)
        : Mesh(V,T,material) {
      
      NN_.resize(T_.rows());
      BN_.resize(T_.rows());
      for (int i = 0; i < T_.rows(); ++i) {
        NN_[i] = N.row(i).transpose() * N.row(i);
        BN_[i] = BN.row(i).transpose() * BN.row(i);
      }
    }

    void volumes(Eigen::VectorXd& vol) override;
    void mass_matrix(Eigen::SparseMatrixdRowMajor& M,
        const Eigen::VectorXd& vols) override;
    void jacobian(Eigen::SparseMatrixdRowMajor& J, const Eigen::VectorXd& vols,
        bool weighted) override;

    void deformation_gradient(const Eigen::VectorXd& x,
        Eigen::VectorXd& F) override {
          
      std::cout << "rod jacobian update not implemented!" << std::endl;
    }


    std::vector<Eigen::Matrix3d> NN_; // N * N^T (normal outer product)
    std::vector<Eigen::Matrix3d> BN_; // BN * BN^T (binormal outer product)
  };
}
