#pragma once

#include "boundary_condition.h"

namespace mfem {

  // Boundary condition (BC) with no fixed points. Basically a no-op used for
  // default BCs when none are specified.
  class NullBC : public BoundaryCondition {
  public:
    NullBC(std::shared_ptr<Mesh> mesh, const BoundaryConditionConfig& config)
        : BoundaryCondition(mesh, config) {
      int N = mesh->V_.rows();
      is_fixed_ = Eigen::VectorXi::Zero(N);
      free_map_ = Eigen::VectorXi::LinSpaced(N, 0, N-1);
    }
  };

  // BC with no fixed points, but scales the vertices of the mesh, inducing
  // an initial deformation.
  class ScaleBC : public NullBC {
  public:
    ScaleBC(std::shared_ptr<Mesh> mesh, const BoundaryConditionConfig& config)
        : NullBC(mesh, config) {

      int N = mesh->V_.cols();
      Eigen::MatrixXd T(N, N);
      T.setZero();
      T.diagonal().array() = 1.5;
      mesh->V_ = mesh->V_ * T;
    }
  };

  // BC with no fixed points, but randomizes the vertex positions.
  class RandomizeBC : public NullBC {
  public:
    RandomizeBC(std::shared_ptr<Mesh> mesh,
        const BoundaryConditionConfig& config)
        : NullBC(mesh, config) {

      Eigen::RowVectorXd bmin = mesh->V_.colwise().minCoeff();
      Eigen::RowVectorXd bmax = mesh->V_.colwise().maxCoeff();
      Eigen::RowVectorXd offset = 0.5 * (bmin + bmax);
      Eigen::RowVectorXd scale = (bmax-bmin);
      mesh->V_.setRandom();
      mesh->V_ = (mesh->V_.array() + 1.0) * 0.5;
      mesh->V_ *= scale.asDiagonal();
      mesh->V_.rowwise() += offset;
    }
  };

  // BC for pinning a single point.
  class OnePointBC : public BoundaryCondition {
  public:
    OnePointBC(std::shared_ptr<Mesh> mesh,
        const BoundaryConditionConfig& config)
        : BoundaryCondition(mesh, config) {
      int N = mesh->V_.rows();
      is_fixed_ = Eigen::VectorXi::Zero(N);
      update_free_map();
    }
  };

  // Pins two points at opposing ends of the mesh.
  class HangBC : public BoundaryCondition {
  public:
    HangBC(std::shared_ptr<Mesh> mesh, const BoundaryConditionConfig& config)
        : BoundaryCondition(mesh, config) {
      
      is_fixed_ = Eigen::VectorXi::Zero(mesh->V_.rows());
      for (const auto& group : groups_) {
        is_fixed_(group.back()) = 1;
      }
      update_free_map();
    }
  };

  // Pins one end of the mesh
  class HangEndsBC : public BoundaryCondition {
  public:
    HangEndsBC(std::shared_ptr<Mesh> mesh,
        const BoundaryConditionConfig& config)
        : BoundaryCondition(mesh, config) {
      is_fixed_ = Eigen::VectorXi::Zero(mesh->V_.rows());

      for (int i : groups_[group_id_]) {
        is_fixed_(i) = 1;
      }
      update_free_map();
    }
  private:
    int group_id_ = 1;
  };
}
