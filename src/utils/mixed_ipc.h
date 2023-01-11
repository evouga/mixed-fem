#pragma once

#include <Eigen/Core>
#include <ipc/utils/unordered_map_and_set.hpp>
#include <ipc/collision_mesh.hpp>
#include "mixed_collision_constraint.h"

namespace ipc {

  struct MixedConstraints {
      std::vector<EdgeVertexMixedConstraint> ev_constraints;
      std::vector<EdgeEdgeMixedConstraint> ee_constraints;
      std::vector<FaceVertexMixedConstraint> fv_constraints;

      void clear();
      size_t size() const;

      // size_t num_constraints() const;

      bool empty() const;

      // Get collision constraint by index
      CollisionConstraint& operator[](size_t idx);
      const CollisionConstraint& operator[](size_t idx) const;

      // Get mixed distance for constraint by index
      double& distance(size_t idx);
      const double& distance(size_t idx) const;

      // Get constraint lagrange multiplier for constraint by index
      double& lambda(size_t idx);
      const double& lambda(size_t idx) const;

      void update_distances(const Eigen::VectorXd& distances);
      void update_lambdas(const Eigen::VectorXd& lambdas);

      Eigen::VectorXd get_distances() const {
        Eigen::VectorXd x(size());
        for (int i = 0; i < x.size(); ++i) {
          x(i) = distance(i);
        }
        return x;
      }

      Eigen::VectorXd get_lambdas() const {
        Eigen::VectorXd x(size());
        for (int i = 0; i < x.size(); ++i) {
          x(i) = lambda(i);
        }
        return x;
      }
  };

  /// @brief Construct a set of constraints used to compute the barrier potential.
  /// @param[in] candidates Distance candidates from which the constraint set is built.
  /// @param[in] mesh The collision mesh.
  /// @param[in] V Vertices of the collision mesh.
  /// @param[in] dhat The activation distance of the barrier.
  /// @param[out] constraint_set The constructed set of constraints (any existing constraints will be cleared).
  /// @param[in]  dmin  Minimum distance.
  void construct_constraint_set(
      const Candidates& candidates,
      const CollisionMesh& mesh,
      const Eigen::MatrixXd& V,
      const double dhat,
      MixedConstraints& constraint_set,
      const double dmin = 0);
}