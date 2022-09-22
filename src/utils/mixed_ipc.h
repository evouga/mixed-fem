#pragma once

#include <Eigen/Core>
#include <ipc/collision_constraint.hpp>
#include <ipc/utils/unordered_map_and_set.hpp>
#include <ipc/collision_mesh.hpp>

namespace ipc {

  ///////////////////////////////////////////////////////////////////////////////

  struct MixedState {
    Constraints constraint_set;
    unordered_map<EdgeVertexCandidate, long> ev_map;
    unordered_map<EdgeEdgeCandidate, long> ee_map;
    unordered_map<FaceVertexCandidate, long> fe_map;
    Eigen::VectorXd d;
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
      Constraints& constraint_set,
      MixedState& state,
      const double dmin = 0);



  


}