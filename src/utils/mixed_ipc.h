#pragma once

#include <Eigen/Core>
#include <ipc/collision_constraint.hpp>
#include <ipc/utils/unordered_map_and_set.hpp>
#include <ipc/collision_mesh.hpp>

namespace ipc {

  struct MixedConstraints : public Constraints {
      // std::vector<EdgeVertexConstraint> ev_constraints;
      // std::vector<EdgeEdgeConstraint> ee_constraints;
      // std::vector<FaceVertexConstraint> fv_constraints;
      // // std::vector<
      // size_t size() const;

      // size_t num_constraints() const;

      // bool empty() const;

      void clear();

      double& distance(size_t idx);
      const double& distance(size_t idx) const;
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

      std::vector<double> ev_distances;
      std::vector<double> ee_distances;
      std::vector<double> fv_distances;
      std::vector<double> ev_lambdas;
      std::vector<double> ee_lambdas;
      std::vector<double> fv_lambdas;

  };

  ///////////////////////////////////////////////////////////////////////////////

  struct MixedState {
    MixedConstraints constraint_set;
    // unordered_map<EdgeVertexConstraint, long> ev_map;
    // unordered_map<EdgeEdgeConstraint, long> ee_map;
    // unordered_map<FaceVertexConstraint, long> fv_map;
    // Eigen::VectorXd d;
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