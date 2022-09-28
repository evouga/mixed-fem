#pragma once

#include <Eigen/Core>
#include <ipc/collision_constraint.hpp>
#include <ipc/utils/unordered_map_and_set.hpp>
#include <ipc/collision_mesh.hpp>

namespace ipc {

  struct MixedConstraint {
    double distance;
    double lambda;
  };

  struct EdgeVertexMixedConstraint : EdgeVertexConstraint, MixedConstraint {
    EdgeVertexMixedConstraint(long edge_index, long vertex_index);
    EdgeVertexMixedConstraint(const EdgeVertexCandidate& candidate);

    double compute_distance(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& E,
        const Eigen::MatrixXi& F,
        const DistanceMode dmode = DistanceMode::SQUARED) const override;

    VectorMax12d compute_distance_gradient(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& E,
        const Eigen::MatrixXi& F,
        const DistanceMode dmode = DistanceMode::SQUARED) const override;

    MatrixMax12d compute_distance_hessian(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& E,
        const Eigen::MatrixXi& F,
        const DistanceMode dmode = DistanceMode::SQUARED) const override;

    PointEdgeDistanceType dtype;
  };

  ///////////////////////////////////////////////////////////////////////////////

  struct FaceVertexMixedConstraint : FaceVertexConstraint, MixedConstraint {
    FaceVertexMixedConstraint(long face_index, long vertex_index);
    FaceVertexMixedConstraint(const FaceVertexCandidate& candidate);

    double compute_distance(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& E,
        const Eigen::MatrixXi& F,
        const DistanceMode dmode = DistanceMode::SQUARED) const override;

    VectorMax12d compute_distance_gradient(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& E,
        const Eigen::MatrixXi& F,
        const DistanceMode dmode = DistanceMode::SQUARED) const override;

    MatrixMax12d compute_distance_hessian(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& E,
        const Eigen::MatrixXi& F,
        const DistanceMode dmode = DistanceMode::SQUARED) const override;
    
    PointTriangleDistanceType dtype;
  };

  ///////////////////////////////////////////////////////////////////////////////

  struct EdgeEdgeMixedConstraint : EdgeEdgeConstraint, MixedConstraint {
    EdgeEdgeMixedConstraint(long edge0_index, long edge1_index, double eps_x);
    EdgeEdgeMixedConstraint(const EdgeEdgeCandidate& candidate, double eps_x);
  };

  ///////////////////////////////////////////////////////////////////////////////

  struct MixedConstraints {
      std::vector<EdgeVertexMixedConstraint> ev_constraints;
      std::vector<EdgeEdgeMixedConstraint> ee_constraints;
      std::vector<FaceVertexMixedConstraint> fv_constraints;

      void clear();
      size_t size() const;

      // size_t num_constraints() const;

      bool empty() const;


      CollisionConstraint& operator[](size_t idx);
      const CollisionConstraint& operator[](size_t idx) const;
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