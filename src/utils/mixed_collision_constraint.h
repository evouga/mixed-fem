#pragma once

#include <Eigen/Core>
#include <ipc/collision_constraint.hpp>

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

}