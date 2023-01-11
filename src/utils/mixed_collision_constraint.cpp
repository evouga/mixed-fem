#include "mixed_collision_constraint.h"

#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/edge_edge_mollifier.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/point_triangle.hpp>

namespace ipc {

  ///////////////////////////////////////////////////////////////////////////////
  
  EdgeVertexMixedConstraint::EdgeVertexMixedConstraint(long edge_index, long vertex_index)
    : EdgeVertexConstraint(edge_index, vertex_index)
  {}

  EdgeVertexMixedConstraint::EdgeVertexMixedConstraint(const EdgeVertexCandidate& candidate)
      : EdgeVertexConstraint(candidate)
  {}

  double EdgeVertexMixedConstraint::compute_distance(
      const Eigen::MatrixXd& V,
      const Eigen::MatrixXi& E,
      const Eigen::MatrixXi& F,
      const DistanceMode dmode) const
  {
    return point_edge_distance(
        V.row(vertex_index), V.row(E(edge_index, 0)), V.row(E(edge_index, 1)),
        dtype, dmode);
  }

  VectorMax12d EdgeVertexMixedConstraint::compute_distance_gradient(
      const Eigen::MatrixXd& V,
      const Eigen::MatrixXi& E,
      const Eigen::MatrixXi& F,
      const DistanceMode dmode) const
  {
    VectorMax9d distance_grad;
    point_edge_distance_gradient(
        V.row(vertex_index), V.row(E(edge_index, 0)), V.row(E(edge_index, 1)),
        dtype, dmode, distance_grad);
    return distance_grad;
  }

  MatrixMax12d EdgeVertexMixedConstraint::compute_distance_hessian(
      const Eigen::MatrixXd& V,
      const Eigen::MatrixXi& E,
      const Eigen::MatrixXi& F,
      const DistanceMode dmode) const 
  {
    MatrixMax9d distance_hess;
    point_edge_distance_hessian(
        V.row(vertex_index), V.row(E(edge_index, 0)), V.row(E(edge_index, 1)),
        dtype, dmode, distance_hess);
    return distance_hess;      
  }

  ///////////////////////////////////////////////////////////////////////////////

  FaceVertexMixedConstraint::FaceVertexMixedConstraint(long face_index,
      long vertex_index) : FaceVertexConstraint(face_index, vertex_index)
  {
  }

  FaceVertexMixedConstraint::FaceVertexMixedConstraint(
      const FaceVertexCandidate& candidate) : FaceVertexConstraint(candidate)
  {
  }

  double FaceVertexMixedConstraint::compute_distance(
      const Eigen::MatrixXd& V,
      const Eigen::MatrixXi& E,
      const Eigen::MatrixXi& F,
      const DistanceMode dmode) const
  {
      // The distance type is known because of construct_constraint_set()
      return point_triangle_distance(
          V.row(vertex_index), V.row(F(face_index, 0)), V.row(F(face_index, 1)),
          V.row(F(face_index, 2)), dtype, dmode);
  }

  VectorMax12d FaceVertexMixedConstraint::compute_distance_gradient(
      const Eigen::MatrixXd& V,
      const Eigen::MatrixXi& E,
      const Eigen::MatrixXi& F,
      const DistanceMode dmode) const
  {
      VectorMax12d distance_grad;
      point_triangle_distance_gradient(
          V.row(vertex_index), V.row(F(face_index, 0)), V.row(F(face_index, 1)),
          V.row(F(face_index, 2)), dtype, dmode, distance_grad);
      return distance_grad;
  }

  MatrixMax12d FaceVertexMixedConstraint::compute_distance_hessian(
      const Eigen::MatrixXd& V,
      const Eigen::MatrixXi& E,
      const Eigen::MatrixXi& F,
      const DistanceMode dmode) const
  {
      MatrixMax12d distance_hess;
      point_triangle_distance_hessian(
          V.row(vertex_index), V.row(F(face_index, 0)), V.row(F(face_index, 1)),
          V.row(F(face_index, 2)), dtype, dmode, distance_hess);
      return distance_hess;
  }
  ///////////////////////////////////////////////////////////////////////////////

  EdgeEdgeMixedConstraint::EdgeEdgeMixedConstraint(
      long edge0_index, long edge1_index, double eps_x)
      : EdgeEdgeConstraint(edge0_index, edge1_index, eps_x)
  {}

  EdgeEdgeMixedConstraint::EdgeEdgeMixedConstraint(
      const EdgeEdgeCandidate& candidate, double eps_x)
      : EdgeEdgeConstraint(candidate, eps_x)
  {}
}