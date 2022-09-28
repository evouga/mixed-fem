#include "mixed_ipc.h"

#include <ipc/barrier/barrier.hpp>

#define IPC_EARLIEST_TOI_USE_MUTEX
#ifdef IPC_EARLIEST_TOI_USE_MUTEX
#include <mutex>
#endif
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>

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

  ///////////////////////////////////////////////////////////////////////////////

  size_t MixedConstraints::size() const
  {
      return ev_constraints.size() + ee_constraints.size()
          + fv_constraints.size();
  }

  bool MixedConstraints::empty() const
  {
      return ev_constraints.empty()
          && ee_constraints.empty() && fv_constraints.empty();
  }

  void MixedConstraints::clear() {
    ev_constraints.clear();
    ee_constraints.clear();
    fv_constraints.clear();
  }

  CollisionConstraint& MixedConstraints::operator[](size_t idx)
  {
    if (idx < ev_constraints.size()) {
        return ev_constraints[idx];
    }
    idx -= ev_constraints.size();
    if (idx < ee_constraints.size()) {
        return ee_constraints[idx];
    }
    idx -= ee_constraints.size();
    if (idx < fv_constraints.size()) {
        return fv_constraints[idx];
    }
    throw std::out_of_range("Constraint index is out of range!");
  }

  const CollisionConstraint& MixedConstraints::operator[](size_t idx) const
  {
    if (idx < ev_constraints.size()) {
        return ev_constraints[idx];
    }
    idx -= ev_constraints.size();
    if (idx < ee_constraints.size()) {
        return ee_constraints[idx];
    }
    idx -= ee_constraints.size();
    if (idx < fv_constraints.size()) {
        return fv_constraints[idx];
    }
    throw std::out_of_range("Constraint index is out of range!");
  }


  double& MixedConstraints::distance(size_t idx) {
    if (idx < ev_constraints.size()) {
      return ev_constraints[idx].distance;
    }
    idx -= ev_constraints.size();
    if (idx < ee_constraints.size()) {
      return ee_constraints[idx].distance;
    }
    idx -= ee_constraints.size();
    if (idx < fv_constraints.size()) {
      return fv_constraints[idx].distance;
    }
    throw std::out_of_range("Constraint index is out of range!");
  }

  const double& MixedConstraints::distance(size_t idx) const {
    if (idx < ev_constraints.size()) {
      return ev_constraints[idx].distance;
    }
    idx -= ev_constraints.size();
    if (idx < ee_constraints.size()) {
      return ee_constraints[idx].distance;
    }
    idx -= ee_constraints.size();
    if (idx < fv_constraints.size()) {
      return fv_constraints[idx].distance;
    }
    throw std::out_of_range("Constraint index is out of range!");
  }

  double& MixedConstraints::lambda(size_t idx) {
    if (idx < ev_constraints.size()) {
      return ev_constraints[idx].lambda;
    }
    idx -= ev_constraints.size();
    if (idx < ee_constraints.size()) {
      return ee_constraints[idx].lambda;
    }
    idx -= ee_constraints.size();
    if (idx < fv_constraints.size()) {
      return fv_constraints[idx].lambda;
    }
    throw std::out_of_range("Constraint index is out of range!");
  }

  const double& MixedConstraints::lambda(size_t idx) const {
    if (idx < ev_constraints.size()) {
      return ev_constraints[idx].lambda;
    }
    idx -= ev_constraints.size();
    if (idx < ee_constraints.size()) {
      return ee_constraints[idx].lambda;
    }
    idx -= ee_constraints.size();
    if (idx < fv_constraints.size()) {
      return fv_constraints[idx].lambda;
    }
    throw std::out_of_range("Constraint index is out of range!");
  }

  void MixedConstraints::update_distances(const Eigen::VectorXd& distances) {
    assert(distances.size() == size());
    for (size_t i = 0; i < distances.size(); ++i) {
      distance(i) = distances(i);
    }
  }
  
  void MixedConstraints::update_lambdas(const Eigen::VectorXd& lambdas) {
    assert(lambdas.size() == size());
    for (size_t i = 0; i < lambdas.size(); ++i) {
      lambda(i) = lambdas(i);
    }
  }
  
  template <typename T>
  void create_constraint_map(const std::vector<T>& constraints,
      unordered_set<T>& set) {
    for (size_t i = 0; i < constraints.size(); ++i) {
      set.emplace(constraints[i]);
    }  
  }

  void construct_constraint_set(
      const Candidates &candidates,
      const CollisionMesh &mesh,
      const Eigen::MatrixXd &V,
      const double dhat,
      MixedConstraints &constraint_set,
      const double dmin)
  {
    assert(V.rows() == mesh.num_vertices());

    MixedConstraints new_constraints;

    const Eigen::MatrixXd &V_rest = mesh.vertices_at_rest();
    const Eigen::MatrixXi &E = mesh.edges();
    const Eigen::MatrixXi &F = mesh.faces();
    const Eigen::MatrixXi &F2E = mesh.faces_to_edges();

    // Cull the candidates by measuring the distance and dropping those that are
    // greater than dhat.
    const double offset_sqr = (dmin + dhat) * (dmin + dhat);
    auto is_active = [&](double distance_sqr)
    {
      return distance_sqr < offset_sqr;
    };

    unordered_set<EdgeVertexMixedConstraint> ev_map;
    unordered_set<EdgeEdgeMixedConstraint> ee_map;
    unordered_set<FaceVertexMixedConstraint> fv_map;
    create_constraint_map(constraint_set.ev_constraints, ev_map);
    create_constraint_map(constraint_set.ee_constraints, ee_map);
    create_constraint_map(constraint_set.fv_constraints, fv_map); 

    std::mutex vv_mutex, ev_mutex, ee_mutex, fv_mutex;

    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), candidates.ev_candidates.size()),
        [&](const tbb::blocked_range<size_t> &r)
        {
          for (size_t i = r.begin(); i < r.end(); i++)
          {
            const auto &ev_candidate = candidates.ev_candidates[i];
            const auto &[ei, vi] = ev_candidate;
            long e0i = E(ei, 0), e1i = E(ei, 1);

            PointEdgeDistanceType dtype =
                point_edge_distance_type(V.row(vi), V.row(e0i), V.row(e1i));

            double distance_sqr = point_edge_distance(
                V.row(vi), V.row(e0i), V.row(e1i), dtype,
                DistanceMode::SQUARED);

            if (is_active(distance_sqr)) {
              std::lock_guard<std::mutex> lock(ev_mutex);
              // ev_candidates is a set, so no duplicate EV
              // constraints
              EdgeVertexMixedConstraint constraint(ev_candidate);

              auto found_item = ev_map.find(constraint);
              if (found_item != ev_map.end()) {
                constraint.distance = found_item->distance;
                constraint.lambda = found_item->lambda;
                ev_map.erase(constraint);
              } else {
                constraint.distance = std::sqrt(distance_sqr);
                constraint.lambda = 0.0;
              }
              constraint.dtype = dtype;
              new_constraints.ev_constraints.emplace_back(constraint);
            }
          }
        });

    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), candidates.ee_candidates.size()),
        [&](const tbb::blocked_range<size_t> &r)
        {
          for (size_t i = r.begin(); i < r.end(); i++)
          {
            const auto &ee_candidate = candidates.ee_candidates[i];
            const auto &[eai, ebi] = ee_candidate;
            long ea0i = E(eai, 0), ea1i = E(eai, 1);
            long eb0i = E(ebi, 0), eb1i = E(ebi, 1);

            EdgeEdgeDistanceType dtype = edge_edge_distance_type(
                V.row(ea0i), V.row(ea1i), V.row(eb0i), V.row(eb1i));

            double distance_sqr = edge_edge_distance(
                V.row(ea0i), V.row(ea1i), V.row(eb0i), V.row(eb1i), dtype,
                DistanceMode::SQUARED);

            if (is_active(distance_sqr))
            {
              double eps_x = edge_edge_mollifier_threshold(
                  V_rest.row(ea0i), V_rest.row(ea1i), //
                  V_rest.row(eb0i), V_rest.row(eb1i));
              std::lock_guard<std::mutex> lock(ee_mutex);

              EdgeEdgeMixedConstraint constraint(ee_candidate, eps_x);

              auto found_item = ee_map.find(constraint);
              if (found_item != ee_map.end()) {
                constraint.distance = found_item->distance;
                constraint.lambda = found_item->lambda;
                ee_map.erase(constraint);
              } else {
                constraint.distance = std::sqrt(distance_sqr);
                constraint.lambda = 0.0;
              }
              new_constraints.ee_constraints.emplace_back(constraint);
            }
          }
        });

    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), candidates.fv_candidates.size()),
        [&](const tbb::blocked_range<size_t> &r)
        {
          for (size_t i = r.begin(); i < r.end(); i++)
          {
            const auto &fv_candidate = candidates.fv_candidates[i];
            const auto &[fi, vi] = fv_candidate;
            long f0i = F(fi, 0), f1i = F(fi, 1), f2i = F(fi, 2);

            // Compute distance type
            PointTriangleDistanceType dtype = point_triangle_distance_type(
                V.row(fv_candidate.vertex_index), //
                V.row(f0i), V.row(f1i), V.row(f2i));

            double distance_sqr = point_triangle_distance(
                V.row(fv_candidate.vertex_index), //
                V.row(f0i), V.row(f1i), V.row(f2i), dtype,
                DistanceMode::SQUARED);

            if (is_active(distance_sqr))
            {
              std::lock_guard<std::mutex> lock(fv_mutex);
              
              FaceVertexMixedConstraint constraint(fv_candidate);

              auto found_item = fv_map.find(constraint);
              if (found_item != fv_map.end()) {
                constraint.distance = found_item->distance;
                constraint.lambda = found_item->lambda;
                fv_map.erase(constraint);
              } else {
                constraint.distance = std::sqrt(distance_sqr);
                constraint.lambda = 0.0;
              } 
              constraint.dtype = dtype;
              new_constraints.fv_constraints.emplace_back(constraint);
            }
          }
        });

    for (size_t ci = 0; ci < new_constraints.size(); ci++) {
      new_constraints[ci].minimum_distance = dmin;
    }

    constraint_set = new_constraints;
    // Check if any constraints are in ev_map, ee_map, fv_map, then add them to
    // the new state
  }

  ///////////////////////////////////////////////////////////////////////////////

}