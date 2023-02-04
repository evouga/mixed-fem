#include "mixed_ipc.h"

// #include <ipc/barrier/barrier.hpp>

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

  size_t MixedConstraints::size() const {
      return ev_constraints.size() + ee_constraints.size()
          + fv_constraints.size();
  }

  bool MixedConstraints::empty() const {
      return ev_constraints.empty()
          && ee_constraints.empty() && fv_constraints.empty();
  }

  void MixedConstraints::clear() {
    ev_constraints.clear();
    ee_constraints.clear();
    fv_constraints.clear();
  }

  CollisionConstraint& MixedConstraints::operator[](size_t idx) {
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

  const CollisionConstraint& MixedConstraints::operator[](size_t idx) const {
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

  bool MixedConstraints::constraint_mollifier(size_t idx,
      const Eigen::MatrixXd& V, const Eigen::MatrixXi& E, 
      double& value) const {
    // Mollifier can only be active if constraint is edge-edge
    value = 1.0;
    idx -= ev_constraints.size();
    if (idx < ee_constraints.size()) {
      // Get edges
      long eai = ee_constraints[idx].edge0_index;
      long ebi = ee_constraints[idx].edge1_index;
      double eps_x = ee_constraints[idx].eps_x;

      // Edge vertex indices
      long ea0i = E(eai, 0), ea1i = E(eai, 1);
      long eb0i = E(ebi, 0), eb1i = E(ebi, 1);

      value = edge_edge_mollifier(
          V.row(ea0i), V.row(ea1i), V.row(eb0i), V.row(eb1i), eps_x);
      
      // Return true if mollifier is < 1.0, indicating that the mollifier is on
      return (value < 1.0);
    } else {
      return false;
    }
  }

  VectorMax12d MixedConstraints::constraint_mollifier_gradient(size_t idx,
            const Eigen::MatrixXd& V, const Eigen::MatrixXi& E) const {
    // Mollifier can only be active if constraint is edge-edge
    idx -= ev_constraints.size();
    if (idx < ee_constraints.size()) {
      // Get edges
      long eai = ee_constraints[idx].edge0_index;
      long ebi = ee_constraints[idx].edge1_index;
      double eps_x = ee_constraints[idx].eps_x;

      // Edge vertex indices
      long ea0i = E(eai, 0), ea1i = E(eai, 1);
      long eb0i = E(ebi, 0), eb1i = E(ebi, 1);

      VectorMax12d mollifier_grad;
      edge_edge_mollifier_gradient(
          V.row(ea0i), V.row(ea1i), V.row(eb0i), V.row(eb1i), eps_x,
          mollifier_grad);
      return mollifier_grad;
    }
    throw std::out_of_range("index is not for edge-edge constraint");
  }


  void MixedConstraints::update_distances(const Eigen::VectorXd& distances) {
    assert(distances.size() == size());
    for (int i = 0; i < distances.size(); ++i) {
      distance(i) = distances(i);
    }
  }
  
  void MixedConstraints::update_lambdas(const Eigen::VectorXd& lambdas) {
    assert(lambdas.size() == size());
    for (int i = 0; i < lambdas.size(); ++i) {
      lambda(i) = lambdas(i);
    }
  }
  
  template <typename T>
  void create_constraint_set(const std::vector<T>& constraints,
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
      const double dmin) {
    assert(V.rows() == mesh.num_vertices());

    MixedConstraints new_constraints;

    const Eigen::MatrixXd &V_rest = mesh.vertices_at_rest();
    const Eigen::MatrixXi &E = mesh.edges();
    const Eigen::MatrixXi &F = mesh.faces();
    const Eigen::MatrixXi &F2E = mesh.faces_to_edges();

    // Cull the candidates by measuring the distance and dropping those that are
    // greater than dhat.
    const double offset_sqr = (dmin + dhat) * (dmin + dhat);
    auto is_active = [&](double distance_sqr) {
      return distance_sqr < offset_sqr;
    };

    unordered_set<EdgeVertexMixedConstraint> ev_set;
    unordered_set<EdgeEdgeMixedConstraint> ee_set;
    unordered_set<FaceVertexMixedConstraint> fv_set;
    create_constraint_set(constraint_set.ev_constraints, ev_set);
    create_constraint_set(constraint_set.ee_constraints, ee_set);
    create_constraint_set(constraint_set.fv_constraints, fv_set); 

    std::mutex vv_mutex, ev_mutex, ee_mutex, fv_mutex;

    // Process edge-vertex constraint candidates
    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), candidates.ev_candidates.size()),
        [&](const tbb::blocked_range<size_t> &r) {
          for (size_t i = r.begin(); i < r.end(); i++) {
            // Construct edge-vertex constraint candidate
            const auto &ev_candidate = candidates.ev_candidates[i];
            const auto &[ei, vi] = ev_candidate;
            long e0i = E(ei, 0), e1i = E(ei, 1);

            PointEdgeDistanceType dtype =
                point_edge_distance_type(V.row(vi), V.row(e0i), V.row(e1i));

            // Squared distance from to point to edge
            double distance_sqr = point_edge_distance(
                V.row(vi), V.row(e0i), V.row(e1i), dtype,
                DistanceMode::SQUARED);

            if (is_active(distance_sqr)) {
              std::lock_guard<std::mutex> lock(ev_mutex);
              // ev_candidates is a set, so no duplicate EV
              // constraints
              EdgeVertexMixedConstraint constraint(ev_candidate);

              // Check if this constraint exists from a previous iteration
              auto found_item = ev_set.find(constraint);
              if (found_item != ev_set.end()) {
                // If this already exists, then it has a mixed distance
                // and lagrange multiplier value, so we  re-use hese values.
                constraint.distance = found_item->distance;
                constraint.lambda = found_item->lambda;
                ev_set.erase(constraint);
              } else {
                // Otherwise initialize the mixed distance
                // to the true distance and set lambda to 0.
                // constraint.distance = distance_sqr;
                constraint.distance = std::sqrt(distance_sqr);
                constraint.lambda = 0.0;
              }
              constraint.dtype = dtype;
              new_constraints.ev_constraints.emplace_back(constraint);
            }
          }
        });
    // Process edge-edge constraint candidates
    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), candidates.ee_candidates.size()),
        [&](const tbb::blocked_range<size_t> &r) {
          for (size_t i = r.begin(); i < r.end(); i++) {
            // Construct edge-edge constraint candidate
            const auto &ee_candidate = candidates.ee_candidates[i];
            const auto &[eai, ebi] = ee_candidate;
            long ea0i = E(eai, 0), ea1i = E(eai, 1);
            long eb0i = E(ebi, 0), eb1i = E(ebi, 1);

            EdgeEdgeDistanceType dtype = edge_edge_distance_type(
                V.row(ea0i), V.row(ea1i), V.row(eb0i), V.row(eb1i));

            // Squared distance between edges
            double distance_sqr = edge_edge_distance(
                V.row(ea0i), V.row(ea1i), V.row(eb0i), V.row(eb1i), dtype,
                DistanceMode::SQUARED);

            if (is_active(distance_sqr)) {
              double eps_x = edge_edge_mollifier_threshold(
                  V_rest.row(ea0i), V_rest.row(ea1i), //
                  V_rest.row(eb0i), V_rest.row(eb1i));
              std::lock_guard<std::mutex> lock(ee_mutex);

              EdgeEdgeMixedConstraint constraint(ee_candidate, eps_x);

              // Again, check if this constraint already exists. If so,
              // reuse the values, otherwise initialize them.
              auto found_item = ee_set.find(constraint);
              if (found_item != ee_set.end()) {
                constraint.distance = found_item->distance;
                constraint.lambda = found_item->lambda;
                ee_set.erase(constraint);
              } else {
                constraint.distance = std::sqrt(distance_sqr);
                constraint.lambda = 0.0;
              }
              new_constraints.ee_constraints.emplace_back(constraint);
            }
          }
        });

    // Process face-vertex constraint candidates
    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), candidates.fv_candidates.size()),
        [&](const tbb::blocked_range<size_t> &r) {
          for (size_t i = r.begin(); i < r.end(); i++) {
            // Construct face-vertex constraint candidate
            const auto &fv_candidate = candidates.fv_candidates[i];
            const auto &[fi, vi] = fv_candidate;
            long f0i = F(fi, 0), f1i = F(fi, 1), f2i = F(fi, 2);

            // Compute distance type
            PointTriangleDistanceType dtype = point_triangle_distance_type(
                V.row(fv_candidate.vertex_index), //
                V.row(f0i), V.row(f1i), V.row(f2i));

            // Squared point-triangle fact distance
            double distance_sqr = point_triangle_distance(
                V.row(fv_candidate.vertex_index), //
                V.row(f0i), V.row(f1i), V.row(f2i), dtype,
                DistanceMode::SQUARED);

            if (is_active(distance_sqr)) {
              std::lock_guard<std::mutex> lock(fv_mutex);
              
              FaceVertexMixedConstraint constraint(fv_candidate);

              // Again, check if this constraint already exists. If so,
              // reuse the values, otherwise initialize them.
              auto found_item = fv_set.find(constraint);
              if (found_item != fv_set.end()) {
                constraint.distance = found_item->distance;
                constraint.lambda = found_item->lambda;
                fv_set.erase(constraint);
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

    // if ((ee_set.size() + ev_set.size() + fv_set.size()) > 0) {
    //   std::cout << "WARNING: " << ee_set.size() << " edge-edge, " << ev_set.size()
    //             << " edge-vertex, and " << fv_set.size()
    //             << " face-vertex constraints were not removed." << std::endl;
    // }

    // For each remaining element in sets, add elements into new_constraints
    // IF the mixed distance is below the dhat value. This happens when the
    // true distance is greater than dhat, but the mixed still hasn't caught
    // up. In such a case, we need to NOT remove these constraints, and add them
    // to the new constraint set.
    //
    // NOTE: adding this in MURDERS convergence.
    //
    // for (const auto& constraint : ee_set) {
    //   if (constraint.distance < dhat) {
    //     new_constraints.ee_constraints.emplace_back(constraint);
    //   }
    // }
    // for (const auto& constraint : ev_set) {
    //   if (constraint.distance < dhat) {
    //     new_constraints.ev_constraints.emplace_back(constraint);
    //   }
    // }
    // for (const auto& constraint : fv_set) {
    //   if (constraint.distance < dhat) {
    //     new_constraints.fv_constraints.emplace_back(constraint);
    //   }
    // }
    constraint_set = new_constraints;
  }
}