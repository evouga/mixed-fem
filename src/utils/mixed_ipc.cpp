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

  void MixedConstraints::clear() {
      vv_constraints.clear();
      ev_constraints.clear();
      ee_constraints.clear();
      fv_constraints.clear();
      pv_constraints.clear();
      ev_distances.clear();
      ee_distances.clear();
      fv_distances.clear();
  }

  double& MixedConstraints::distance(size_t idx) {
    if (idx < ev_constraints.size()) {
        return ev_distances[idx];
    }
    idx -= ev_constraints.size();
    if (idx < ee_constraints.size()) {
        return ee_distances[idx];
    }
    idx -= ee_constraints.size();
    if (idx < fv_constraints.size()) {
        return fv_distances[idx];
    }
    throw std::out_of_range("Constraint index is out of range!");
  }

  const double& MixedConstraints::distance(size_t idx) const {
    if (idx < ev_constraints.size()) {
        return ev_distances[idx];
    }
    idx -= ev_constraints.size();
    if (idx < ee_constraints.size()) {
        return ee_distances[idx];
    }
    idx -= ee_constraints.size();
    if (idx < fv_constraints.size()) {
        return fv_distances[idx];
    }
    throw std::out_of_range("Constraint index is out of range!");
  }


  template <typename Hash>
  void add_vertex_vertex_constraint(
      std::vector<VertexVertexConstraint> &vv_constraints,
      unordered_map<VertexVertexConstraint, long, Hash> &vv_to_index,
      const long v0i,
      const long v1i)
  {
    VertexVertexConstraint vv_constraint(v0i, v1i);
    auto found_item = vv_to_index.find(vv_constraint);
    if (found_item != vv_to_index.end())
    {
      // Constraint already exists, so increase multiplicity
      vv_constraints[found_item->second].multiplicity++;
    }
    else
    {
      // New constraint, so add it to the end of vv_constraints
      vv_to_index.emplace(vv_constraint, vv_constraints.size());
      vv_constraints.push_back(vv_constraint);
    }
  }

  template <typename Hash>
  void add_edge_vertex_constraint(
      std::vector<EdgeVertexConstraint> &ev_constraints,
      unordered_map<EdgeVertexConstraint, long, Hash> &ev_to_index,
      const long ei,
      const long vi)
  {
    EdgeVertexConstraint ev_constraint(ei, vi);
    auto found_item = ev_to_index.find(ev_constraint);
    if (found_item != ev_to_index.end())
    {
      // Constraint already exists, so increase multiplicity
      ev_constraints[found_item->second].multiplicity++;
    }
    else
    {
      // New constraint, so add it to the end of vv_constraints
      ev_to_index.emplace(ev_constraint, ev_constraints.size());
      ev_constraints.push_back(ev_constraint);
    }
  }

  template <typename T>
  void create_constraint_map(const std::vector<T>& constraints,
      const std::vector<double>& distances, unordered_map<T,double>& map) {

    for (size_t i = 0; i < constraints.size(); ++i) {
      map.emplace(constraints[i], distances[i]);
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

    unordered_map<EdgeVertexConstraint, double> ev_map;
    unordered_map<EdgeEdgeConstraint, double> ee_map;
    unordered_map<FaceVertexConstraint, double> fv_map;
    create_constraint_map(constraint_set.ev_constraints,
        constraint_set.ev_distances, ev_map);
    create_constraint_map(constraint_set.ee_constraints,
        constraint_set.ee_distances, ee_map);
    create_constraint_map(constraint_set.fv_constraints,
        constraint_set.fv_distances, fv_map); 
    // TODO create ev,ee,fv_map from mixed constraint set

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
              EdgeVertexConstraint constraint(ev_candidate);

              new_constraints.ev_constraints.emplace_back(
                  ev_candidate);

              auto found_item = ev_map.find(constraint);
              if (found_item != ev_map.end()) {
                double dist = constraint_set.distance(found_item->second);
                ev_map.erase(constraint);
                new_constraints.ev_distances.emplace_back(dist);
              } else {
                new_constraints.ev_distances.emplace_back(std::sqrt(distance_sqr));
              }
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

              EdgeEdgeConstraint constraint(ee_candidate, eps_x);
              new_constraints.ee_constraints.emplace_back(constraint);

              auto found_item = ee_map.find(constraint);
              if (found_item != ee_map.end()) {
                double dist = constraint_set.distance(found_item->second);
                new_constraints.ee_distances.emplace_back(dist);
                ee_map.erase(constraint);

              } else {
                new_constraints.ee_distances.emplace_back(std::sqrt(distance_sqr));
              }    
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
              
              FaceVertexConstraint constraint(fv_candidate);
              new_constraints.fv_constraints.emplace_back(constraint);

              auto found_item = fv_map.find(constraint);
              if (found_item != fv_map.end()) {
                double dist = constraint_set.distance(found_item->second);
                fv_map.erase(constraint);
                new_constraints.fv_distances.emplace_back(dist);
              } else {
                new_constraints.fv_distances.emplace_back(std::sqrt(distance_sqr));
              }    
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