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

template <typename Hash>
void add_vertex_vertex_constraint(
    std::vector<VertexVertexConstraint>& vv_constraints,
    unordered_map<VertexVertexConstraint, long, Hash>& vv_to_index,
    const long v0i,
    const long v1i)
{
    VertexVertexConstraint vv_constraint(v0i, v1i);
    auto found_item = vv_to_index.find(vv_constraint);
    if (found_item != vv_to_index.end()) {
        // Constraint already exists, so increase multiplicity
        vv_constraints[found_item->second].multiplicity++;
    } else {
        // New constraint, so add it to the end of vv_constraints
        vv_to_index.emplace(vv_constraint, vv_constraints.size());
        vv_constraints.push_back(vv_constraint);
    }
}

template <typename Hash>
void add_edge_vertex_constraint(
    std::vector<EdgeVertexConstraint>& ev_constraints,
    unordered_map<EdgeVertexConstraint, long, Hash>& ev_to_index,
    const long ei,
    const long vi)
{
    EdgeVertexConstraint ev_constraint(ei, vi);
    auto found_item = ev_to_index.find(ev_constraint);
    if (found_item != ev_to_index.end()) {
        // Constraint already exists, so increase multiplicity
        ev_constraints[found_item->second].multiplicity++;
    } else {
        // New constraint, so add it to the end of vv_constraints
        ev_to_index.emplace(ev_constraint, ev_constraints.size());
        ev_constraints.push_back(ev_constraint);
    }
}


void construct_constraint_set(
    const Candidates& candidates,
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& V,
    const double dhat,
    Constraints& constraint_set,
    MixedState& state,
    const double dmin)
{
    assert(V.rows() == mesh.num_vertices());

    constraint_set.clear();

    const Eigen::MatrixXd& V_rest = mesh.vertices_at_rest();
    const Eigen::MatrixXi& E = mesh.edges();
    const Eigen::MatrixXi& F = mesh.faces();
    const Eigen::MatrixXi& F2E = mesh.faces_to_edges();

    // Cull the candidates by measuring the distance and dropping those that are
    // greater than dhat.
    const double offset_sqr = (dmin + dhat) * (dmin + dhat);
    auto is_active = [&](double distance_sqr) {
        return distance_sqr < offset_sqr;
    };

    // Store the indices to VV and EV pairs to avoid duplicates.
    unordered_map<VertexVertexConstraint, long> vv_to_index;
    unordered_map<EdgeVertexConstraint, long> ev_to_index;

    std::mutex vv_mutex, ev_mutex, ee_mutex, fv_mutex;

    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), candidates.ev_candidates.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                const auto& ev_candidate = candidates.ev_candidates[i];
                const auto& [ei, vi] = ev_candidate;
                long e0i = E(ei, 0), e1i = E(ei, 1);

                PointEdgeDistanceType dtype =
                    point_edge_distance_type(V.row(vi), V.row(e0i), V.row(e1i));

                double distance_sqr = point_edge_distance(
                    V.row(vi), V.row(e0i), V.row(e1i), dtype,
                    DistanceMode::SQUARED);

                if (is_active(distance_sqr)) {
                    switch (dtype) {
                    case PointEdgeDistanceType::P_E0: {
                        std::lock_guard<std::mutex> lock(vv_mutex);
                        add_vertex_vertex_constraint(
                            constraint_set.vv_constraints, vv_to_index, vi,
                            e0i);
                    } break;

                    case PointEdgeDistanceType::P_E1: {
                        std::lock_guard<std::mutex> lock(vv_mutex);
                        add_vertex_vertex_constraint(
                            constraint_set.vv_constraints, vv_to_index, vi,
                            e1i);
                    } break;

                    case PointEdgeDistanceType::P_E: {
                        std::lock_guard<std::mutex> lock(ev_mutex);
                        // ev_candidates is a set, so no duplicate EV
                        // constraints
                        constraint_set.ev_constraints.emplace_back(
                            ev_candidate);
                        ev_to_index.emplace(
                            constraint_set.ev_constraints.back(),
                            constraint_set.ev_constraints.size() - 1);
                    } break;
                    }
                }
            }
        });

    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), candidates.ee_candidates.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                const auto& ee_candidate = candidates.ee_candidates[i];
                const auto& [eai, ebi] = ee_candidate;
                long ea0i = E(eai, 0), ea1i = E(eai, 1);
                long eb0i = E(ebi, 0), eb1i = E(ebi, 1);

                EdgeEdgeDistanceType dtype = edge_edge_distance_type(
                    V.row(ea0i), V.row(ea1i), V.row(eb0i), V.row(eb1i));

                double distance_sqr = edge_edge_distance(
                    V.row(ea0i), V.row(ea1i), V.row(eb0i), V.row(eb1i), dtype,
                    DistanceMode::SQUARED);

                if (is_active(distance_sqr)) {
                    double eps_x = edge_edge_mollifier_threshold(
                        V_rest.row(ea0i), V_rest.row(ea1i), //
                        V_rest.row(eb0i), V_rest.row(eb1i));
                    double ee_cross_norm_sqr = edge_edge_cross_squarednorm(
                        V.row(ea0i), V.row(ea1i), V.row(eb0i), V.row(eb1i));
                    if (ee_cross_norm_sqr < eps_x) {
                        // NOTE: This may not actually be the distance type, but
                        // all EE pairs requiring mollification must be
                        // mollified later.
                        dtype = EdgeEdgeDistanceType::EA_EB;
                    }

                    switch (dtype) {
                    case EdgeEdgeDistanceType::EA0_EB0: {
                        std::lock_guard<std::mutex> lock(vv_mutex);
                        add_vertex_vertex_constraint(
                            constraint_set.vv_constraints, vv_to_index, ea0i,
                            eb0i);
                    } break;

                    case EdgeEdgeDistanceType::EA0_EB1: {
                        std::lock_guard<std::mutex> lock(vv_mutex);
                        add_vertex_vertex_constraint(
                            constraint_set.vv_constraints, vv_to_index, ea0i,
                            eb1i);
                    } break;

                    case EdgeEdgeDistanceType::EA1_EB0: {
                        std::lock_guard<std::mutex> lock(vv_mutex);
                        add_vertex_vertex_constraint(
                            constraint_set.vv_constraints, vv_to_index, ea1i,
                            eb0i);
                    } break;

                    case EdgeEdgeDistanceType::EA1_EB1: {
                        std::lock_guard<std::mutex> lock(vv_mutex);
                        add_vertex_vertex_constraint(
                            constraint_set.vv_constraints, vv_to_index, ea1i,
                            eb1i);
                    } break;

                    case EdgeEdgeDistanceType::EA_EB0: {
                        std::lock_guard<std::mutex> lock(ev_mutex);
                        add_edge_vertex_constraint(
                            constraint_set.ev_constraints, ev_to_index, eai,
                            eb0i);
                    } break;

                    case EdgeEdgeDistanceType::EA_EB1: {
                        std::lock_guard<std::mutex> lock(ev_mutex);
                        add_edge_vertex_constraint(
                            constraint_set.ev_constraints, ev_to_index, eai,
                            eb1i);
                    } break;

                    case EdgeEdgeDistanceType::EA0_EB: {
                        std::lock_guard<std::mutex> lock(ev_mutex);
                        add_edge_vertex_constraint(
                            constraint_set.ev_constraints, ev_to_index, ebi,
                            ea0i);
                    } break;

                    case EdgeEdgeDistanceType::EA1_EB: {
                        std::lock_guard<std::mutex> lock(ev_mutex);
                        add_edge_vertex_constraint(
                            constraint_set.ev_constraints, ev_to_index, ebi,
                            ea1i);
                    } break;

                    case EdgeEdgeDistanceType::EA_EB: {
                        std::lock_guard<std::mutex> lock(ee_mutex);
                        constraint_set.ee_constraints.emplace_back(
                            ee_candidate, eps_x);
                    } break;
                    }
                }
            }
        });

    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), candidates.fv_candidates.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                const auto& fv_candidate = candidates.fv_candidates[i];
                const auto& [fi, vi] = fv_candidate;
                long f0i = F(fi, 0), f1i = F(fi, 1), f2i = F(fi, 2);

                // Compute distance type
                PointTriangleDistanceType dtype = point_triangle_distance_type(
                    V.row(fv_candidate.vertex_index), //
                    V.row(f0i), V.row(f1i), V.row(f2i));

                double distance_sqr = point_triangle_distance(
                    V.row(fv_candidate.vertex_index), //
                    V.row(f0i), V.row(f1i), V.row(f2i), dtype,
                    DistanceMode::SQUARED);

                if (is_active(distance_sqr)) {
                    switch (dtype) {
                    case PointTriangleDistanceType::P_T0: {
                        std::lock_guard<std::mutex> lock(vv_mutex);
                        add_vertex_vertex_constraint(
                            constraint_set.vv_constraints, vv_to_index, vi,
                            f0i);
                    } break;

                    case PointTriangleDistanceType::P_T1: {
                        std::lock_guard<std::mutex> lock(vv_mutex);
                        add_vertex_vertex_constraint(
                            constraint_set.vv_constraints, vv_to_index, vi,
                            f1i);
                    } break;

                    case PointTriangleDistanceType::P_T2: {
                        std::lock_guard<std::mutex> lock(vv_mutex);
                        add_vertex_vertex_constraint(
                            constraint_set.vv_constraints, vv_to_index, vi,
                            f2i);
                    } break;

                    case PointTriangleDistanceType::P_E0: {
                        std::lock_guard<std::mutex> lock(ev_mutex);
                        add_edge_vertex_constraint(
                            constraint_set.ev_constraints, ev_to_index,
                            F2E(fi, 0), vi);
                    } break;

                    case PointTriangleDistanceType::P_E1: {
                        std::lock_guard<std::mutex> lock(ev_mutex);
                        add_edge_vertex_constraint(
                            constraint_set.ev_constraints, ev_to_index,
                            F2E(fi, 1), vi);
                    } break;

                    case PointTriangleDistanceType::P_E2: {
                        std::lock_guard<std::mutex> lock(ev_mutex);
                        add_edge_vertex_constraint(
                            constraint_set.ev_constraints, ev_to_index,
                            F2E(fi, 2), vi);
                    } break;

                    case PointTriangleDistanceType::P_T: {
                        std::lock_guard<std::mutex> lock(fv_mutex);
                        constraint_set.fv_constraints.emplace_back(
                            fv_candidate);
                    } break;
                    }
                }
            }
        });

    for (size_t ci = 0; ci < constraint_set.size(); ci++) {
        constraint_set[ci].minimum_distance = dmin;
    }
}

///////////////////////////////////////////////////////////////////////////////

}