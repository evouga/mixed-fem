#pragma once

#include <ipc/broad_phase/broad_phase.hpp>
#include "ConvexPolyhedron.h"

namespace ipc {

class BroadPhaseSpacetime : public BroadPhase {
public:

  /// @brief Build the broadphase data structure
  /// @param[in] V0 The vertex positions at time t0
  /// @param[in] V1 The vertex positions at time t1
  /// @param[in] E The edge connectivity
  /// @param[in] F The face connectivity
  /// @param[in] inflation_radius The inflation radius for the AABBs
  virtual void build(
      const Eigen::MatrixXd& V0,
      const Eigen::MatrixXd& V1,
      const Eigen::MatrixXi& E,
      const Eigen::MatrixXi& F,
      double inflation_radius = 0) override;

  /// @brief Clear any built data.
  virtual void clear() override;

  /// @brief Find the candidate edge-vertex collisisons.
  /// @param[out] candidates The candidate edge-vertex collisisons.
  virtual void detect_edge_vertex_candidates(
      std::vector<EdgeVertexCandidate>& candidates) const override;

  /// @brief Find the candidate edge-edge collisions.
  /// @param[out] candidates The candidate edge-edge collisisons.
  virtual void detect_edge_edge_candidates(
      std::vector<EdgeEdgeCandidate>& candidates) const override;

  /// @brief Find the candidate face-vertex collisions.
  /// @param[out] candidates The candidate face-vertex collisisons.
  virtual void detect_face_vertex_candidates(
      std::vector<FaceVertexCandidate>& candidates) const override;

  /// @brief Find the candidate edge-face intersections.
  /// @param[out] candidates The candidate edge-face intersections.
  virtual void detect_edge_face_candidates(
      std::vector<EdgeFaceCandidate>& candidates) const override;

protected:
    std::vector<AABB> vertex_boxes_t1;
    std::vector<AABB> edge_boxes_t1;
    std::vector<AABB> face_boxes_t1;
private:

  void make_polyhedron(const AABB& b_t0, const AABB& b_t1,
      ConvexPolyhedron<double>& p) const;

  template <typename Candidate>
  void detect_candidates(
      const std::vector<AABB>& boxes0_t0,
      const std::vector<AABB>& boxes0_t1,
      const std::vector<AABB>& boxes1_t0,
      const std::vector<AABB>& boxes1_t1,
      const std::function<bool(size_t, size_t)>& can_collide,
      std::vector<Candidate>& candidates) const;
};

}
