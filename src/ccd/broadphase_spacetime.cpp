#include "broadphase_spacetime.h"
#include <iostream>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <ipc/utils/merge_thread_local_vectors.hpp>

namespace ipc {

void BroadPhaseSpacetime::make_polyhedron(const AABB& b_t0,
    const AABB& b_t1, ConvexPolyhedron<double>& p) const {
  const ArrayMax3d& b0_min_t0 = b_t0.min;
  const ArrayMax3d& b0_max_t0 = b_t0.max;
  const ArrayMax3d& b1_min_t1 = b_t1.min;
  const ArrayMax3d& b1_max_t1 = b_t1.max;

  using V3Array = typename ConvexPolyhedron<double>::V3Array;
  using IArray = typename ConvexPolyhedron<double>::IArray;

  // NOTE Assuming 2D right now
  // The resulting polyhedron will have 8 vertices, 12 triangles, and
  // will be swept in space-time from t0 to t1
  V3Array points(8);
  // t0 points
  points[0] = { b0_min_t0[0], b0_min_t0[1], 0 };
  points[1] = { b0_max_t0[0], b0_min_t0[1], 0 };
  points[2] = { b0_max_t0[0], b0_max_t0[1], 0 };
  points[3] = { b0_min_t0[0], b0_max_t0[1], 0 };
  // t1 points
  points[4] = { b1_min_t1[0], b1_min_t1[1], 1 };
  points[5] = { b1_max_t1[0], b1_min_t1[1], 1 };
  points[6] = { b1_max_t1[0], b1_max_t1[1], 1 };
  points[7] = { b1_min_t1[0], b1_max_t1[1], 1 };
  
  // Form polyhedron faces
  IArray indices(36);
  indices[0] = 0; indices[1] = 1; indices[2] = 5;
  indices[3] = 0; indices[4] = 5; indices[5] = 4;
  indices[6] = 1; indices[7] = 2; indices[8] = 6;
  indices[9] = 1; indices[10] = 6; indices[11] = 5;
  indices[12] = 2; indices[13] = 3; indices[14] = 7;
  indices[15] = 2; indices[16] = 7; indices[17] = 6;
  indices[18] = 3; indices[19] = 0; indices[20] = 4;
  indices[21] = 3; indices[22] = 4; indices[23] = 7;
  indices[24] = 4; indices[25] = 5; indices[26] = 6;
  indices[27] = 4; indices[28] = 6; indices[29] = 7;
  indices[30] = 3; indices[31] = 2; indices[32] = 1;
  indices[33] = 3; indices[34] = 1; indices[35] = 0;

  //p.SetInitialELabel(0);
  p.Create(points, indices);

  //if (p.ValidateHalfSpaceProperty(-1e-8)) {
  //} else {
  //    std::cout << "polyhedron aint manifold" << std::endl;
  //    std::cout << "b0 min: " << b0_min_t0[0] << ", " << b0_min_t0[1] << std::endl;
  //    std::cout << "b0 max: " << b0_max_t0[0] << ", " << b0_max_t0[1] << std::endl;
  //    std::cout << "b1 min: " << b1_min_t1[0] << ", " << b1_min_t1[1] << std::endl;
  //    std::cout << "b1 max: " << b1_max_t1[0] << ", " << b1_max_t1[1] << std::endl;
  //    p.Print("polyhedron.txt");
  //    exit(1);
  //}
}

template <typename Candidate>
void BroadPhaseSpacetime::detect_candidates(
    const std::vector<AABB>& boxes0_t0,
    const std::vector<AABB>& boxes0_t1,
    const std::vector<AABB>& boxes1_t0,
    const std::vector<AABB>& boxes1_t1,
    const std::function<bool(size_t, size_t)>& can_collide,
    std::vector<Candidate>& candidates) const
{
  for (size_t i = 0; i < boxes0_t0.size(); i++) {
    const AABB& box0_t0 = boxes0_t0[i];
    const AABB& box0_t1 = boxes0_t1[i];
    
    // Form ConvexPolyhedron from box0_t0 and box0_t1
    ConvexPolyhedron<double> b0;
    make_polyhedron(box0_t0, box0_t1, b0);
    
    for (size_t j = 0; j < boxes1_t0.size(); j++) {
      if (!can_collide(i, j)) {
        continue;
      }
      const AABB& box1_t0 = boxes1_t0[j];
      const AABB& box1_t1 = boxes1_t1[j];

      // Spacetime AABB polyhedron
      ConvexPolyhedron<double> b1;
      make_polyhedron(box1_t0, box1_t1, b1);
      
      // Check intersection
      if (b0.HasIntersection(b1)) {
        candidates.emplace_back(i, j);
      }
    }
  }
}

void BroadPhaseSpacetime::build(
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    const Eigen::MatrixXi& E,
    const Eigen::MatrixXi& F,
    double inflation_radius)
{
  assert(E.size() == 0 || E.cols() == 2);
  assert(F.size() == 0 || F.cols() == 3);
  clear();

  // Building spatial bounding boxes for t0 vertex set
  build_vertex_boxes(V0, vertex_boxes, inflation_radius);
  build_edge_boxes(vertex_boxes, E, edge_boxes);
  build_face_boxes(vertex_boxes, F, face_boxes);

  // Building spatial bounding boxes for t1 vertex set
  build_vertex_boxes(V1, vertex_boxes_t1, inflation_radius);
  build_edge_boxes(vertex_boxes_t1, E, edge_boxes_t1);
  build_face_boxes(vertex_boxes_t1, F, face_boxes_t1);
}

void BroadPhaseSpacetime::clear()
{
    vertex_boxes.clear();
    edge_boxes.clear();
    face_boxes.clear();
    vertex_boxes_t1.clear();
    edge_boxes_t1.clear();
    face_boxes_t1.clear();
}

void BroadPhaseSpacetime::detect_edge_vertex_candidates(
    std::vector<EdgeVertexCandidate>& candidates) const
{
  detect_candidates(
      edge_boxes, edge_boxes_t1, vertex_boxes, vertex_boxes_t1,
      [&](size_t ei, size_t vi) { return can_edge_vertex_collide(ei, vi); },
      candidates);
}

void BroadPhaseSpacetime::detect_edge_edge_candidates(
    std::vector<EdgeEdgeCandidate>& candidates) const
{
  std::cout << "ee broadphase_spacetime not implemented" << std::endl;
}

void BroadPhaseSpacetime::detect_face_vertex_candidates(
    std::vector<FaceVertexCandidate>& candidates) const
{
  std::cout << "fv broadphase_spacetime not implemented" << std::endl;
}

void BroadPhaseSpacetime::detect_edge_face_candidates(
    std::vector<EdgeFaceCandidate>& candidates) const
{
  std::cout << "ef broadphase_spacetime not implemented" << std::endl;
}

}
