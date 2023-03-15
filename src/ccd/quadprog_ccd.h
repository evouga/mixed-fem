#pragma once

#include "ipc/broad_phase/collision_candidate.hpp"
#include "ipc/collision_mesh.hpp"
#include <ipc/broad_phase/broad_phase.hpp>

namespace mfem {

// NOTE Feel free to replace with your own 
// BVH Node struct, I'm just implementing this
// with IPC AABBs right now since I'm familiar with them
struct SpacetimeAABB {
  ipc::AABB aabb0; // aabb at t=0
  ipc::AABB aabb1; // aabb at t=1
};

template<int DIM>
bool evaluate_certificate(const SpacetimeAABB& b0,
    const SpacetimeAABB& b1);

template<int DIM>
double quadprog_ccd(
    const ipc::CollisionMesh& mesh,
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1);

}
