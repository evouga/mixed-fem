#ifndef AABB_H
#define AABB_H

#include "ipc/broad_phase/collision_candidate.hpp"
#include "ipc/collision_mesh.hpp"

template<int dim>
void AABBBroadPhase(
    const ipc::CollisionMesh& mesh,
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    double inflation_radius,
    ipc::Candidates& collisionCandidates);

#endif