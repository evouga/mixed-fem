#ifndef CLASSICALCCD_H
#define CLASSICALCCD_H

#include "ipc/broad_phase/collision_candidate.hpp"
#include "ipc/collision_mesh.hpp"

template<int dim> double CCDNarrowPhase(const ipc::CollisionMesh& mesh,
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    const ipc::Candidates& candidates);

#endif