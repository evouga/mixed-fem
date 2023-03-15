#ifndef CLASSICALCCD_H
#define CLASSICALCCD_H

#include "ipc/broad_phase/collision_candidate.hpp"
#include "ipc/collision_mesh.hpp"

template<int dim>
double ClassicalCCD(
    const ipc::CollisionMesh& mesh,
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    double inflation_radius);

#endif