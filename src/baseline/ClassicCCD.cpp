#include "ClassicCCD.h"
#include "AABB.h"

template<int dim>
double ClassicalCCD(
    const ipc::CollisionMesh& mesh,
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1)
{
    ipc::Candidates candidates;
    AABBBroadPhase(mesh, V0, V1, 0.0, candidates);
}