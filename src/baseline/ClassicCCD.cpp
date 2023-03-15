#include "ClassicCCD.h"
#include "AABB.h"
#include "CTCD.h"

template<int dim> double narrowPhase(const ipc::CollisionMesh& mesh,
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    const ipc::Candidates& candidates)
{
    // unimplemented
    exit(0);
}

template<>
double narrowPhase<2>(const ipc::CollisionMesh& mesh,
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    const ipc::Candidates& candidates)
{
    double endtime = 1.0;
    for (auto &c : candidates.ev_candidates)
    {
        Eigen::Vector2i edge = mesh.edges().row(c.edge_index).transpose();
        double t;
        if (CTCD::vertexEdgeExactCTCD2D(
            V0.row(c.vertex_index).transpose(),
            V0.row(edge[0]).transpose(),
            V0.row(edge[1]).transpose(),
            V1.row(c.vertex_index).transpose(),
            V1.row(edge[0]).transpose(),
            V1.row(edge[1]).transpose(), 
            t))
        {
            endtime = std::min(endtime, t);
        }
    }
    return endtime;
}



template<int dim>
double ClassicCCD(
    const ipc::CollisionMesh& mesh,
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1)
{
    ipc::Candidates candidates;
    AABBBroadPhase<dim>(mesh, V0, V1, 0.0, candidates);

    return narrowPhase<dim>(mesh, V0, V1, candidates);
}

template double ClassicCCD<2>(const ipc::CollisionMesh& mesh, const Eigen::MatrixXd& V0, const Eigen::MatrixXd& V1);
template double ClassicCCD<3>(const ipc::CollisionMesh& mesh, const Eigen::MatrixXd& V0, const Eigen::MatrixXd& V1);