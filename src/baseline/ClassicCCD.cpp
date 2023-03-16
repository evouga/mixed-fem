#include "ClassicCCD.h"
#include "AABB.h"
#include "CTCD.h"

template<int dim> double CCDNarrowPhase(const ipc::CollisionMesh& mesh,
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    const ipc::Candidates& candidates)
{
    // unimplemented
    exit(0);
}

template<>
double CCDNarrowPhase<2>(const ipc::CollisionMesh& mesh,
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

template double CCDNarrowPhase<3>(const ipc::CollisionMesh& mesh,
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    const ipc::Candidates& candidates);
