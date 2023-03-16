#include "AABB.h"
#include <vector>
#include <algorithm>

template<int dim> struct BBox
{
    double mins[dim];
    double maxs[dim];

    void unionWith(BBox<dim>& other)
    {
        for (int i = 0; i < dim; i++)
        {
            mins[i] = std::min(mins[i], other.mins[i]);
            maxs[i] = std::max(maxs[i], other.maxs[i]);
        }
    }

    Eigen::Matrix<double, 1, dim> centroid() const
    {
        Eigen::Matrix<double, 1, dim> ret;
        for (int i = 0; i < dim; i++)
            ret[i] = 0.5 * (mins[i] + maxs[i]);
        return ret;
    }    
};

template<int dim> bool intersects(BBox<dim> b1, BBox<dim> b2)
{
    for (int i = 0; i < dim; i++)
    {
        if (b1.maxs[i] < b2.mins[i])
            return false;
        if (b1.mins[i] > b2.maxs[i])
            return false;
    }
    return true;
}

enum LeafType {
    LT_NONE,
    LT_VERTEX,
    LT_EDGE,
    LT_FACE
};


template<int dim>
struct AABB
{
    AABB(LeafType type, int id, BBox<dim> boundingBox) : leafType(type), leafId(id), boundingBox(boundingBox)
    {
        left = right = NULL;       
    }

    AABB(AABB* left, AABB* right, BBox<dim> boundingBox) : left(left), right(right), boundingBox(boundingBox)
    {
        leafType = LT_NONE;
        leafId = -1;
    }

    AABB* left, * right;
    BBox<dim> boundingBox;
    LeafType leafType;
    int leafId;
    ~AABB()
    {
        delete left;
        delete right;
    }
};

template<int dim>
AABB<dim> *buildAABB(std::vector<AABB<dim>*>& leaves, int start, int end)
{
    if (end - start == 1)
        return leaves[start];
    BBox<dim> bbox;
    for (int i = 0; i < dim; i++)
    {
        bbox.mins[i] = std::numeric_limits<double>::infinity();
        bbox.maxs[i] = -std::numeric_limits<double>::infinity();
    }

    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            bbox.mins[j] = std::min(bbox.mins[j], leaves[i]->boundingBox.mins[j]);
            bbox.maxs[j] = std::max(bbox.maxs[j], leaves[i]->boundingBox.maxs[j]);
        }
    }

    double bestextent = -std::numeric_limits<double>::infinity();
    int bestaxis = -1;
    for (int i = 0; i < dim; i++)
    {
        double extent = bbox.maxs[i] - bbox.mins[i];
        if (extent > bestextent)
        {
            bestextent = extent;
            bestaxis = i;
        }
    }

    std::sort(leaves.begin() + start, leaves.begin() + end, [bestaxis](const AABB<dim>* node1, const AABB<dim>* node2) -> bool
        {
            return node1->boundingBox.centroid()[bestaxis] < node2->boundingBox.centroid()[bestaxis];
        });
    int mid = start + (end - start) / 2;
    AABB<dim> * left = buildAABB(leaves, start, mid);
    AABB<dim> * right = buildAABB(leaves, mid, end);
    return new AABB<dim>(left, right, bbox);
}

template<int dim>
void resolveLeafLeaf(const ipc::CollisionMesh& mesh, LeafType type1, int id1, LeafType type2, int id2, ipc::Candidates& collisionCandidates)
{
    if (type2 < type1)
    {
        std::swap(type1, type2);
        std::swap(id1, id2);
    }
    if (type1 == LT_VERTEX)
    {
        if (type2 == LT_EDGE)
        {
            Eigen::Vector2i edge = mesh.edges().row(id2);
            if (edge[0] != id1 && edge[1] != id1)
            {
                collisionCandidates.ev_candidates.push_back({ id2, id1 });
            }
        }
        else if (dim > 2 && type2 == LT_FACE)
        {
            Eigen::Vector3i face = mesh.faces().row(id2);
            if (face[0] != id1 && face[1] != id1 && face[2] != id1)
            {
                collisionCandidates.fv_candidates.push_back({ id2, id1 });
            }
        }
    }
    else if (type1 == LT_EDGE)
    {
        if (dim > 2 && type2 == LT_EDGE)
        {
            Eigen::Vector2i edge1 = mesh.edges().row(id1);
            Eigen::Vector2i edge2 = mesh.edges().row(id2);
            if (edge1[0] != edge2[0] && edge1[0] != edge2[1] && edge1[1] != edge2[0] && edge1[1] != edge2[1])
            {
                collisionCandidates.ee_candidates.push_back({ id1, id2 });
            }
        }
    }
}

template<int dim>
void intersect(const AABB<dim>* node1, const AABB<dim>* node2, const ipc::CollisionMesh& mesh, ipc::Candidates& collisionCandidates)
{
    if (!intersects(node1->boundingBox, node2->boundingBox))
        return;

    if (node1->leafType != LT_NONE)
    {
        if (node2->leafType != LT_NONE)
        {
            resolveLeafLeaf<dim>(mesh, node1->leafType, node1->leafId, node2->leafType, node2->leafId, collisionCandidates);
        }
        else
        {
            intersect(node1, node2->left, mesh, collisionCandidates);
            intersect(node1, node2->right, mesh, collisionCandidates);
        }
    }
    else
    {
        intersect(node1->left, node2, mesh, collisionCandidates);
        intersect(node1->right, node2, mesh, collisionCandidates);
    }
}

template<int dim>
void TrivialBroadPhase(
    const ipc::CollisionMesh& mesh,
    double inflation_radius,
    ipc::Candidates& collisionCandidates)
{
    for (int i = 0; i < mesh.num_vertices(); i++)
    {
        if (dim == 2)
        {
            for (int j = 0; j < mesh.edges().size(); j++)
            {
                resolveLeafLeaf<dim>(mesh, LT_VERTEX, i, LT_EDGE, j, collisionCandidates);
            }
        }
        else if (dim == 3)
        {
            for (int j = 0; j < mesh.faces().size(); j++)
            {
                resolveLeafLeaf<dim>(mesh, LT_VERTEX, i, LT_FACE, j, collisionCandidates);
            }
        }
    }
    if (dim == 3)
    {
        for (int i = 0; i < mesh.edges().size(); i++)
        {
            for (int j = i + 1; j < mesh.edges().size(); j++)
            {
                resolveLeafLeaf<dim>(mesh, LT_EDGE, i, LT_EDGE, j, collisionCandidates);
            }
        }
    }
}


template<int dim>
void AABBBroadPhase(
    const ipc::CollisionMesh& mesh,
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    double inflation_radius,
    ipc::Candidates& collisionCandidates)
{
    std::vector<AABB<dim> *> leaves;
    // vertices
    for (int i = 0; i < mesh.num_vertices(); i++)
    {
        BBox<dim> bbox;
        for (int j = 0; j < dim; j++)
        {
            bbox.mins[j] = std::numeric_limits<double>::max();
            bbox.maxs[j] = std::numeric_limits<double>::min();
        }
        for (int j = 0; j < dim; j++)
        {
            bbox.mins[j] = std::min(bbox.mins[j], V0(i, j) - inflation_radius);
            bbox.mins[j] = std::min(bbox.mins[j], V1(i, j) - inflation_radius);
            bbox.maxs[j] = std::max(bbox.maxs[j], V0(i, j) + inflation_radius);
            bbox.maxs[j] = std::max(bbox.maxs[j], V1(i, j) + inflation_radius);
        }
        AABB<dim>* leaf = new AABB(LT_VERTEX, i, bbox);
        leaves.push_back(leaf);
    }
    // edges
    for (int i = 0; i < mesh.edges().rows(); i++)
    {
        BBox<dim> bbox;
        for (int j = 0; j < dim; j++)
        {
            bbox.mins[j] = std::numeric_limits<double>::max();
            bbox.maxs[j] = std::numeric_limits<double>::min();
        }
        Eigen::Vector2i verts = mesh.edges().row(i);
        for (int k = 0; k < 2; k++)
        {
            int vert = verts[k];
            for (int j = 0; j < dim; j++)
            {
                bbox.mins[j] = std::min(bbox.mins[j], V0(vert, j) - inflation_radius);
                bbox.mins[j] = std::min(bbox.mins[j], V1(vert, j) - inflation_radius);
                bbox.maxs[j] = std::max(bbox.maxs[j], V0(vert, j) + inflation_radius);
                bbox.maxs[j] = std::max(bbox.maxs[j], V1(vert, j) + inflation_radius);
            }
        }
        AABB<dim>* leaf = new AABB(LT_EDGE, i, bbox);
        leaves.push_back(leaf);
    }
    // triangles
    if (dim > 2)
    {
        for (int i = 0; i < mesh.faces().rows(); i++)
        {
            BBox<dim> bbox;
            for (int j = 0; j < dim; j++)
            {
                bbox.mins[j] = std::numeric_limits<double>::max();
                bbox.maxs[j] = std::numeric_limits<double>::min();
            }
            Eigen::Vector3i verts = mesh.faces().row(i);
            for (int k = 0; k < 3; k++)
            {
                int vert = verts[k];
                for (int j = 0; j < dim; j++)
                {
                    bbox.mins[j] = std::min(bbox.mins[j], V0(vert, j) - inflation_radius);
                    bbox.mins[j] = std::min(bbox.mins[j], V1(vert, j) - inflation_radius);
                    bbox.maxs[j] = std::max(bbox.maxs[j], V0(vert, j) + inflation_radius);
                    bbox.maxs[j] = std::max(bbox.maxs[j], V1(vert, j) + inflation_radius);
                }
            }
            AABB<dim>* leaf = new AABB(LT_FACE, i, bbox);
            leaves.push_back(leaf);
        }
    }
    AABB<dim> *root = buildAABB(leaves, 0, leaves.size());
    intersect(root, root, mesh, collisionCandidates);
    delete root;
}

template void AABBBroadPhase<2>(
    const ipc::CollisionMesh& mesh,
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    double inflation_radius,
    ipc::Candidates& collisionCandidates);

template void AABBBroadPhase<3>(
    const ipc::CollisionMesh& mesh,
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    double inflation_radius,
    ipc::Candidates& collisionCandidates);

template void TrivialBroadPhase<2>(
    const ipc::CollisionMesh& mesh,
    double inflation_radius,
    ipc::Candidates& collisionCandidates);

template void TrivialBroadPhase<3>(
    const ipc::CollisionMesh& mesh,
    double inflation_radius,
    ipc::Candidates& collisionCandidates);