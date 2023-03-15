// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2023
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
// https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// Version: 6.0.2022.12.14

#pragma once

#include "Mathematics/DistPointHyperplane.h"
#include "Mathematics/Vector2.h"
#include "MTMesh.h"
#include <iostream>

template <typename Real>
class ConvexPolyhedron : public MTMesh
{
public:
    typedef typename std::vector<gte::Vector2<Real>> V2Array;
    typedef typename std::vector<gte::Vector3<Real>> V3Array;
    typedef typename std::vector<gte::Plane3<Real>> PArray;
    typedef typename std::vector<int32_t> IArray;

    // Construction.
    ConvexPolyhedron()
        :
        mPoints{},
        mPlanes{},
        mCentroid{ static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(0) },
        mQuery{}
    {
    }

    ConvexPolyhedron(V3Array const& points, IArray const& indices)
    {
        Create(points, indices);
    }

    ConvexPolyhedron(V3Array const& points, IArray const& indices, PArray const& planes)
    {
        Create(points, indices, planes);
    }

    ConvexPolyhedron(ConvexPolyhedron const& other)
        :
        MTMesh(other),
        mPoints(other.mPoints),
        mPlanes(other.mPlanes)
    {
    }

    void Create(V3Array const& points, IArray const& indices)
    {
        LogAssert(points.size() >= 4 && indices.size() >= 4,
            "Polyhedron must be at least a tetrahedron.");

        int32_t const numVertices = static_cast<int32_t>(points.size());
        int32_t const numTriangles = static_cast<int32_t>(indices.size() / 3);
        int32_t const numEdges = numVertices + numTriangles - 2;
        Reset(numVertices, numEdges, numTriangles);
        mPoints = points;

        // Copy polyhedron points into vertex array.  Compute centroid for use
        // in making sure the triangles are counterclockwise oriented when
        // viewed from the outside.
        ComputeCentroid();

        // Get polyhedron edge and triangle information.
        int32_t const* currentIndex = indices.data();
        for (int32_t t = 0; t < numTriangles; ++t)
        {
            // Get vertex indices for triangle.
            int32_t v0 = *currentIndex++;
            int32_t v1 = *currentIndex++;
            int32_t v2 = *currentIndex++;

            // Make sure triangle is counterclockwise.
            gte::Vector3<Real>& vertex0 = mPoints[v0];
            gte::Vector3<Real>& vertex1 = mPoints[v1];
            gte::Vector3<Real>& vertex2 = mPoints[v2];

            gte::Vector3<Real> diff = mCentroid - vertex0;
            gte::Vector3<Real> edge1 = vertex1 - vertex0;
            gte::Vector3<Real> edge2 = vertex2 - vertex0;
            gte::Vector3<Real> normal = Cross(edge1, edge2);
            Real length = Length(normal);
            if (length > (Real)0)
            {
                normal /= length;
            }
            else
            {
                // The triangle is degenerate, use a "normal" that points
                // towards the centroid.
                normal = diff;
                Normalize(normal);
            }

            Real signedDistance = Dot(normal, diff);
            if (signedDistance < (Real)0)
            {
                // The triangle is counterclockwise.
                Insert(v0, v1, v2);
            }
            else
            {
                // The triangle is clockwise.
                Insert(v0, v2, v1);
            }
        }

        UpdatePlanes();
    }

    void Create(const V3Array& points, const IArray& indices, const PArray& planes)
    {
        LogAssert(points.size() >= 4 && indices.size() >= 4,
            "Polyhedron must be at least a tetrahedron.");

        int32_t const numVertices = static_cast<int32_t>(points.size());
        int32_t const numTriangles = static_cast<int32_t>(indices.size() / 3);
        int32_t const numEdges = numVertices + numTriangles - 2;
        Reset(numVertices, numEdges, numTriangles);
        mPoints = points;
        mPlanes = planes;

        // Copy polyhedron points into vertex array.  Compute centroid for use
        // in making sure the triangles are counterclockwise oriented when
        // viewed from the outside.
        ComputeCentroid();

        // Get polyhedron edge and triangle information.
        int32_t const* currentIndex = indices.data();
        for (int32_t t = 0; t < numTriangles; ++t)
        {
            // Get vertex indices for triangle.
            int32_t v0 = *currentIndex++;
            int32_t v1 = *currentIndex++;
            int32_t v2 = *currentIndex++;

            Real signedDistance = mQuery(mCentroid, mPlanes[t]).signedDistance;
            if (signedDistance > (Real)0)
            {
                // The triangle is counterclockwise.
                Insert(v0, v1, v2);
            }
            else
            {
                // The triangle is clockwise.
                Insert(v0, v2, v1);
            }
        }
    }

    // Assignment.
    ConvexPolyhedron& operator=(const ConvexPolyhedron& polyhedron)
    {
        MTMesh::operator=(polyhedron);
        mPoints = polyhedron.mPoints;
        mPlanes = polyhedron.mPlanes;
        return *this;
    }

    // Read points and planes.
    inline V3Array const& GetPoints() const
    {
        return mPoints;
    }

    inline gte::Vector3<Real> const& GetPoint(int32_t i) const
    {
        return mPoints[i];
    }

    inline void SetPoint(int32_t i, gte::Vector3<Real> const& point)
    {
        mPoints[i] = point;
    }

    inline PArray const& GetPlanes() const
    {
        return mPlanes;
    }

    inline gte::Plane3<Real> const& GetPlane(int32_t i) const
    {
        return mPlanes[i];
    }

    // Allow vertex modification.  The caller is responsible for preserving
    // the convexity.  After modifying the vertices, call UpdatePlanes to
    // recompute the planes of the polyhedron faces.
    int32_t AddPoint(gte::Vector3<Real> const& point)
    {
        int32_t numPoints = static_cast<int32_t>(mPoints.size());
        mPoints.push_back(point);
        return numPoints;
    }

    void UpdatePlanes()
    {
        ComputeCentroid();

        int32_t const numTriangles = mTriangles.GetNumElements();
        mPlanes.resize(numTriangles);
        for (int32_t t = 0; t < numTriangles; ++t)
        {
            MTTriangle& triangle = mTriangles[t];
            int32_t v0 = GetVLabel(triangle.GetVertex(0));
            int32_t v1 = GetVLabel(triangle.GetVertex(1));
            int32_t v2 = GetVLabel(triangle.GetVertex(2));
            gte::Vector3<Real>& vertex0 = mPoints[v0];
            gte::Vector3<Real>& vertex1 = mPoints[v1];
            gte::Vector3<Real>& vertex2 = mPoints[v2];

            gte::Vector3<Real> diff = -(mCentroid - vertex0);
            gte::Vector3<Real> edge1 = vertex1 - vertex0;
            gte::Vector3<Real> edge2 = vertex2 - vertex0;
            gte::Vector3<Real> normal = Cross(edge2, edge1);
            Real length = Length(normal);
            if (length > (Real)0)
            {
                normal /= length;
                Real dot = Dot(normal, diff);
                if (dot < (Real)0)
                {
                    normal = -normal;
                }
            }
            else
            {
                // The triangle is degenerate, use a "normal" that points
                // from the centroid.
                normal = diff;
                Normalize(normal);
            }

            // The plane has outer-pointing normal.
            mPlanes[t] = gte::Plane3<Real>(normal, Dot(normal, vertex0));
        }
    }

    void ComputeCentroid()
    {
        mCentroid = { (Real)0, (Real)0, (Real)0 };
        for (auto const& point : mPoints)
        {
            mCentroid += point;
        }
        mCentroid /= static_cast<Real>(mPoints.size());
    }

    inline gte::Vector3<Real> const& GetCentroid() const
    {
        return mCentroid;
    }

    int WhichSide(const ConvexPolyhedron& p, const gte::Vector3<Real>& P,
        const gte::Vector3<Real>& D) const {
      int numPositive = 0;
      int numNegative = 0;

      for (int i = 0; i < p.GetPoints().size(); ++i) {
        gte::Vector3<Real> diff = p.GetPoint(i) - P;
        Real dot = Dot(diff, D);
        if (dot > 0) {
          ++numPositive;
        }
        else if (dot < 0) {
          ++numNegative;
        }
        if (numPositive > 0 && numNegative > 0) {
          return 0;
        }
      }
      return (numPositive > 0 ? 1 : -1);
    }


    // Compute the polyhedron of intersection.
    bool HasIntersection(const ConvexPolyhedron& polyhedron) const
    {
      // Compute sides from other polyhedron verts and this polyhedron's planes
      int32_t const numTriangles = mTriangles.GetNumElements();
      for (int t = 0; t < numTriangles; ++t)
      {
        gte::Plane3<Real> const& plane = mPlanes[t];
        const MTTriangle& triangle = mTriangles[t];
        int32_t v0 = GetVLabel(triangle.GetVertex(0));
        const gte::Vector3<Real>& vertex0 = mPoints[v0];
        if (WhichSide(polyhedron, vertex0, plane.normal) > 0) {
            return false;
        }
      }

      // And now for the other polyhedron's planes
      for (int t = 0; t < polyhedron.GetNumTriangles(); ++t) {
        gte::Plane3<Real> const& plane = polyhedron.GetPlane(t);
        const MTTriangle& triangle = polyhedron.GetTriangle(t);
        int32_t v0 = GetVLabel(triangle.GetVertex(0));
        const gte::Vector3<Real>& vertex0 = polyhedron.GetPoint(v0);
        if (WhichSide(polyhedron, vertex0, plane.normal) > 0) {
            return false;
        }
      }

      // Test cross product of pairs of edge directions,
      // one from each polyhedron
      for (int i = 0; i < GetNumEdges(); ++i) {
        auto const& edge = GetEdge(i);
        auto const& e0_v0 = mPoints[GetVLabel(edge.GetVertex(0))];
        auto const& e0_v1 = mPoints[GetVLabel(edge.GetVertex(1))];

        auto D0 = e0_v1 - e0_v0;

        for (int j = 0; j < polyhedron.GetNumEdges(); ++j) {
          auto const& edge_j = polyhedron.GetEdge(j);
          int32_t v0 = polyhedron.GetVLabel(edge_j.GetVertex(0));
          int32_t v1 = polyhedron.GetVLabel(edge_j.GetVertex(1));
          auto const& e1_v0 = polyhedron.mPoints[v0];
          auto const& e1_v1 = polyhedron.mPoints[v1];
          auto D1 = e1_v1 - e1_v0;
          auto N = Cross(D0, D1);

          double len = Length(N);
          if (len > 1e-8) {
            int side0 = WhichSide(*this, e0_v0, N);
            if (side0 == 0) {
              continue;
            }
            int side1 = WhichSide(polyhedron, e0_v0, N);
            if (side1 == 0) {
              continue;
            }
            if (side0 * side1 < 0) {
              return false;
            }
          }
          

        }

      }
      return true;
    }

    Real GetSurfaceArea() const
    {
        Real surfaceArea = (Real)0;

        int32_t const numTriangles = mTriangles.GetNumElements();
        for (int32_t t = 0; t < numTriangles; ++t)
        {
            MTTriangle const& triangle = mTriangles[t];
            int32_t v0 = GetVLabel(triangle.GetVertex(0));
            int32_t v1 = GetVLabel(triangle.GetVertex(1));
            int32_t v2 = GetVLabel(triangle.GetVertex(2));
            gte::Vector3<Real> const& vertex0 = mPoints[v0];
            gte::Vector3<Real> const& vertex1 = mPoints[v1];
            gte::Vector3<Real> const& vertex2 = mPoints[v2];
            gte::Vector3<Real> const& normal = mPlanes[t].normal;

            surfaceArea += GetTriangleArea(normal, vertex0, vertex1, vertex2);
        }

        return surfaceArea;
    }

    Real GetVolume() const
    {
        Real volume = (Real)0;

        int32_t const numTriangles = mTriangles.GetNumElements();
        for (int32_t t = 0; t < numTriangles; ++t)
        {
            MTTriangle const& triangle = mTriangles[t];
            int32_t v0 = GetVLabel(triangle.GetVertex(0));
            int32_t v1 = GetVLabel(triangle.GetVertex(1));
            int32_t v2 = GetVLabel(triangle.GetVertex(2));
            gte::Vector3<Real> const& vertex0 = mPoints[v0];
            gte::Vector3<Real> const& vertex1 = mPoints[v1];
            gte::Vector3<Real> const& vertex2 = mPoints[v2];
            volume += Dot(vertex0, Cross(vertex1, vertex2));
        }

        volume /= (Real)6;
        return volume;
    }

    bool ContainsPoint(gte::Vector3<Real> const& point) const
    {
        int32_t const numTriangles = mTriangles.GetNumElements();
        for (int32_t t = 0; t < numTriangles; ++t)
        {
            Real signedDistance = mQuery(point, mPlanes[t]).signedDistance;
            if (signedDistance < (Real)0)
            {
                return false;
            }
        }
        return true;
    }

    // Create an egg-shaped object that is axis-aligned and centered at
    // (xc,yc,zc).  The input bounds are all positive and represent the
    // distances from the center to the six extreme points on the egg.
    static void CreateEggShape(gte::Vector3<Real> const& center, Real x0,
        Real x1, Real y0, Real y1, Real z0, Real z1, int32_t maxSteps,
        ConvexPolyhedron& egg)
    {
        LogAssert(x0 > (Real)0 && x1 > (Real)0, "Invalid input.");
        LogAssert(y0 > (Real)0 && y1 > (Real)0, "Invalid input.");
        LogAssert(z0 > (Real)0 && z1 > (Real)0, "Invalid input.");
        LogAssert(maxSteps >= 0, "Invalid input.");

        // Start with an octahedron whose 6 vertices are (-x0,0,0), (x1,0,0),
        // (0,-y0,0), (0,y1,0), (0,0,-z0), (0,0,z1).  The center point will be
        // added later.
        V3Array points(6);
        points[0] = { -x0, (Real)0, (Real)0 };
        points[1] = { +x1, (Real)0, (Real)0 };
        points[2] = { (Real)0, -y0, (Real)0 };
        points[3] = { (Real)0, +y1, (Real)0 };
        points[4] = { (Real)0, (Real)0, -z0 };
        points[5] = { (Real)0, (Real)0, +z1 };

        IArray indices(24);
        indices[0] = 1;  indices[1] = 3;  indices[2] = 5;
        indices[3] = 3;  indices[4] = 0;  indices[5] = 5;
        indices[6] = 0;  indices[7] = 2;  indices[8] = 5;
        indices[9] = 2;  indices[10] = 1;  indices[11] = 5;
        indices[12] = 3;  indices[13] = 1;  indices[14] = 4;
        indices[15] = 0;  indices[16] = 3;  indices[17] = 4;
        indices[18] = 2;  indices[19] = 0;  indices[20] = 4;
        indices[21] = 1;  indices[22] = 2;  indices[23] = 4;

        egg.SetInitialELabel(0);
        egg.Create(points, indices);

        // Subdivide the triangles.  The midpoints of the edges are computed.
        // The triangle is replaced by four subtriangles using the original 3
        // vertices and the 3 new edge midpoints.
        for (int32_t step = 1; step <= maxSteps; ++step)
        {
            int32_t numVertices = egg.GetNumVertices();
            int32_t numEdges = egg.GetNumEdges();
            int32_t numTriangles = egg.GetNumTriangles();

            // Compute lifted edge midpoints.
            for (int32_t i = 0; i < numEdges; ++i)
            {
                // Get an edge.
                MTEdge const& edge = egg.GetEdge(i);
                int32_t v0 = egg.GetVLabel(edge.GetVertex(0));
                int32_t v1 = egg.GetVLabel(edge.GetVertex(1));

                // Compute lifted centroid to points.
                gte::Vector3<Real> lifted = egg.GetPoint(v0) + egg.GetPoint(v1);
                Real xr = (lifted[0] > (Real)0 ? lifted[0] / x1 : lifted[0] / x0);
                Real yr = (lifted[1] > (Real)0 ? lifted[1] / y1 : lifted[1] / y0);
                Real zr = (lifted[2] > (Real)0 ? lifted[2] / z1 : lifted[2] / z0);
                lifted *= (Real)1 / std::sqrt(xr * xr + yr * yr + zr * zr);

                // Add the point to the array.  Store the point index in the
                // edge label for support in adding new triangles.
                egg.SetELabel(i, numVertices);
                ++numVertices;
                egg.AddPoint(lifted);
            }

            // Add the new triangles and remove the old triangle.  The removal
            // in slot i will cause the last added triangle to be moved to
            // that slot.  This side effect will not interfere with the
            // iteration and removal of the triangles.
            for (int32_t i = 0; i < numTriangles; ++i)
            {
                MTTriangle const& triangle = egg.GetTriangle(i);
                int32_t v0 = egg.GetVLabel(triangle.GetVertex(0));
                int32_t v1 = egg.GetVLabel(triangle.GetVertex(1));
                int32_t v2 = egg.GetVLabel(triangle.GetVertex(2));
                int32_t v01 = egg.GetELabel(triangle.GetEdge(0));
                int32_t v12 = egg.GetELabel(triangle.GetEdge(1));
                int32_t v20 = egg.GetELabel(triangle.GetEdge(2));
                egg.Insert(v0, v01, v20);
                egg.Insert(v01, v1, v12);
                egg.Insert(v20, v12, v2);
                egg.Insert(v01, v12, v20);
                egg.Remove(v0, v1, v2);
            }
        }

        // Add the center.
        for (auto& point : egg.mPoints)
        {
            point += center;
        }

        egg.UpdatePlanes();
    }

    // Debugging support.
    virtual void Print(std::ofstream& output) const override
    {
        MTMesh::Print(output);

        output << "points:" << std::endl;
        int32_t const numPoints = static_cast<int32_t>(mPoints.size());
        for (int32_t i = 0; i < numPoints; ++i)
        {
            gte::Vector3<Real> const& point = mPoints[i];
            output << "point<" << i << "> = (";
            output << point[0] << ", ";
            output << point[1] << ", ";
            output << point[2] << ") ";
            output << std::endl;
        }
        output << std::endl;

        output << "planes:" << std::endl;
        int32_t const numPlanes = static_cast<int32_t>(mPlanes.size());
        for (int32_t i = 0; i < numPlanes; ++i)
        {
            gte::Plane3<Real> const& plane = mPlanes[i];
            output << "plane<" << i << "> = (";
            output << plane.normal[0] << ", ";
            output << plane.normal[1] << ", ";
            output << plane.normal[2] << ", ";
            output << plane.constant << ")";
            output << std::endl;
        }
        output << std::endl;
    }

    virtual bool Print(std::string const& filename) const override
    {
        std::ofstream output(filename);
        if (!output)
        {
            return false;
        }

        Print(output);
        return true;
    }

private:

    // Support for computing surface area.
    Real GetTriangleArea(gte::Vector3<Real> const& normal,
        gte::Vector3<Real> const& vertex0, gte::Vector3<Real> const& vertex1,
        gte::Vector3<Real> const& vertex2) const
    {
        // Compute maximum absolute component of normal vector.
        int32_t maxIndex = 0;
        Real maxAbsValue = std::fabs(normal[0]);

        Real absValue = std::fabs(normal[1]);
        if (absValue > maxAbsValue)
        {
            maxIndex = 1;
            maxAbsValue = absValue;
        }

        absValue = std::fabs(normal[2]);
        if (absValue > maxAbsValue)
        {
            maxIndex = 2;
            maxAbsValue = absValue;
        }

        // Trap degenerate triangles.
        if (maxAbsValue == (Real)0)
        {
            return (Real)0;
        }

        // Compute area of projected triangle.
        Real d0, d1, d2, area;
        if (maxIndex == 0)
        {
            d0 = vertex1[2] - vertex2[2];
            d1 = vertex2[2] - vertex0[2];
            d2 = vertex0[2] - vertex1[2];
            area = std::fabs(vertex0[1] * d0 + vertex1[1] * d1 + vertex2[1] * d2);
        }
        else if (maxIndex == 1)
        {
            d0 = vertex1[0] - vertex2[0];
            d1 = vertex2[0] - vertex0[0];
            d2 = vertex0[0] - vertex1[0];
            area = std::fabs(vertex0[2] * d0 + vertex1[2] * d1 + vertex2[2] * d2);
        }
        else
        {
            d0 = vertex1[1] - vertex2[1];
            d1 = vertex2[1] - vertex0[1];
            d2 = vertex0[1] - vertex1[1];
            area = std::fabs(vertex0[0] * d0 + vertex1[0] * d1 + vertex2[0] * d2);
        }

        area *= (Real)0.5 / maxAbsValue;
        return area;
    }

    static bool IsNegativeProduct(Real distance0, Real distance1)
    {
        return (distance0 != (Real)0 ? (distance0 * distance1 <= (Real)0) :
            (distance1 != (Real)0));
    }

    V3Array mPoints;
    PArray mPlanes;
    gte::Vector3<Real> mCentroid;
    mutable gte::DCPQuery<Real, gte::Vector3<Real>, gte::Plane3<Real>> mQuery;
};
