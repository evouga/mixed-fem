#pragma once

#include <igl/AABB.h>
#include <igl/in_element.h>
#include <igl/barycentric_coordinates.h>

namespace mfem {

  // Construct linear blend skinning weight matrix
  // V - Vertices
  // T - Tetrahedral mesh elements
  // Vout - "target" vertices for skinning
  // W - Blending weights
  void linear_blend_skinning(Eigen::MatrixXd V, Eigen::MatrixXi T,
      Eigen::MatrixXd Vout,
      Eigen::SparseMatrix<double, Eigen::RowMajor> W) {

    igl::AABB<Eigen::MatrixXd,3> aabb;
    aabb.init(V, T);
    Eigen::VectorXi I;
    in_element(V,T,Vout,aabb,I);

    std::vector<Eigen::Triplet<double>> trips;
    for (int i = 0; i < I.rows(); ++i) {
      Eigen::RowVector4d coords;
      // If skinning vertex is inside the simulation mesh,
      // just get the barycentric coordinates of the vertex,
      // otherwise find the clsoest point on the mesh and compute
      // the barycentric coordinates
      if (I(i) >= 0) {
        igl::barycentric_coordinates(Vout.row(i),
            V.row(T(I(i),0)),
            V.row(T(I(i),1)),
            V.row(T(I(i),2)),
            V.row(T(I(i),3)), coords);
      } else {
        int ci;
        Eigen::RowVector3d c;
        aabb.squared_distance(V, T, Vout.row(i), ci, c);
        igl::barycentric_coordinates(c,
            V.row(T(ci,0)),
            V.row(T(ci,1)),
            V.row(T(ci,2)),
            V.row(T(ci,3)), coords);
      }
      trips.push_back(Eigen::Triplet<double>(i, T(I(i),0), coords(0)));
      trips.push_back(Eigen::Triplet<double>(i, T(I(i),1), coords(1)));
      trips.push_back(Eigen::Triplet<double>(i, T(I(i),2), coords(2)));
      trips.push_back(Eigen::Triplet<double>(i, T(I(i),3), coords(3)));
    }
    W.resize(Vout.rows(),V.rows());
    W.setFromTriplets(trips.begin(),trips.end());
  }
}
