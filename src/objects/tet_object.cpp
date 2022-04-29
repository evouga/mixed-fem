#include "tet_object.h"
#include <igl/volume.h>
#include "linear_tetmesh_mass_matrix.h"
#include "linear_tet_mass_matrix.h"
#include "linear_tetmesh_dphi_dX.h"

using namespace Eigen;
using namespace mfem;

void TetrahedralObject::volumes(Eigen::VectorXd& vol) {
  igl::volume(V_, T_, vol);
  vol = vol.cwiseAbs();
}

void TetrahedralObject::mass_matrix(Eigen::SparseMatrixd& M) {
  VectorXd densities = VectorXd::Constant(T_.rows(), config_->density);
  sim::linear_tetmesh_mass_matrix(M, V_, T_, densities, vols_);
}

void TetrahedralObject::jacobian(SparseMatrixdRowMajor& J, bool weighted) {
  // J matrix (big jacobian guy)
  MatrixXd dphidX;
  sim::linear_tetmesh_dphi_dX(dphidX, V_, T_);

  std::vector<Triplet<double>> trips;
  for (int i = 0; i < T_.rows(); ++i) { 

    Matrix<double, 4,3> dX = sim::unflatten<4,3>(dphidX.row(i));

    // Local block
    Matrix<double,9,12> B;
    // B  << dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0, 0, dX(3,0), 0, 0,
    //       dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0, 0, dX(3,1), 0, 0, 
    //       dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0, 0, dX(3,2), 0, 0, 
    //       0, dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0, 0, dX(3,0), 0,
    //       0, dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0, 0, dX(3,1), 0,
    //       0, dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0, 0, dX(3,2), 0,
    //       0, 0, dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0, 0, dX(3,0),
    //       0, 0, dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0, 0, dX(3,1),
    //       0, 0, dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0, 0, dX(3,2);
    B  << dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0, 0, dX(3,0), 0, 0,
          0, dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0, 0, dX(3,0), 0,
          0, 0, dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0, 0, dX(3,0),
          dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0, 0, dX(3,1), 0, 0, 
          0, dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0, 0, dX(3,1), 0,
          0, 0, dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0, 0, dX(3,1),
          dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0, 0, dX(3,2), 0, 0, 
          0, dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0, 0, dX(3,2), 0,
          0, 0, dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0, 0, dX(3,2);

    // Assembly for the i-th lagrange multiplier matrix which
    // is associated with 4 vertices (for tetrahedra)
    for (int j = 0; j < 9; ++j) {

      // k-th vertex of the tetrahedra
      for (int k = 0; k < 4; ++k) {
        int vid = T_(i,k); // vertex index

        // x,y,z index for the k-th vertex
        for (int l = 0; l < 3; ++l) {
          double val = B(j,3*k+l) ;//* vols(i); 
          if (weighted)
            val *= vols_(i);
          trips.push_back(Triplet<double>(9*i+j, 3*vid+l, val));
        }
      }
    }
  }
  J.resize(9*T_.rows(), V_.size());
  J.setFromTriplets(trips.begin(),trips.end());
}
