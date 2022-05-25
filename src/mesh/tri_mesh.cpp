#include "tri_mesh.h"
#include <igl/doublearea.h>
#include "linear_tri3dmesh_dphi_dX.h"
#include "svd/svd3x3_sse.h"

using namespace Eigen;
using namespace mfem;


namespace {

  // From dphi/dX, form jacobian dphi/dq where
  // q is an elements local set of vertices
  template <typename Scalar>
  void local_jacobian(Matrix<Scalar,9,9>& B, const Matrix<Scalar,3,3>& dX) {
    B  << dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0, 0,
          0, dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0,
          0, 0, dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0),
          dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0, 0, 
          0, dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0,
          0, 0, dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1),
          dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0, 0,
          0, dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0,
          0, 0, dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2);
  }

}

void TriMesh::volumes(Eigen::VectorXd& vol) {
  igl::doublearea(V0_, T_, vol);
  vol.array() *= (config_->thickness/2);
}

void TriMesh::mass_matrix(Eigen::SparseMatrixdRowMajor& M,
    const VectorXd& vols) {
std::cerr << "WRONG use volumes dummy" << std::endl;
  std::vector<Triplet<double>> trips;
  VectorXd dblA;
  igl::doublearea(V0_, T_, dblA);

  // 1. Mass matrix terms
  for (int i = 0; i < T_.rows(); ++i) {
    for (int j = 0; j < 3; ++j) {
      int id1 = T_(i,j);
      for (int k = 0; k < 3; ++k) {
        int id2 = T_(i,k);
        double val = dblA(i);
        if (j == k)
          val /= 12;
        else
          val /= 24;
        trips.push_back(Triplet<double>(3*id1+0,3*id2,val));
        trips.push_back(Triplet<double>(3*id1+1,3*id2+1,val));
        trips.push_back(Triplet<double>(3*id1+2,3*id2+2,val));
      }
    }
  }
  M.resize(V_.size(),V_.size());
  M.setFromTriplets(trips.begin(),trips.end());

  // note: assuming uniform density and thickness
  M = M * config_->density * config_->thickness; 
}

void TriMesh::jacobian(SparseMatrixdRowMajor& J, const VectorXd& vols,
      bool weighted) {

  MatrixXd dphidX;
  sim::linear_tri3dmesh_dphi_dX(dphidX, V0_, T_);

  std::vector<Triplet<double>> trips;
  for (int i = 0; i < T_.rows(); ++i) { 

    // Local block
    Matrix9d B;
    Matrix3d dX = sim::unflatten<3,3>(dphidX.row(i));
    local_jacobian(B, dX);

    // Assembly for the i-th lagrange multiplier matrix which
    // is associated with 3 vertices (for tetrahedra)
    for (int j = 0; j < 9; ++j) {

      // k-th vertex of the tetrahedra
      for (int k = 0; k < 3; ++k) {
        int vid = T_(i,k); // vertex index

        // x,y,z index for the k-th vertex
        for (int l = 0; l < 3; ++l) {
          double val = B(j,3*k+l);
          if (weighted)
            val *= vols(i);
          trips.push_back(Triplet<double>(9*i+j, 3*vid+l, val));
        }
      }
    }
  }
  J.resize(9*T_.rows(), V_.size());
  J.setFromTriplets(trips.begin(),trips.end());
}

void TriMesh::jacobian(std::vector<MatrixXd>& J) {
  J.resize(T_.rows());

  MatrixXd dphidX;
  sim::linear_tri3dmesh_dphi_dX(dphidX, V0_, T_);

  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) { 
    // Local block
    Matrix9d B;
    Matrix3d dX = sim::unflatten<3,3>(dphidX.row(i));
    local_jacobian(B, dX);
    J[i] = B;
  }
}
