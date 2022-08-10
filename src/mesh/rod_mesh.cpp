#include "rod_mesh.h"
#include "svd/svd3x3_sse.h"
#include "config.h"
#include "energies/material_model.h"

using namespace Eigen;
using namespace mfem;

namespace {
  template<typename DerivedP, typename DerivedV>
  void linear_rod3d_dphi_dX(DenseBase<DerivedP> &dphi,
      const MatrixBase<DerivedV> &V, 
      Ref<const RowVectorXi> E) {

    Matrix<typename DerivedP::Scalar,3,1> T; 
    T.col(0) = (V.row(E(1)) - V.row(E(0))).transpose();
    dphi.block(1,0, 1,3) = (T.transpose()*T).inverse()*T.transpose();
    dphi.row(0) = -dphi.block(1,0, 1,3).colwise().sum();                                
  }
}

void RodMesh::volumes(VectorXd& vol) {
  vol.resize(T_.rows());
  for (int i = 0; i < T_.rows(); ++i) {
    vol(i) = (V_.row(T_(i,0)) - V_.row(T_(i,1))).norm() * material_->config()->thickness;
  }
}

void RodMesh::mass_matrix(SparseMatrixdRowMajor& M, const VectorXd& vols) {

  std::vector<Triplet<double>> trips;

  for (int i = 0; i < T_.rows(); ++i) {
    for (int j = 0; j < 2; ++j) {
      int id1 = T_(i,j);
      for (int k = 0; k < 2; ++k) {
        int id2 = T_(i,k);
        double val = vols(i);
        if (j == k)
          val /= 1;
        else
          val /= 2;
        trips.push_back(Triplet<double>(3*id1+0,3*id2,val));
        trips.push_back(Triplet<double>(3*id1+1,3*id2+1,val));
        trips.push_back(Triplet<double>(3*id1+2,3*id2+2,val));
      }
    }
  }
  M.resize(V_.size(),V_.size());
  M.setFromTriplets(trips.begin(),trips.end());

  // note: assuming uniform density and thickness
  M = M * material_->config()->density; 
}

void RodMesh::jacobian(SparseMatrixdRowMajor& J, const VectorXd& vols,
      bool weighted) {

  std::vector<Triplet<double>> trips;
  for (int i = 0; i < T_.rows(); ++i) { 

    Matrix<double,2,3> dX; 
    linear_rod3d_dphi_dX(dX, V_, T_.row(i));

    // Local block
    std::cerr << "wrong" << std::endl;
    Matrix<double,9,6> B;
    B  << 
      dX(0,0), 0      , 0      , dX(1,0), 0      , 0,     
      dX(0,1), 0      , 0      , dX(1,1), 0      , 0,     
      dX(0,2), 0      , 0      , dX(1,2), 0      , 0,     
      0      , dX(0,0), 0      , 0      , dX(1,0), 0,     
      0      , dX(0,1), 0      , 0      , dX(1,1), 0,     
      0      , dX(0,2), 0      , 0      , dX(1,2), 0,     
      0      , 0      , dX(0,0), 0      , 0      , dX(1,0),
      0      , 0      , dX(0,1), 0      , 0      , dX(1,1),
      0      , 0      , dX(0,2), 0      , 0      , dX(1,2);

    // Assembly for the i-th lagrange multiplier matrix which
    // is associated with 3 vertices (for tetrahedra)
    for (int j = 0; j < 9; ++j) {

      // k-th vertex of the tetrahedra
      for (int k = 0; k < 2; ++k) {
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