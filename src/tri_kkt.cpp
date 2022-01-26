#include "tri_kkt.h"

#include "linear_tri3dmesh_dphi_dX.h"

using namespace Eigen;
using SparseMatrixdRowMajor = Eigen::SparseMatrix<double,RowMajor>;

//TODO duplicate of tet one. When im not sleepy fix this
void tri_kkt_lhs(const SparseMatrixd& M, const SparseMatrixdRowMajor& Jw,
    double ih2, std::vector<Triplet<double>>& trips) {

  trips.clear();

  // Mass matrix terms
  for (int k=0; k < M.outerSize(); ++k) {
    for (SparseMatrixd::InnerIterator it(M,k); it; ++it) {
      trips.push_back(Triplet<double>(it.row(),it.col(),ih2*it.value()));
    }
  }

  int offset = M.rows(); // offset for off diagonal blocks

  // Jacobian off-diagonal entries
  for (int k=0; k < Jw.outerSize(); ++k) {
    for (SparseMatrixdRowMajor::InnerIterator it(Jw,k); it; ++it) {
      trips.push_back(Triplet<double>(offset+it.row(),it.col(),it.value()));
      trips.push_back(Triplet<double>(it.col(),offset+it.row(),it.value()));
    }
  }
}


SparseMatrixdRowMajor tri_jacobian(const MatrixXd& V, const MatrixXi& T,
    const VectorXd& vols, bool weighted) {
  // J matrix (big jacobian guy)
  SparseMatrixdRowMajor J;
  MatrixXd dphidX;
  sim::linear_tri3dmesh_dphi_dX(dphidX, V, T);

  std::vector<Triplet<double>> trips;
  for (int i = 0; i < T.rows(); ++i) { 

    Matrix3d dX = sim::unflatten<3,3>(dphidX.row(i));

    // Local block
    Matrix9d B;
    B  << 
      dX(0,0), 0      , 0      , dX(1,0), 0      , 0      , dX(2,0), 0      , 0, 
      dX(0,1), 0      , 0      , dX(1,1), 0      , 0      , dX(2,1), 0      , 0, 
      dX(0,2), 0      , 0      , dX(1,2), 0      , 0      , dX(2,2), 0      , 0,
      0      , dX(0,0), 0      , 0      , dX(1,0), 0      , 0      , dX(2,0), 0,
      0      , dX(0,1), 0      , 0      , dX(1,1), 0      , 0      , dX(2,1), 0,
      0      , dX(0,2), 0      , 0      , dX(1,2), 0      , 0      , dX(2,2), 0,
      0      , 0      , dX(0,0), 0      , 0      , dX(1,0), 0      , 0      , dX(2,0),
      0      , 0      , dX(0,1), 0      , 0      , dX(1,1), 0      , 0      , dX(2,1),
      0      , 0      , dX(0,2), 0      , 0      , dX(1,2), 0      , 0      , dX(2,2);

    // Assembly for the i-th lagrange multiplier matrix which
    // is associated with 3 vertices (for tetrahedra)
    for (int j = 0; j < 9; ++j) {

      // k-th vertex of the tetrahedra
      for (int k = 0; k < 3; ++k) {
        int vid = T(i,k); // vertex index

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
  J.resize(9*T.rows(), V.size());
  J.setFromTriplets(trips.begin(),trips.end());
  return J;
}

SparseMatrixd trimesh_massmatrix(const MatrixXd& V, const MatrixXi& F,
    const VectorXd& dblA) {
  std::vector<Triplet<double>> trips;

  // 1. Mass matrix terms
  for (int i = 0; i < F.rows(); ++i) {
    for (int j = 0; j < 3; ++j) {
      int id1 = F(i,j);
      for (int k = 0; k < 3; ++k) {
        int id2 = F(i,k);
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
  SparseMatrixd M;
  M.resize(V.size(),V.size());
  M.setFromTriplets(trips.begin(),trips.end());
  return M;
}
