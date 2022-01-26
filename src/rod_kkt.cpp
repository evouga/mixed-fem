#include "rod_kkt.h"

#include "linear_tri3dmesh_dphi_dX.h"

using namespace Eigen;
using SparseMatrixdRowMajor = Eigen::SparseMatrix<double,RowMajor>;

//dphi is 2x3 
template<typename DerivedP, typename DerivedV>
void linear_rod3d_dphi_dX(Eigen::DenseBase<DerivedP> &dphi, const  Eigen::MatrixBase<DerivedV> &V, 
                      Eigen::Ref<const Eigen::RowVectorXi> E) {

  Eigen::Matrix<typename DerivedP::Scalar,3,1> T; 
  T.col(0) = (V.row(E(1)) - V.row(E(0))).transpose();
  dphi.block(1,0, 1,3) = (T.transpose()*T).inverse()*T.transpose();
  dphi.row(0) = -dphi.block(1,0, 1,3).colwise().sum();                                
}

SparseMatrixdRowMajor rod_jacobian(const MatrixXd& V, const MatrixXi& E,
    const VectorXd& vols, bool weighted) {
  // J matrix (big jacobian guy)
  SparseMatrixdRowMajor J;
  // 
  std::vector<Triplet<double>> trips;
  for (int i = 0; i < E.rows(); ++i) { 

    Eigen::Matrix<double,2,3> dX; 
    linear_rod3d_dphi_dX(dX, V, E.row(i));
    //std::cout << "DPHIDX: " << dX << std::endl;

    // Local block
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
        int vid = E(i,k); // vertex index

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
  J.resize(9*E.rows(), V.size());
  J.setFromTriplets(trips.begin(),trips.end());
  return J;
}

SparseMatrixd rod_massmatrix(const MatrixXd& V, const MatrixXi& E,
    const VectorXd& vols) {
  std::vector<Triplet<double>> trips;

  for (int i = 0; i < E.rows(); ++i) {
    //for (int j = 0; j < 2; ++j) {
    //  int id = E(i,j);
    //  double val = vols(i);
    //  trips.push_back(Triplet<double>(3*id+0,3*id+0,val));
    //  trips.push_back(Triplet<double>(3*id+1,3*id+1,val));
    //  trips.push_back(Triplet<double>(3*id+2,3*id+2,val));
    //}
    for (int j = 0; j < 2; ++j) {
      int id1 = E(i,j);
      for (int k = 0; k < 2; ++k) {
        int id2 = E(i,k);
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
  SparseMatrixd M;
  M.resize(V.size(),V.size());
  M.setFromTriplets(trips.begin(),trips.end());
  return M;
}
