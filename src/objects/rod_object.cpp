#include "rod_object.h"
#include "svd/svd3x3_sse.h"

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

void RodObject::volumes(VectorXd& vol) {
  vol.resize(T_.rows());
  for (int i = 0; i < T_.rows(); ++i) {
    vol(i) = (V_.row(T_(i,0)) - V_.row(T_(i,1))).norm() * config_->thickness;
  }
}

void RodObject::mass_matrix(SparseMatrixd& M) {

  std::vector<Triplet<double>> trips;

  for (int i = 0; i < T_.rows(); ++i) {
    for (int j = 0; j < 2; ++j) {
      int id1 = T_(i,j);
      for (int k = 0; k < 2; ++k) {
        int id2 = T_(i,k);
        double val = vols_(i);
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
  M = M * config_->density; 
}

void RodObject::jacobian(SparseMatrixdRowMajor& J, bool weighted) {

  std::vector<Triplet<double>> trips;
  for (int i = 0; i < T_.rows(); ++i) { 

    Matrix<double,2,3> dX; 
    linear_rod3d_dphi_dX(dX, V_, T_.row(i));

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
        int vid = T_(i,k); // vertex index

        // x,y,z index for the k-th vertex
        for (int l = 0; l < 3; ++l) {
          double val = B(j,3*k+l);
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


void RodObject::build_rhs() {
  SimObject::build_rhs();

  // Rod mesh has additional normal term on right hand side
  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) {
    // World-space normal and binormals
    Vector9d n = sim::flatten((R_[i]*NN_[i]));
    Vector9d bn = sim::flatten((R_[i]*BN_[i]));

    rhs_.segment(qt_.size() + 9*i, 9) -= vols_(i) * (n + bn);
  }
}

void RodObject::update_SR() {

  VectorXd def_grad = J_*(P_.transpose()*(qt_+dq_)+b_);

  int N = (T_.rows() / 4) + int(T_.rows() % 4 != 0);
  double fac = std::max((la_.array().abs().maxCoeff() + 1e-6), 1.0);

  //std::cout << "NN: " << NN_[0] << " BN: " << BN_[0] << std::endl;

  #pragma omp parallel for 
  for (int ii = 0; ii < N; ++ii) {

    Matrix<float,12,3> Y4,R4;

    // SSE implementation operates on 4 matrices at a time, so assemble
    // 12 x 3 matrices
    for (int jj = 0; jj < 4; ++jj) {
      int i = ii*4 +jj;
      if (i >= T_.rows())
        break;
      Vector9d li = ibeta_*(la_.segment(9*i,9)/fac) + def_grad.segment(9*i,9);
      
      // Update S[i] using new lambdas
      dS_[i] = material_->dS(R_[i], S_[i], la_.segment(9*i,9), Hinv_[i]);
      Vector6d s = S_[i] + dS_[i];

      // Solve rotation matrices
      Matrix3d Cs;
      Cs << s(0), s(3), s(4), 
            s(3), s(1), s(5), 
            s(4), s(5), s(2);
      Cs -= (NN_[i] + BN_[i]); // NOTE only additional line
      Matrix3d y4 = Map<Matrix3d>(li.data())*Cs;
      Y4.block(3*jj, 0, 3, 3) = y4.cast<float>();
    }

    // Solve rotations
    polar_svd3x3_sse(Y4,R4);

    // Assign rotations to per-element matrices
    for (int jj = 0; jj < 4; jj++) {
      int i = ii*4 +jj;
      if (i >= T_.rows())
        break;
      R_[i] = R4.block(3*jj,0,3,3).cast<double>();
    }
  }
}

void RodObject::jacobian_regularized(SparseMatrixdRowMajor& J, bool weighted) {
  std::cerr << "jacobian_regularized not implemented for rod" << std::endl;
}
