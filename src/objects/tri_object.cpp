#include "tri_object.h"
#include <igl/doublearea.h>
#include "linear_tri3dmesh_dphi_dX.h"
#include "svd/svd3x3_sse.h"

using namespace Eigen;
using namespace mfem;

void TriObject::volumes(Eigen::VectorXd& vol) {
  igl::doublearea(V_, T_, vol);
  vol.array() *= (config_->thickness/2);
}

void TriObject::mass_matrix(Eigen::SparseMatrixd& M) {

  std::vector<Triplet<double>> trips;
  VectorXd dblA;
  igl::doublearea(V_, T_, dblA);

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

void TriObject::jacobian(SparseMatrixdRowMajor& J, bool weighted) {

  MatrixXd dphidX;
  sim::linear_tri3dmesh_dphi_dX(dphidX, V_, T_);

  std::vector<Triplet<double>> trips;
  for (int i = 0; i < T_.rows(); ++i) { 

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

void TriObject::build_rhs() {
  SimObject::build_rhs();

  // Trimesh has additional normal term on right hand side
  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) {
    Vector9d n = sim::flatten((R_[i]*NN_[i]));
    rhs_.segment(qt_.size() + 9*i, 9) -= vols_(i) * n;
  }
}

/*
void TriObject::fit_rotations() {

  VectorXd def_grad = J_*(P_.transpose()*(qt_+dq_)+b_);

  int N = (T_.rows() / 4) + int(T_.rows() % 4 != 0);
  double fac = std::max((la_.array().abs().maxCoeff() + 1e-6), 1.0);

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
      Cs -= NN_[i]; // NOTE only additional line with cloth
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
*/

VectorXd TriObject::collision_force() {

  Vector3d N(.4,.2,.8);
  N = N / N.norm();
  double d = config_->plane_d;

  int n = qt_.size() / 3;
  VectorXd ret(qt_.size());
  ret.setZero();

  double k = -3e2;
  
  double min_y = V_.col(1).minCoeff();
  double max_y = V_.col(1).maxCoeff();
  double pin_y = min_y + (max_y-min_y)*0.05;
  VectorXi toforce = (V_.col(1).array() < pin_y).cast<int>(); 
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    if (toforce(i)) {
      ret.segment(3*i,3) = k*N;
    }
  }

  return M_*ret;
}

void TriObject::jacobian_regularized(SparseMatrixdRowMajor& J, bool weighted) {
  std::cerr << "jacobian_regularized not implemented for tri" << std::endl;
}
