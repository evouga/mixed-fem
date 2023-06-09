#include "tri_mesh.h"
#include <igl/doublearea.h>
#include "linear_tri3dmesh_dphi_dX.h"
#include "svd/svd3x3_sse.h"
#include "config.h"
#include "energies/material_model.h"

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

TriMesh::TriMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    const Eigen::MatrixXd& N,
    std::shared_ptr<MaterialModel> material)
    : Mesh(V,T,material), N_(N) {
  sim::linear_tri3dmesh_dphi_dX(dphidX_, Vref_, T_);
}

void TriMesh::volumes(Eigen::VectorXd& vol) {
  igl::doublearea(Vref_, T_, vol);
  vol.array() *= (material_->config()->thickness/2);
}

void TriMesh::mass_matrix(Eigen::SparseMatrixdRowMajor& M,
    const VectorXd& vols) {
  std::vector<Triplet<double>> trips;

  // 1. Mass matrix terms
  for (int i = 0; i < T_.rows(); ++i) {
    for (int j = 0; j < 3; ++j) {
      int id1 = T_(i,j);
      for (int k = 0; k < 3; ++k) {
        int id2 = T_(i,k);
        double val = vols(i);
        if (j == k)
          val /= 6;
        else
          val /= 12;
        trips.push_back(Triplet<double>(3*id1+0,3*id2,val));
        trips.push_back(Triplet<double>(3*id1+1,3*id2+1,val));
        trips.push_back(Triplet<double>(3*id1+2,3*id2+2,val));
      }
    }
  }
  M.resize(V_.size(),V_.size());
  M.setFromTriplets(trips.begin(),trips.end());

  // note: assuming uniform density 
  M = M * material_->config()->density ; 
}

void TriMesh::init_jacobian() {
  Jloc_.resize(T_.rows());

  std::vector<Triplet<double>> trips;
  for (int i = 0; i < T_.rows(); ++i) { 

    // Local block
    Matrix9d B;
    Matrix3d dX = sim::unflatten<3,3>(dphidX_.row(i));
    local_jacobian(B, dX);

    Jloc_[i] = B;

    for (int j = 0; j < 9; ++j) {

      // k-th vertex of the tetrahedra
      for (int k = 0; k < 3; ++k) {
        int vid = T_(i,k); // vertex index

        // x,y,z index for the k-th vertex
        for (int l = 0; l < 3; ++l) {
          double val = B(j,3*k+l);
          trips.push_back(Triplet<double>(9*i+j, 3*vid+l, val));
        }
      }
    }
  }
  J_.resize(9*T_.rows(), V_.size());
  J_.setFromTriplets(trips.begin(),trips.end());  
  J0_ = J_;
  Jloc0_ = Jloc_;
}

void TriMesh::deformation_gradient(const VectorXd& x, VectorXd& F) {
  assert(x.size() == J_.cols());
  F = J0_ * x;

  #pragma omp parallel for 
  for (int i = 0; i < T_.rows(); ++i) {
      Matrix<double, 9, 3> N;
      N << N_(i,0), 0, 0,
           0, N_(i,0), 0,
           0, 0, N_(i,0),
           N_(i,1), 0, 0,
           0, N_(i,1), 0,
           0, 0, N_(i,1),
           N_(i,2), 0, 0,
           0, N_(i,2), 0,
           0, 0, N_(i,2);
      Vector3d v1 = x.segment<3>(3*T_(i,1)) - x.segment<3>(3*T_(i,0));
      Vector3d v2 = x.segment<3>(3*T_(i,2)) - x.segment<3>(3*T_(i,0));
      Vector3d n = v1.cross(v2);
      n.normalize();
      F.segment<9>(9*i) += N*n;
  }
}

void TriMesh::update_jacobian(const VectorXd& x) {
  assert(x.size() == J_.cols());

  Jloc_.resize(T_.rows());

  auto cross_product_mat = [](const RowVector3d& v)-> Matrix3d {
    Matrix3d mat;
    mat <<     0, -v(2),  v(1),
            v(2),     0, -v(0),
           -v(1),  v(0),     0;
    return mat;
  };

  std::vector<Triplet<double>> trips;

  // #pragma omp parallel for
  for(int i = 0; i < T_.rows(); ++i) {
    Matrix<double, 9, 3> N;
    N << N_(i,0), 0, 0,
         0, N_(i,0), 0,
         0, 0, N_(i,0),
         N_(i,1), 0, 0,
         0, N_(i,1), 0,
         0, 0, N_(i,1),
         N_(i,2), 0, 0,
         0, N_(i,2), 0,
         0, 0, N_(i,2);
    Vector3d v1 = x.segment<3>(3*T_(i,1)) - x.segment<3>(3*T_(i,0));
    Vector3d v2 = x.segment<3>(3*T_(i,2)) - x.segment<3>(3*T_(i,0));
    Vector3d n = v1.cross(v2);
    double l = n.norm();
    n /= l;

    Matrix3d dx1 = cross_product_mat(v1);
    Matrix3d dx2 = cross_product_mat(v2);

    Matrix<double, 3, 9> dn_dq;
    dn_dq.setZero();
    dn_dq.block<3,3>(0,0) = dx2 - dx1;
    dn_dq.block<3,3>(0,3) = -dx2;
    dn_dq.block<3,3>(0,6) = dx1;
    Jloc_[i] = Jloc0_[i]
        + N * (Matrix3d::Identity() - n*n.transpose()) * dn_dq / l;

    
    for (int j = 0; j < 9; ++j) {

      // k-th vertex of the tetrahedra
      for (int k = 0; k < 3; ++k) {
        int vid = T_(i,k); // vertex index

        // x,y,z index for the k-th vertex
        for (int l = 0; l < 3; ++l) {
          double val = Jloc_[i](j,3*k+l);
          trips.push_back(Triplet<double>(9*i+j, 3*vid+l, val));
        }
      }
    }     
  }
  J_.resize(9*T_.rows(), V_.size());
  J_.setFromTriplets(trips.begin(),trips.end());
  PJW_ = P_ * J_.transpose() * W_;
}
