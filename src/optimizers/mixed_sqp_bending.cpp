#include "mixed_sqp_bending.h"

#include <chrono>
#include "sparse_utils.h"
#include "svd/dsvd.h"
#include "svd/svd3x3_sse.h"
#include "linesearch.h"
#include "pcg.h"
#include "svd/newton_procrustes.h"
#include "energies/material_model.h"
#include "mesh/mesh.h"

#include <iomanip>
#include <fstream>
#include "unsupported/Eigen/SparseExtra"
#include "igl/edge_topology.h"
#include "igl/edge_lengths.h"
#include "igl/slice_mask.h"

using namespace mfem;
using namespace Eigen;

void MixedSQPBending::build_lhs() {
  data_.timer.start("LHS");

  double ih2 = 1. / (wdt_*wdt_*config_->h * config_->h);
  double h2 = (wdt_*wdt_*config_->h * config_->h);

  Ha_.resize(nedges_);
  Ha_inv_.resize(nedges_);
  grad_a_.resize(nedges_);
  // std::cout << "nedges_: " << nedges_ << std::endl;
  // std::cout << "a_ : "<< a_.size() << " a0_: " << a0_.size() << " l size: " << l_.size() << std::endl;
  std::vector<Eigen::Triplet<double>> trips(nedges_);
  #pragma omp parallel for
  for (int i = 0; i < nedges_; ++i) {
    Ha_inv_[i] = 1 / h2 / config_->kappa;
    grad_a_[i] = h2 * config_->kappa * (a_(i) - a0_(i));
    Ha_[i] = (1.0 / l_(i)) * h2 * config_->kappa;
    trips[i] = Triplet<double>(i,i,Ha_[i]);
  }
  SparseMatrixd Ha(PDW_.cols(), PDW_.cols());
  Ha.setFromTriplets(trips.begin(), trips.end());
  SparseMatrixdRowMajor Ha2 = PDW_ * Ha * PDW_.transpose();
  // std::cout << "PDW: " << PDW_.rows() << " PDW_.cols: " << PDW_.cols() << std::endl
  // std::cout << "Ha2 .rows() : " << Ha2.rows() << M_.rows() << std::endl;

  data_.timer.start("Hinv");
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment(6*i,6);
    Matrix6d H = h2 * object_->material_->hessian(si);
    Hinv_[i] = H.inverse();
    g_[i] = h2 * object_->material_->gradient(si);
    H_[i] = (1.0 / vols_[i]) * (Syminv * H * Syminv);
  }
  data_.timer.stop("Hinv");
  
  data_.timer.start("Local H");

  const std::vector<MatrixXd>& Jloc = object_->local_jacobians();
  std::vector<MatrixXd> Hloc(nelem_); 
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    Hloc[i] = (Jloc[i].transpose() * (dS_[i] * H_[i]
        * dS_[i].transpose()) * Jloc[i]) * (vols_[i] * vols_[i]);
  }
  data_.timer.stop("Local H");
  data_.timer.start("Update LHS");
  assembler_->update_matrix(Hloc);
  data_.timer.stop("Update LHS");

  lhs_ = M_ + assembler_->A + Ha2;
  data_.timer.stop("LHS");
  //saveMarket(lhs_, "lhs_full.mkt");
}

void MixedSQPBending::build_rhs() {
  data_.timer.start("RHS");

  rhs_.resize(x_.size());
  rhs_.setZero();
  gl_.resize(6*nelem_);
  gg_.resize(nedges_);
  
  double h = wdt_*config_->h;
  double h2 = h*h;

  VectorXd xt = P_.transpose()*x_ + b_;

  VectorXd ax;
  normals(xt, n_);
  angles(xt, n_, ax);
  #pragma omp parallel for
  for (int i = 0; i < nedges_; ++i) {
    gg_(i) = l_(i) * Ha_[i] * (ax(i) - a_(i)) + grad_a_[i];
  }

  VectorXd tmp(9*nelem_);
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment<6>(6*i);
    gl_.segment<6>(6*i) = vols_[i] * H_[i] * Sym * (S_[i] - si) + Syminv*g_[i];
    tmp.segment<9>(9*i) = dS_[i]*gl_.segment<6>(6*i);
  }

  rhs_ = -object_->jacobian() * tmp - PDW_ * gg_ - PM_*(wx_*xt + wx0_*x0_
      + wx1_*x1_ + wx2_*x2_ - h2*f_ext_);
  data_.timer.stop("RHS");

  grad_.resize(x_.size() + 6*nelem_);
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    tmp.segment<9>(9*i) = dS_[i]*la_.segment<6>(6*i);
    grad_.segment<6>(x_.size() + 6*i) = vols_[i] * (g_[i] + Sym*la_.segment<6>(6*i));
  }
  grad_.segment(0,x_.size()) = PM_*(wx_*xt + wx0_*x0_ + wx1_*x1_ + wx2_*x2_
      - h2*f_ext_) - object_->jacobian()  * tmp;
}

void MixedSQPBending::substep(int step, double& decrement) {
  int niter = 0;

  data_.timer.start("global");
  solver_.compute(lhs_);
  if(solver_.info()!=Success) {
   std::cerr << "prefactor failed! " << std::endl;
   exit(1);
  }
  dx_ = solver_.solve(rhs_);
  data_.timer.stop("global");


  data_.timer.start("local");
  Jdx_ = -object_->jacobian().transpose() * dx_;
  la_ = -gl_;

  double h2 = config_->h*config_->h;
  ArrayXd Ha = ((config_->kappa * h2)/l_.array());
  ArrayXd ga = (h2*config_->kappa)*(a_-a0_).array();

  ga_ = gg_.array() - (PDW_.transpose() * dx_).array() * Ha;
  
  // Update per-element R & S matrices
  ds_.resize(6*nelem_);

  da_.resize(nedges_);
  da_ = -(l_.array()*(ga_.array() + ga)) / Ha;
  std::cout << "da: " << da_.norm() << std::endl;
  a_ += da_;
  #pragma omp parallel for 
  for (int i = 0; i < nelem_; ++i) {
    la_.segment<6>(6*i) += H_[i] * (dS_[i].transpose() * Jdx_.segment<9>(9*i));
    ds_.segment<6>(6*i) = -Hinv_[i] * (Sym * la_.segment<6>(6*i)+ g_[i]);
  }
  data_.timer.stop("local");

  decrement = std::max(dx_.lpNorm<Infinity>(), ds_.lpNorm<Infinity>());
}

void MixedSQPBending::update_system() {

  if (!object_->fixed_jacobian()) {
    VectorXd x = P_.transpose()*x_ + b_;
    object_->update_jacobian(x);
  }

  VectorXd x = P_.transpose()*x_ + b_;

  normals(x, n_);
  std::vector<Eigen::VectorXd> tmp;
  grad_angles(x, n_, tmp); 

  // Compute rotations and rotation derivatives
  update_rotations();

  // Assemble blocks for left and right hand side
  build_lhs();
  build_rhs();

  energy(x_, s_, a_, la_, ga_);
}

double MixedSQPBending::energy(const VectorXd& x, const VectorXd& s,
    const VectorXd& a, const VectorXd& la, const Eigen::VectorXd& ga) {
  double h = wdt_*config_->h;
  double h2 = h*h;
  VectorXd xt = P_.transpose()*x + b_;
  VectorXd xdiff = P_ * (wx_*xt + wx0_*x0_ + wx1_*x1_ + wx2_*x2_ - h*h*f_ext_);
  double Em = 0.5*xdiff.transpose()*M_*xdiff;

  VectorXd def_grad;
  object_->deformation_gradient(xt, def_grad);

  VectorXd ax;
  normals(xt, n_);
  angles(xt, n_, ax);
  double el = 0;
  #pragma omp parallel for reduction( + : el)
  for (int i = 0; i < nedges_; ++i) {
    el += l_(i) * (h2 * 0.5 * config_->kappa * std::pow(a(i) - a0_(i), 2)
        - ga(i)*(ax(i) - a(i)));
  }

  double e = 0;
  //#pragma omp parallel for reduction( + : e )
  for (int i = 0; i < nelem_; ++i) {
  
    Matrix3d R = R_[i];
    newton_procrustes(R, Matrix3d::Identity(), Map<Matrix3d>(def_grad.segment<9>(9*i).data()));
    Matrix3d S = R.transpose()*Eigen::Map<Matrix3d>(def_grad.segment<9>(9*i).data());

  std::cout << "Si: \n" << S << std::endl;

    Vector6d stmp; stmp << S(0,0), S(1,1), S(2,2),
        0.5*(S(1,0) + S(0,1)),
        0.5*(S(2,0) + S(0,2)),
        0.5*(S(2,1) + S(1,2));
    const Vector6d& si = s.segment<6>(6*i);
    Vector6d diff = Sym * (stmp - si);
    e += h2 * object_->material_->energy(si) * vols_[i]
        - la.segment<6>(6*i).dot(diff) * vols_[i];
  }
  e += (Em + el);
  return e;
}

void MixedSQPBending::normals(const VectorXd& x, MatrixXd& n) {
  const MatrixXi& T = object_->T_;
  n.resize(T.rows(), T.cols());

  #pragma omp parallel for
  for(int i = 0; i < T.rows(); ++i) {
    Matrix<double, 9, 3> N;
    Vector3d v1 = x.segment<3>(3*T(i,1)) - x.segment<3>(3*T(i,0));
    Vector3d v2 = x.segment<3>(3*T(i,2)) - x.segment<3>(3*T(i,0));
    Vector3d ni = v1.cross(v2);
    ni.normalize();
    n.row(i) = ni.transpose();
  }
}
void MixedSQPBending::angles(const VectorXd& x, const MatrixXd& n,
    VectorXd& a) {
  // std::cout << "EF: " << EF_ << std::endl;
  a.resize(nedges_);
  #pragma omp parallel for
  for (int i = 0; i < nedges_; ++i) {
    const RowVector3d& n1 = n.row(EF_(i,0));
    const RowVector3d& n2 = n.row(EF_(i,1));
    a(i) = n1.dot(n2);
  }
}

void MixedSQPBending::grad_angles(const VectorXd& x, const MatrixXd& N,
    std::vector<VectorXd> da) {
  da.resize(nedges_);
  Dx_.resize(nedges_, J_.cols());
  auto cross_product_mat = [](const RowVector3d& v)-> Matrix3d {
    Matrix3d mat;
    mat <<     0, -v(2),  v(1),
            v(2),     0, -v(0),
           -v(1),  v(0),     0;
    return mat;
  };

// std::cout << "grad angles: " << std::endl; return;
  // a(i) = n1.dot(n2)
  // da/dx = dn1/dx * n2 + dn2/dx * n1
  std::vector<Triplet<double>> trips(nedges_*18);

  #pragma omp parallel for
  for(int i = 0; i < nedges_; ++i) {
    for (int j = 0; j < 2; ++j) {
      const RowVector3i& T = object_->T_.row(EF_(i,j));
      Vector3d v1 = x.segment<3>(3*T(1)) - x.segment<3>(3*T(0));
      Vector3d v2 = x.segment<3>(3*T(2)) - x.segment<3>(3*T(0));
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

      const RowVector3d& other = N.row(EF_(i,(j+1)%2));
      Vector9d da_dx = (other * ((Matrix3d::Identity() - n*n.transpose()) 
          * dn_dq / l)).transpose();
      for (int k =0; k < 3; ++k) {
        trips[18*i + 9*j + 3*k + 0] = Triplet<double>(i,3*T(k)+0,da_dx(3*k+0));
        trips[18*i + 9*j + 3*k + 1] = Triplet<double>(i,3*T(k)+1,da_dx(3*k+1));
        trips[18*i + 9*j + 3*k + 2] = Triplet<double>(i,3*T(k)+2,da_dx(3*k+2));
      }

    }
  }
  Dx_.setFromTriplets(trips.begin(),trips.end());
  PDW_ = P_ * Dx_.transpose() * L_;

}

void MixedSQPBending::reset() {


  igl::edge_topology(object_->V0_, object_->T_, EV_, FE_, EF_);

  ArrayXX<bool> valid = (EF_.col(0).array() != -1 && EF_.col(1).array() != -1);
  MatrixXi tmp1,tmp2;
  std::cout << "1" << std::endl;
  igl::slice_mask(EV_, valid, Array<bool,2,1>::Ones(), tmp1);
  igl::slice_mask(EF_, valid, Array<bool,2,1>::Ones(), tmp2);
    std::cout << "1" << std::endl;

  EV_ = tmp1;
  //FE_ = tmp2;
  EF_ = tmp2;
  igl::edge_lengths(object_->V0_, EV_, l_);

  nedges_ = l_.size(); 
  std::cout << "nedges: " << nedges_ << std::endl;

  // Initialize volume sparse matrix
  L_.resize(nedges_, nedges_);
  std::vector<Triplet<double>> trips;
  for (int i = 0; i < nedges_; ++i) {
    trips.push_back(Triplet<double>(i, i, l_(i)));
  }
  L_.setFromTriplets(trips.begin(),trips.end());

  n_.resize(object_->T_.rows(), 3);
  a_.resize(nedges_);
  a0_.resize(nedges_);
  ga_.resize(nedges_);

  MixedSQPPDOptimizer::reset();

  VectorXd xt = P_.transpose()*x_+b_;
  normals(xt, n_);
  angles(xt, n_, a0_);
  a_ = a0_;
  ga_.setZero();
  std::cout << "nedges: " << nedges_ << std::endl;
}
