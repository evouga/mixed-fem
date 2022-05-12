#include "mixed_sqp_optimizer2.h"

#include <chrono>
#include "sparse_utils.h"
#include "svd/dsvd.h"
#include "svd/svd3x3_sse.h"
#include "linesearch.h"
#include "pinning_matrix.h"
#include "pcg.h"

using namespace mfem;
using namespace Eigen;
using namespace std::chrono;

void MixedSQPOptimizer2::build_lhs() {
  double ih2 = 1. / (config_->h * config_->h);

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment(6*i,6);
    Matrix6d H = object_->material_->hessian(si);
    Hinv_[i] = H.inverse();
    g_[i] = object_->material_->gradient(si);
    H_[i] = - ih2 * vols_[i] *  Sym * Hinv_[i] * Sym;
  }

  std::cout << "HX: " << std::endl;
  SparseMatrixd Hx = -Gx_.transpose() * Minv_ * Gx_;
  SparseMatrixd Hs;
  init_block_diagonal<6,6>(Hs, nelem_);
  update_block_diagonal<6,6>(H_, Hs);

  lhs_ = -(Hx + Hs);
  lhs_.makeCompressed();
  std::cout << "end lhs: " << std::endl;
  std::cout << "Hx nnz: " << Hx.nonZeros() << std::endl;
  std::cout << "Gx_ nnz: " << Gx_.nonZeros() << " Gx R C : " << Gx_.size() << std::endl;
  std::cout << "Minv nnz: " << Minv_.nonZeros() << std::endl;

  std::cout << "rows: " << lhs_.rows() << " R*C: " << lhs_.size() << " nnz: " << lhs_.nonZeros() << std::endl;
  // MatrixXd lhs(lhs_);
  // EigenSolver<MatrixXd> es(lhs);
  // std::cout << "LHS EVALS: \n" << es.eigenvalues().real() << std::endl;
}

void MixedSQPOptimizer2::build_rhs() {
  size_t m = 6*nelem_;
  rhs_.resize(m);
  rhs_.setZero();
  double h = config_->h;
  double h2 = h*h;

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment(6*i,6);

    // W * c(x^k, s^k) + H^-1 * g_s
    rhs_.segment<6>(6*i) = vols_[i]*Sym*(S_[i] - si + Hinv_[i] * g_[i]);
  }
  gx_ = (x_ - x0_ - h*vt_ - h2*f_ext_);
  rhs_ += Gx_.transpose() * gx_;
}


void MixedSQPOptimizer2::substep(bool init_guess, double& decrement) {
  int niter = 0;

  std::cout << "factorize lhs: " << std::endl;
  Eigen::SimplicialLLT<SparseMatrixd> solver(lhs_);
  if(solver.info()!=Success) {
   std::cerr << "!!!!!!!!!!!!!!!prefactor failed! " << std::endl;
   exit(1);
  }
  la_ = solver.solve(-rhs_);
  // niter = pcg(la_, lhs_ , rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_, solver_, 1e-8);
  // std::cout << "  - CG iters: " << niter << std::endl;
  //std::cout << "estimated error: " << cg.error()      << std::endl;
  double relative_error = (lhs_*la_ - -rhs_).norm() / rhs_.norm(); // norm() is L2 norm

  decrement = la_.norm(); // if doing "full newton use this"
  //decrement = rhs_.norm();
  //std::cout << "  - # PCG iter: " << niter << std::endl;
  std::cout << "  - RHS Norm: " << rhs_.norm() << std::endl;
  std::cout << "  - Newton decrement: " << decrement  << std::endl;
  std::cout << "  - relative_error: " << relative_error << std::endl;
  // Update per-element R & S matrices
  dx_ = -Minv_ * (Gx_ * la_)  - gx_;
  ds_.resize(6*nelem_);

  double ih2 = 1. / (config_->h * config_->h);
  #pragma omp parallel for 
  for (int i = 0; i < nelem_; ++i) {
    ds_.segment<6>(6*i) = -Hinv_[i] * (ih2 * Sym * la_.segment<6>(6*i)+ g_[i]);
  }



  std::cout << "la norm: " << la_.norm() << " dx norm: "
      << dx_.norm() << " ds_.norm: " << ds_.norm() << std::endl;
// exit(1);

}

void MixedSQPOptimizer2::reset() {
  // Reset variables
    // Initialize rotation matrices to identity
  nelem_ = object_->T_.rows();
  R_.resize(nelem_);
  S_.resize(nelem_);
  H_.resize(nelem_);
  Hinv_.resize(nelem_);
  g_.resize(nelem_);
  s_.resize(6 * nelem_);
  ds_.resize(6 * nelem_);
  ds_.setZero();

  // Make sure matrices are initially zero
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    R_[i].setIdentity();
    H_[i].setIdentity();
    Hinv_[i].setIdentity();
    g_[i].setZero();
    S_[i] = I_vec;
    s_.segment<6>(6*i) = I_vec;
  }

  object_->V_ = object_->V0_;

  // Initialize lambdas
  la_.resize(6 * nelem_);
  la_.setZero();
  E_prev_ = 0;
  
  object_->volumes(vols_);
  object_->mass_matrix(M_, vols_);
  object_->jacobian(J_, vols_, false);
  object_->jacobian(Jw_, vols_, true);
  J2_ = J_;

  // Pinning matrices
  double min_x = object_->V_.col(0).minCoeff();
  double max_x = object_->V_.col(0).maxCoeff();
  double pin_x = min_x + (max_x-min_x)*0.2;
  double min_y = object_->V_.col(1).minCoeff();
  double max_y = object_->V_.col(1).maxCoeff();
  double pin_y = max_y - (max_y-min_y)*0.1;
  //double pin_y = min_y + (max_y-min_y)*0.1;
  //pinnedV_ = (V_.col(0).array() < pin_x).cast<int>(); 
  pinnedV_ = (object_->V_.col(1).array() > pin_y).cast<int>();
  //pinnedV_ = (V_.col(0).array() < pin_x 
  //    && V_.col(1).array() > pin_y).cast<int>();
  //pinnedV_.resize(V_.rows());
  pinnedV_.setZero();
  pinnedV_(0) = 1;

  P_ = pinning_matrix(object_->V_, object_->T_, pinnedV_, false);

  MatrixXd tmp = object_->V_.transpose();
  x_ = Map<VectorXd>(tmp.data(), object_->V_.size());

  b_ = x_ - P_.transpose()*P_*x_;
  x_ = P_ * x_;
  x0_ = x_;
  dx_ = 0*x_;
  vt_ = 0*x_;
  q_.resize(x_.size() + 6 * nelem_);
  q_.setZero();


  // Project out mass matrix pinned point
  M_ = P_ * M_ * P_.transpose();

  // External gravity force
  Vector3d ext = Map<Vector3f>(config_->ext).cast<double>();
  f_ext_ = P_ * ext.replicate(object_->V_.rows(),1);

  init_block_diagonal<9,6>(C_, nelem_);


  // Initialize volume sparse matrix
  W_.resize(nelem_*6, nelem_*6);
  std::vector<Triplet<double>> trips;
  for (int i = 0; i < nelem_; ++i) {
    for (int j = 0; j < 6; ++j) {
      trips.push_back(Triplet<double>(6*i+j, 6*i+j,vols_[i]));
    }
  }
  W_.setFromTriplets(trips.begin(),trips.end());

  trips.clear();
  Gs_.resize(nelem_*6, nelem_*6);
  for (int i = 0; i < nelem_; ++i) {
    for (int j = 0; j < 6; ++j) {
      trips.push_back(Triplet<double>(6*i+j, 6*i+j,Sym(j,j)));
    }
  }
  Gs_.setFromTriplets(trips.begin(),trips.end());

  Eigen::SimplicialLDLT<Eigen::SparseMatrixd> Msolver(M_);
  SparseMatrix<double> I(M_.rows(),M_.cols());
  I.setIdentity();
  Minv_ = Msolver.solve(I);
  std::cout << "bad stuff be here" << std::endl;

  // Initializing gradients and LHS
  update_system();
  
  // Compute preconditioner
  // #if defined(SIM_USE_CHOLMOD)
  // std::cout << "Using CHOLDMOD solver" << std::endl;
  // #endif
  // solver_.compute(lhs_);
  // if(solver_.info()!=Success) {
  //   std::cerr << " KKT prefactor failed! " << std::endl;
  // }

}
