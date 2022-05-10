#include "mixed_sqp_optimizer.h"

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

void MixedSQPOptimizer::step() {

  std::cout << "/////////////////////////////////////////////" << std::endl;
  std::cout << "Simulation step " << std::endl;

  int i = 0;
  double grad_norm;
  do {
    std::cout << "* Newton step: " << i << std::endl;
    update_system();
    substep(i==0, grad_norm);

    // linesearch_x(x_, dx_);
    // linesearch_s(s_, ds_);
    x_ += dx_;
    s_ += ds_;

    energy(x_, s_, la_);

    ++i;
  } while (i < config_->outer_steps && grad_norm > config_->newton_tol);

  update_configuration();
}


void MixedSQPOptimizer::build_lhs() {

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment(6*i,6);
    Matrix6d H = object_->material_->hessian(si);
    H_[i] = H*vols_[i];
  }

  // H_s
  SparseMatrixd Hs;
  init_block_diagonal<6,6>(Hs, nelem_);
  update_block_diagonal<6,6>(H_, Hs);

  size_t n = x_.size();
  size_t m = 6*nelem_;
  size_t sz = n + m + m;

  MatrixXd lhs(sz,sz);
  lhs.setZero();

  // H_x, H_s
  double h2 = config_->h * config_->h;
  lhs.block(0, 0, n, n) = M_ / h2;
  lhs.block(n, n, m, m) = MatrixXd(Hs);

  // G_x blocks
  lhs.block(0, n + m, n, m) = Gx_;
  lhs.block(n + m, 0, m, n) = Gx_.transpose();

  // G_s blocks
  lhs.block(n, n + m, m, m) = W_;
  lhs.block(n + m, n, m, m) = W_;

  // All dense right now for testing...
  lhs_ = lhs.sparseView();

  // std::cout << " LHS \n" << lhs << std::endl;
}

void MixedSQPOptimizer::build_rhs() {
  size_t n = x_.size();
  size_t m = 6*nelem_;
  rhs_.resize(n + m + m);
  rhs_.setZero();
  double h = config_->h;
  double h2 = h*h;

  // -g_x
  rhs_.segment(0, n) = -M_*(x_ - x0_ - h*vt_ - h2*f_ext_)/h2;

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment(6*i,6);

    // -g_s
    Vector6d gs = vols_[i]*object_->material_->gradient(si);
    rhs_.segment<6>(n + 6*i, 6) = -gs;

    // W * c(x^k, s^k)
    rhs_.segment<6>(n + m + 6*i, 6) = vols_[i]*(S_[i] - si);
  }
}

void MixedSQPOptimizer::update_rotations() {
  dS_.resize(nelem_);
  VectorXd def_grad = J_*(P_.transpose()*x_+b_);

  #pragma omp parallel for 
  for (int i = 0; i < nelem_; ++i) {
    JacobiSVD<Matrix3d> svd(Map<Matrix3d>(def_grad.segment(9*i,9).data()),
        ComputeFullU | ComputeFullV);

    // S(x^k)
    Matrix3d S = svd.matrixV() * svd.singularValues().asDiagonal() 
        * svd.matrixV().transpose();
    Vector6d stmp; stmp << S(0,0), S(1,1), S(2,2), S(1,0), S(2,0), S(2,1);
    S_[i] = stmp;

    // Compute SVD derivatives
    Tensor3333d dU, dV;
    Tensor333d dS;
    dsvd(dU, dS, dV, Map<Matrix3d>(def_grad.segment<9>(9*i).data()));

    // Compute dS/dF
    S = svd.singularValues().asDiagonal();
    Matrix3d V = svd.matrixV();
    std::array<Matrix3d, 9> dS_dF;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        dS_dF[3*c + r] = dV[r][c]*S*V.transpose() + V*dS[r][c].asDiagonal()*V.transpose()
            + V*S*dV[r][c].transpose();
      }
    }

    // Final jacobian should just be 6x9 since S is symmetric,
    // so extract the approach entries
    Matrix<double, 9, 9> J;
    for (int i = 0; i < 9; ++i) {
      J.col(i) = Vector9d(dS_dF[i].data());
    }
    Matrix<double, 6, 9> Js;
    Js.row(0) = J.row(0);
    Js.row(1) = J.row(4);
    Js.row(2) = J.row(8);
    Js.row(3) = J.row(1);
    Js.row(4) = J.row(2);
    Js.row(5) = J.row(5);
    // std::cout << "Js: \n" << Js << std::endl;
    // Js <<
    // 1, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 0, 1, 0, 0, 0, 0,
    // 0, 0, 0, 0, 0, 0, 0, 0, 1,
    // 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 0, 0, 0, 0, 0, 0;
    dS_[i] = Js.transpose();
  }
  
}

void MixedSQPOptimizer::update_system() {

  // Compute rotations and rotation derivatives
  update_rotations();

  // Assemble rotation derivatives into block matrices
  update_block_diagonal(dS_, C_);
  Gx_ = -P_ * J_.transpose() * C_.eval() * W_;

  // Assemble blocks for left and right hand side
  build_lhs();
  build_rhs();
}

void MixedSQPOptimizer::substep(bool init_guess, double& decrement) {
  // Factorize LHS (using SparseLU right now)
  solver_.compute(lhs_);
  if(solver_.info()!=Success) {
   std::cerr << "prefactor failed! " << std::endl;
   exit(1);
  }

  // Solve for updates
  VectorXd x = solver_.solve(rhs_);

  double relative_error = (lhs_*x - rhs_).norm() / rhs_.norm(); // norm() is L2 norm
  decrement = x.norm(); // if doing "full newton use this"
  std::cout << "  - RHS Norm: " << rhs_.norm() << std::endl;
  std::cout << "  - Newton decrement: " << decrement  << std::endl;
  std::cout << "  - relative_error: " << relative_error << std::endl;

  // Extract updates
  dx_ = x.segment(0, x_.size());
  ds_ = x.segment(x_.size(), 6*nelem_);
  la_ = x.segment(x_.size() + 6*nelem_, 6*nelem_);

  std::cout << "  - la norm: " << la_.norm() << " dx norm: "
      << dx_.norm() << " ds_.norm: " << ds_.norm() << std::endl;
}

void MixedSQPOptimizer::update_configuration() {

  vt_ = (x_ - x0_) / config_->h;
  x0_ = x_;
  la_.setZero();

  // Update mesh vertices
  VectorXd x = P_.transpose()*x_ + b_;
  MatrixXd V = Map<MatrixXd>(x.data(), object_->V_.cols(), object_->V_.rows());
  object_->V_ = V.transpose();
}

double MixedSQPOptimizer::energy(const VectorXd& x, const VectorXd& s,
        const VectorXd& la) {
  double h = config_->h;
  double h2 = h*h;
  VectorXd xdiff = x - x0_ - h*vt_ - h*h*f_ext_;
  
  double Em = (0.5/h2)*xdiff.transpose()*M_*xdiff;

  VectorXd def_grad = J_*(P_.transpose()*x+b_);

  VectorXd e_L(nelem_);
  VectorXd e_Psi(nelem_);

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    Matrix3d F = Map<Matrix3d>(def_grad.segment<9>(9*i).data());
    JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);

    Matrix3d S = svd.matrixV() * svd.singularValues().asDiagonal() 
        * svd.matrixV().transpose();
    Vector6d stmp; stmp << S(0,0), S(1,1), S(2,2), S(1,0), S(2,0), S(2,1);

    const Vector6d& si = s.segment<6>(6*i);
    Vector6d diff = stmp - si;

    e_L(i) = la.segment<6>(6*i).dot(diff) * vols_[i];
    e_Psi(i) = object_->material_->energy(si) * vols_[i];
  }

  double Ela = e_L.sum();
  double Epsi = e_Psi.sum();

  double e = Em + Epsi - Ela;
  //std::cout << "E: " <<  e << " ";
  std::cout << "  - (Em: " << Em << " Epsi: " << Epsi 
     << " Ela: " << Ela << " )" << std::endl;
  return e;
}


bool MixedSQPOptimizer::linesearch_x(VectorXd& x, const VectorXd& dx) {
 
  auto value = [&](const VectorXd& x)->double {
    return energy(x, s_, la_);
  };

  VectorXd xt = x;
  VectorXd tmp;
  SolverExitStatus status = linesearch_backtracking_bisection(xt, dx, value,
      tmp, config_->ls_iters, 1.0, 0.1, 0.5, E_prev_);
  bool done = (status == MAX_ITERATIONS_REACHED ||
              (xt-dx).norm() < config_->ls_tol);
  x = xt;
  E_prev_ = energy(xt, s_, la_);
  return done;
}

bool MixedSQPOptimizer::linesearch_s(VectorXd& s, const VectorXd& ds) {
 
  auto value = [&](const VectorXd& s)->double {
    return energy(x_, s, la_);
  };

  VectorXd st = s;
  VectorXd tmp;
  SolverExitStatus status = linesearch_backtracking_bisection(st, ds, value,
      tmp, config_->ls_iters, 1.0, 0.1, 0.5, E_prev_);
  bool done = (status == MAX_ITERATIONS_REACHED ||
              (st-ds).norm() < config_->ls_tol);
  s = st;
  E_prev_ = energy(x_, s, la_);
  return done;
}

void MixedSQPOptimizer::reset() {
  // Reset variables
    // Initialize rotation matrices to identity
  nelem_ = object_->T_.rows();
  R_.resize(nelem_);
  S_.resize(nelem_);
  H_.resize(nelem_);
  g_.resize(nelem_);
  s_.resize(6 * nelem_);
  ds_.resize(6 * nelem_);
  ds_.setZero();

  // Make sure matrices are initially zero
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    R_[i].setIdentity();
    H_[i].setIdentity();
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

  // Initializing gradients and LHS
  update_system();
  
  // Compute preconditioner
  #if defined(SIM_USE_CHOLMOD)
  std::cout << "Using CHOLDMOD solver" << std::endl;
  #endif
  solver_.compute(lhs_);
  if(solver_.info()!=Success) {
    std::cerr << " KKT prefactor failed! " << std::endl;
  }

}
