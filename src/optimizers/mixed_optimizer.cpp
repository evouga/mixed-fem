#include "mixed_optimizer.h"

#include <chrono>
#include "sparse_utils.h"
#include "svd/dsvd.h"
#include "svd/svd3x3_sse.h"
#include "linesearch.h"
#include "pinning_matrix.h"
#include "pcg.h"
// #include "linsolver/nasoq_lbl_eigen.h"
#include "svd/svd_eigen.h"

#include <fstream>
#include "unsupported/Eigen/src/SparseExtra/MarketIO.h"


using namespace mfem;
using namespace Eigen;
using namespace std::chrono;


double MixedOptimizer::primal_energy(const VectorXd& x, const VectorXd& s, 
    VectorXd& gx, VectorXd& gs) {

  double h = config_->h;
  VectorXd xt = P_.transpose()*x + b_;
  VectorXd xdiff = xt - x0_ - h*vt_ - h*h*f_ext_;
  gx = Mfull_*xdiff;

  double Em = 0.5*xdiff.dot(gx);

  VectorXd def_grad = J_*(P_.transpose()*x+b_);


  gs.resize(s.size());
  double Epsi = 0;
  #pragma omp parallel for reduction(+ : Epsi)
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s.segment<6>(6*i);
    Epsi += object_->material_->energy(si) * vols_[i];
    gs.segment<6>(6*i) = h*h*object_->material_->gradient(si) * vols_[i];
  }
  return Em + h*h*Epsi;
}
      
void MixedOptimizer::step() {
  data_.clear();

  E_prev_ = 0;
  // setup_preconditioner();

  int i = 0;
  double grad_norm;
  q_.setZero();
  q_.segment(0, x_.size()) = x_ - P_*x0_;

  do {
    data_.timer.start("step");
    update_system();
    substep(i, grad_norm);

    // TODO:
    // convergence check with norm([dx;ds])
    // data ordering for performance :)

    //linesearch_x(x_, dx_);
    //linesearch_s(s_, ds_);
    linesearch(x_, dx_, s_, ds_);

    // x_ += dx_;
    // s_ += ds_;

    double E = energy(x_, s_, la_);
    double res = std::abs((E - E_prev_) / E);
    data_.egrad_.push_back(rhs_.norm());
    data_.energies_.push_back(E);
    data_.energy_residuals_.push_back(res);
    E_prev_ = E;
    data_.timer.stop("step");


    ++i;
  } while (i < config_->outer_steps && grad_norm > config_->newton_tol);

  data_.print_data();
  update_configuration();
}

void MixedOptimizer::reset() {
  Optimizer::reset();
  // Reset variables
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

  // Initialize lambdas
  la_.resize(6 * nelem_);
  la_.setZero();
  E_prev_ = 0;
  
  object_->volumes(vols_);
  object_->mass_matrix(Mfull_, vols_);
  object_->jacobian(J_, vols_, false);

  MatrixXd tmp = object_->V_.transpose();
  x_ = Map<VectorXd>(tmp.data(), object_->V_.size());

  x0_ = x_;
  vt_ = 0*x_;

  b_ = x_ - P_.transpose()*P_*x_;
  x_ = P_ * x_;
  dx_ = 0*x_;
  q_.resize(x_.size() + 6 * nelem_);
  q_.setZero();

  // Project out mass matrix pinned point
  M_ = P_ * Mfull_ * P_.transpose();

  // External gravity force
  Vector3d ext = Map<Vector3f>(config_->ext).cast<double>();
  f_ext_ = P_.transpose()*P_*ext.replicate(object_->V_.rows(),1);
}

void MixedOptimizer::update_rotations() {
  data_.timer.start("Rot Update");

  dS_.resize(nelem_);

  VectorXd def_grad = J_*(P_.transpose()*x_+b_);

  #pragma omp parallel for 
  for (int i = 0; i < nelem_; ++i) {
    Vector3d sigma;
    Matrix3d U,V;
    svd(Map<Matrix3d>(def_grad.segment(9*i,9).data()), sigma, U, V);

    Eigen::Vector3d stemp;
    Matrix3d S = V * sigma.asDiagonal() * V.transpose();
    Vector6d stmp; stmp << S(0,0), S(1,1), S(2,2), S(1,0), S(2,0), S(2,1);
    S_[i] = stmp;
    R_[i] = U * V.transpose();

    // Compute SVD derivatives
    Tensor3333d dU, dV;
    Tensor333d dS;
    dsvd(dU, dS, dV, Map<Matrix3d>(def_grad.segment<9>(9*i).data()));

    // Compute dS/dF
    S = sigma.asDiagonal();
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
    dS_[i] = Js.transpose() * Sym;
  }
  data_.timer.stop("Rot Update");
}

bool MixedOptimizer::linesearch_x(VectorXd& x, const VectorXd& dx) {
  data_.timer.start("LS_x");
  auto value = [&](const VectorXd& x)->double {
    return energy(x, s_, la_);
  };

  VectorXd xt = x;
  VectorXd tmp = grad_.segment(0,xt.size());
  double alpha = 1.0;
  // SolverExitStatus status = linesearch_backtracking_bisection(xt, dx, value,
      // tmp, alpha, config_->ls_iters, 0.1, 0.66, E_prev_);
  SolverExitStatus status = linesearch_backtracking_cubic(xt, dx, value,
      tmp, alpha, config_->ls_iters, 1e-4, 0.5, E_prev_);  
  bool done = status == MAX_ITERATIONS_REACHED;
  if (done)
    std::cout << "linesearch_x max iters" << std::endl;
  x = xt;
  data_.timer.stop("LS_x");
  return done;
}

bool MixedOptimizer::linesearch_s(VectorXd& s, const VectorXd& ds) {
  data_.timer.start("LS_s");
  auto value = [&](const VectorXd& s)->double {
    return energy(x_, s, la_);
  };

  VectorXd st = s;
  VectorXd tmp;
  double alpha = 1.0;
  SolverExitStatus status = linesearch_backtracking_bisection(st, ds, value,
      tmp, alpha, config_->ls_iters, 0.1, 0.5, E_prev_);

  bool done = status == MAX_ITERATIONS_REACHED;
  s = st;
  data_.timer.stop("LS_s");
  return done;
}

bool MixedOptimizer::linesearch_s_local(VectorXd& s, const VectorXd& ds) {
  data_.timer.start("LS_s_local");

  double h2 = config_->h * config_->h;
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    Ref<Vector6d> la = la_.segment<6>(6*i);

    auto value = [&](const Vector6d& s)->double {
      return vols_[i] * (h2 * object_->material_->energy(s)
          - la.dot(-Sym * s));
    };

    const Vector6d& si = s.segment<6>(6*i);
    const Vector6d& dsi = ds.segment<6>(6*i);
    Vector6d gsi = grad_.segment<6>(x_.size() + 6*i);
    double alpha = 1.0;
    Vector6d st = si;
    Vector6d tmp;
    // SolverExitStatus status = linesearch_backtracking_bisection(st, dsi, value,
    //     tmp, alpha, config_->ls_iters, 0.1, 0.5, E_prev_);
    SolverExitStatus status = linesearch_backtracking_cubic(st, dsi, value,
        gsi, alpha, config_->ls_iters, 1e-4, 0.5, E_prev_);  
    bool done = (status == MAX_ITERATIONS_REACHED);
    s.segment<6>(6*i) = st;
    
  }
  data_.timer.stop("LS_s_local");
  return true;
}

bool MixedOptimizer::linesearch(Eigen::VectorXd& x, const Eigen::VectorXd& dx,
        Eigen::VectorXd& s, const Eigen::VectorXd& ds) {
  data_.timer.start("linesearch");

  auto value = [&](const VectorXd& xs)->double {
    return energy(xs.segment(0,x.size()), xs.segment(x.size(),s.size()), la_);
  };

  VectorXd f(x.size() + s.size());
  f.segment(0,x.size()) = x;
  f.segment(x.size(),s.size()) = s;

  VectorXd g(x.size() + s.size());
  g.segment(0,x.size()) = dx;
  g.segment(x.size(),s.size()) = ds;

  double alpha = 1.0;
  VectorXd grad;

  // std::cout << "grad_ norm: " << grad_.norm() << std::endl;
  // SolverExitStatus status = linesearch_backtracking_bisection(f, g, value,
  //     grad, alpha, config_->ls_iters, 0.1, 0.5, E_prev_);
  SolverExitStatus status = linesearch_backtracking_cubic(f, g, value,
      grad_, alpha, config_->ls_iters, 1e-4, 0.5, E_prev_);    
  std::cout << "ALPHA: " << alpha << std::endl;
  bool done = (status == MAX_ITERATIONS_REACHED);
  x = f.segment(0, x.size());
  s = f.segment(x.size(), s.size());
  data_.timer.stop("linesearch");
  return done;
}

void MixedOptimizer::update_configuration() {
  // Update boundary positions
  BCs_.step_script(object_, config_->h);
  #pragma omp parallel for
  for (int i = 0; i < object_->V_.rows(); ++i) {
    if (object_->is_fixed_(i)) {
      b_.segment(3*i,3) = object_->V_.row(i).transpose();
    }
  }

  VectorXd x = P_.transpose()*x_ + b_;
  vt_ = (x - x0_) / config_->h;
  x0_ = x;
  la_.setZero();

  // Update mesh vertices
  MatrixXd V = Map<MatrixXd>(x.data(), object_->V_.cols(), object_->V_.rows());
  object_->V_ = V.transpose();
}
