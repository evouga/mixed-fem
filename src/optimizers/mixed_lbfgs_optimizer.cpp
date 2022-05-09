#include "mixed_lbfgs_optimizer.h"

#include <chrono>
#include "sparse_utils.h"
#include "svd/dsvd.h"
#include "svd/svd3x3_sse.h"
#include "linesearch.h"
#include "pinning_matrix.h"

using namespace mfem;
using namespace Eigen;
using namespace std::chrono;

void MixedLBFGSOptimizer::step() {

  // Warm start solver
  if (config_->warm_start) {
    warm_start();
  }

  double kappa0 = config_->kappa;

  std::cout << "/////////////////////////////////////////////" << std::endl;
  std::cout << "Simulation step " << std::endl;
  gradient_x();

  int i = 0;
  double grad_norm;
  bool ls_done;
  do {
    std::cout << "* Newton step: " << i << std::endl;
    auto start = high_resolution_clock::now();
    substep(i==0, grad_norm);
    auto end = high_resolution_clock::now();
    double t1 = duration_cast<nanoseconds>(end-start).count()/1e6;

    start = high_resolution_clock::now();
    VectorXd tmp = xt_;
    bool ls_done = linesearch(xt_, dx_);
    end = high_resolution_clock::now();
    double t2 = duration_cast<nanoseconds>(end-start).count()/1e6;
   
    // Compute rotations and rotation derivatives
    update_rotations();

    // Assemble rotation derivatives into block matrices
    update_block_diagonal(dRL_, WhatL_);
    update_block_diagonal(dRS_, WhatS_);
    update_block_diagonal(dRe_, Whate_);

    // Rotation matrices in block diagonal matrix
    std::vector<Matrix<double,9,6>> Wvec(nelem_);
    #pragma omp parallel for
    for (int i = 0; i < nelem_; ++i) {
      Wmat(R_[i],Wvec[i]);
    }
    update_block_diagonal(Wvec, W_);

    // Assemble blocks for left hand side
    hessian_s();
    gradient_s();
    // Solve for 's' variables
    #pragma omp parallel for
    for (int i = 0; i < nelem_; ++i) {
      ds_.segment(6*i,6) = Hs_[i].inverse() * gs_.segment(6*i,6);
    }
    linesearch_s(s_, ds_);

    // Update kappa and lambdas
    update_constraints(grad_norm);



    if (ls_done) {
      std::cout << "  - Linesearch done " << std::endl;
      // break;
    }


    // Performing LBFGS post step udpates
    VectorXd tmpg = -gx_;
    gradient_x();

    si_.emplace_back(xt_-tmp); // bad
    ti_.emplace_back(-gx_ - tmpg);
    rho_.emplace_back(si_.back().dot(ti_.back()));
    if(rho_.back() <= 0.0) {
      si_.pop_back();
      ti_.pop_back();
      rho_.pop_back();
    } else {
      if(si_.size() > history_) {
        si_.pop_front();
        ti_.pop_front();
        rho_.pop_front();
      }
    }
    
    ++i;
  } while (i < config_->outer_steps && grad_norm > config_->newton_tol);

  config_->kappa = kappa0;
  update_configuration();
}

void MixedLBFGSOptimizer::hessian_s() {
  double h2 = config_->h*config_->h;
  Vector6d tmp; tmp << 1,1,1,2,2,2;
  Matrix6d WTW = tmp.asDiagonal();
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment(6*i,6);
    g_[i] = object_->material_->gradient(si);
    Matrix6d H = object_->material_->hessian(si);
    Hs_[i] = vols_[i]*h2*(H + config_->kappa*WTW);
  }
}

void MixedLBFGSOptimizer::substep(bool init_guess, double& decrement) {
  VectorXd q = -gx_;
  int hsize = si_.size();
  std::vector<double> zeta(history_);
  for (int i = hsize - 1; i >= hsize - history_; -- i) {
    if (i < 0) {
      break;
    }
    zeta[i] = si_[i].dot(q) / rho_[i];
    q -= zeta[i] * ti_[i];
  }

  // Preconditioner solve
  VectorXd dx_ = solver_.solve(q);

  for (int i = hsize - history_; i < hsize; ++i) {
    if (i < 0) {
      continue;
    }
    // update search direction
    dx_ += si_[i] * (zeta[i] - ti_[i].dot(dx_)/rho_[i]);
  }

  decrement = dx_.norm(); // if doing "full newton use this"
  std::cout << "  - RHS Norm: " << gx_.norm() << std::endl;
  std::cout << "  - Newton decrement: " << decrement  << std::endl;
}


void MixedLBFGSOptimizer::reset() {
  // Reset variables
  MixedADMMOptimizer::reset();
  si_.clear();
  ti_.clear();
  rho_.clear();

}