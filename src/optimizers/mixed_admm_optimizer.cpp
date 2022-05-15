#include "mixed_admm_optimizer.h"

#include <chrono>
#include "sparse_utils.h"
#include "svd/dsvd.h"
#include "svd/svd3x3_sse.h"
#include "linesearch.h"
#include "pinning_matrix.h"

using namespace mfem;
using namespace Eigen;
using namespace std::chrono;

void MixedADMMOptimizer::step() {

  // Warm start solver
  if (config_->warm_start) {
    warm_start();
  }
  update_system();

  double kappa0 = config_->kappa;

  std::cout << "/////////////////////////////////////////////" << std::endl;
  std::cout << "Simulation step " << std::endl;

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
    bool ls_done = linesearch_x(x_, dx_);
    end = high_resolution_clock::now();
    double t2 = duration_cast<nanoseconds>(end-start).count()/1e6;

    start = high_resolution_clock::now();
    update_system();
    end = high_resolution_clock::now();
    double t3 = duration_cast<nanoseconds>(end-start).count()/1e6;
    // std::cout << "  - ! Timing Substep time: " << t1
    //     << " Linesearch: " << t2
    //     << " Update gradients: " << t3 << std::endl;

    gradient_s();
    // Solve for 's' variables
    #pragma omp parallel for
    for (int i = 0; i < nelem_; ++i) {
      ds_.segment(6*i,6) = Hs_[i].inverse() * gs_.segment(6*i,6);
    }
    linesearch_s(s_, ds_);

    //double relative_error = std::fabs(m_objectives[m_iter+1] - m_objectives[m_iter]) / (std::fabs(m_objectives[m_iter+1]) + 1.0);  // the 1.0 was added to denominator for better handling of zero-energy solutions


    update_system();
    
    update_constraints(grad_norm);

    double E0 = E_prev_;
    E_prev_ = energy(x_, s_, la_);
    double relative_obj = std::abs(E_prev_ - E0) / std::abs(E_prev_ + 1.0);
    std::cout << "  - Objective Residual: " << relative_obj << std::endl;
    if (ls_done) {
      std::cout << "  - Linesearch done " << std::endl;
      // break;
    }
    ++i;
  } while (i < config_->outer_steps && grad_norm > config_->newton_tol);

  config_->kappa = kappa0;
  update_configuration();
}


void MixedADMMOptimizer::build_lhs() {
  double h2 = config_->h * config_->h;
  double k = config_->kappa;

  auto start = high_resolution_clock::now();
  G_ = WhatS_.transpose() * J2_ - J2_;
  L_ = (P_* (G_.transpose() * A_ * G_) * P_.transpose());
  Hx_ = M_ + h2 * config_->kappa * L_;
  //SparseMatrixd L = P_* (J_.transpose() * A_ * J_) * P_.transpose();
  // J_tilde_ = 1*h2*(1*k*(Whate_.transpose()*A_*J2_ + W_.transpose() * A_ * G_)
  //     - WhatL_.transpose() * A_ * J2_) * P_.transpose();
  auto end = high_resolution_clock::now();
  double t_1 = duration_cast<nanoseconds>(end-start).count()/1e6;

  Vector6d tmp; tmp << 1,1,1,2,2,2;
  Matrix6d WTW = tmp.asDiagonal();
  start = high_resolution_clock::now();
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment(6*i,6);
    g_[i] = object_->material_->gradient(si);
    Matrix6d H = object_->material_->hessian(si);
    Hs_[i] = vols_[i]*h2*(H + config_->kappa*WTW);
  }
  end = high_resolution_clock::now();
  double t_2 = duration_cast<nanoseconds>(end-start).count()/1e6;
  // std::cout << "  - Timing LHS [1]: " << t_1 << " [2]: " << t_2 << std::endl;
}

void MixedADMMOptimizer::gradient_x() {
  //int sz = x_.size() + nelem_*6;
  gx_.resize(x_.size());
  gx_.setZero();

  // Positional forces 
  double h = config_->h;
  double h2 = h*h;
  gx_.segment(0, x_.size()) = -M_*(x_ - x0_ - h*vt_ - h*h*f_ext_);

  VectorXd reg_x(9*nelem_);
  VectorXd def_grad = J_*(P_.transpose()*x_+b_);

  // Lagrange multiplier forces
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    Matrix<double,9,6> W;
    Wmat(R_[i],W);
    const Vector6d& s = s_.segment(6*i,6);
    Vector9d diff = W*s - def_grad.segment(9*i,9);
    reg_x.segment(9*i,9) = diff;
  }

  gx_.segment(0, x_.size()) += -h2 * P_ * (Jw_.transpose() 
     - Jw_.transpose() * WhatS_) * (la_ - config_->kappa*reg_x);
}

void MixedADMMOptimizer::gradient_s() {
  gs_.resize(nelem_*6);
  gs_.setZero();

  // Positional forces 
  double h = config_->h;
  double h2 = h*h;

  VectorXd def_grad = J_*(P_.transpose()*x_+b_);

  // Lagrange multiplier forces
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    Matrix<double,9,6> W;
    Wmat(R_[i],W);
    const Vector9d& la = la_.segment(9*i,9);
    const Vector6d& s = s_.segment(6*i,6);

    Vector9d diff = W*s - def_grad.segment(9*i,9);
    gs_.segment(6*i, 6) = vols_(i) * h2 * (W.transpose()*la
        - g_[i] - config_->kappa * W.transpose()*diff);
  }
}

void MixedADMMOptimizer::update_rotations() {
  VectorXd def_grad = J_*(P_.transpose()*x_+b_);

  int N = (nelem_ / 4) + int(nelem_ % 4 != 0);

  #pragma omp parallel for 
  for (int ii = 0; ii < N; ++ii) {
    Matrix<float,12,3> F4,R4,U4,V4;
    Matrix<float,12,1> S4;
    // SSE implementation operates on 4 matrices at a time, so assemble
    // 12 x 3 matrices
    for (int jj = 0; jj < 4; ++jj) {
      int i = ii*4 +jj;
      if (i >= nelem_)
        break;
      Matrix3d f4 = Map<Matrix3d>(def_grad.segment(9*i,9).data());
      F4.block(3*jj, 0, 3, 3) = f4.cast<float>();
    }
 
    // Solve rotations
    //polar_svd3x3_sse(F4,R4);
    svd3x3_sse(F4, U4, S4, V4);

    // Assign rotations to per-element matrices
    for (int jj = 0; jj < 4; jj++) {
      int i = ii*4 +jj;
      if (i >= nelem_)
        break;

      std::array<Matrix3d, 9> dR_dF;

      // dsvd(F4.block(3*jj,0,3,3).cast<double>(),
      //     U4.block(3*jj,0,3,3).cast<double>(),
      //     S4.segment(3*jj,3).cast<double>(),
      //     V4.block(3*jj,0,3,3).cast<double>(),
      //     dR_dF
      // );
      // R_[i] = (U4.block(3*jj,0,3,3) 
      //     * Vt4.block(3*jj,0,3,3).transpose()).cast<double>();
      JacobiSVD<Matrix3d> svd(Map<Matrix3d>(def_grad.segment(9*i,9).data()),
          ComputeFullU | ComputeFullV);
      dsvd(Map<Matrix3d>(def_grad.segment(9*i,9).data()),
          svd.matrixU(), svd.singularValues(), svd.matrixV(), dR_dF);
      R_[i] = svd.matrixU() * svd.matrixV().transpose();

      // Matrix3d S = svd.matrixV() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
      // Vector6d stmp; stmp << S(0,0), S(1,1), S(2,2), S(1,0), S(2,0), S(2,1);
      // s_.segment(6*i, 6) = stmp;

      Matrix<double, 9, 6> What;
      Matrix<double, 9, 6> W;
      Wmat(R_[i] , W);
      const Vector6d& s = s_.segment(6*i,6);
      Vector9d error = W*s - def_grad.segment(9*i,9);

      for (int kk = 0; kk < 9; ++kk) {
        Wmat(dR_dF[kk] , What);
        dRS_[i].row(kk) = (What * s).transpose();
        dRL_[i].row(kk) = (What.transpose() * la_.segment(9*i,9)).transpose();
        dRe_[i].row(kk) = (What.transpose() * error).transpose();
      }
    }
  }
}

void MixedADMMOptimizer::update_system() {
  
  auto start = high_resolution_clock::now();
  // Compute rotations and rotation derivatives
  update_rotations();
  auto end = high_resolution_clock::now();
  double t_1 = duration_cast<nanoseconds>(end-start).count()/1e6;

  // Assemble rotation derivatives into block matrices
  start = high_resolution_clock::now();
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
  end = high_resolution_clock::now();
  double t_2 = duration_cast<nanoseconds>(end-start).count()/1e6;

  // Assemble blocks for left hand side
  start = high_resolution_clock::now();
  build_lhs();
  end = high_resolution_clock::now();
  double t_3 = duration_cast<nanoseconds>(end-start).count()/1e6;
  // std::cout << "  - Timing [update rotations]: " << t_1 
  //     << " [update blocks]: " << t_2 << " [LHS]: " << t_3 << std::endl;
}

void MixedADMMOptimizer::substep(bool init_guess, double& decrement) {
  double t_rhs = 0;
  double t_solve = 0;
  double t_SR = 0;
  auto start = high_resolution_clock::now();
  gradient_x();
  auto end = high_resolution_clock::now();
  t_rhs += duration_cast<nanoseconds>(end-start).count()/1e6;
  start = end;


  // if (init_guess) {
  //   dx_ds_ = solver_.solve(rhs_);
  // }
  start=end;

  // Solve used for preconditioner
  #if defined(SIM_USE_CHOLMOD)
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrixd> solver(Hx_);
  #else
  Eigen::SimplicialLLT<Eigen::SparseMatrixd> solver(Hx_);
  #endif
  if(solver.info()!=Success) {
   std::cerr << "!!!!!!!!!!!!!!!prefactor failed! " << std::endl;
   std::cout <<"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" << std::endl;
   exit(1);
  }
  dx_ = solver.solve(gx_);
  //dx_ds_ = solver_.solve(rhs_);
  //int niter = pcg(dx_ds_, lhs_ , rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_, solver_);
  // ConjugateGradient<SparseMatrix<double>, Lower|Upper> cg;
  // cg.compute(lhs_);
  // dx_ds_ = cg.solve(rhs_);
  // int niter = cg.iterations();
  // std::cout << "  - CG iters: " << niter << std::endl;
  //std::cout << "estimated error: " << cg.error()      << std::endl;
  //double relative_error = (lhs_*dx_ds_ - rhs_).norm() / rhs_.norm(); // norm() is L2 norm

  decrement = dx_.norm(); // if doing "full newton use this"
  //decrement = rhs_.norm();
  //std::cout << "  - # PCG iter: " << niter << std::endl;
  std::cout << "  - RHS Norm: " << gx_.norm() << std::endl;
  std::cout << "  - Newton decrement: " << decrement  << std::endl;

  end = high_resolution_clock::now();
  t_solve += duration_cast<nanoseconds>(end-start).count()/1e6;
  
  // std::cout << "  - Timing Substep [RHS]: " << t_rhs << " [Solve]: "
  //     << t_solve << std::endl;
}

void MixedADMMOptimizer::warm_start() {
  double h = config_->h;
  dx_ = h*vt_ + h*h*f_ext_;
  x_ += dx_;
}


void MixedADMMOptimizer::update_constraints(double residual) {
  VectorXd def_grad = J_*(P_.transpose()*x_+b_);

  // Evaluate constraint
  VectorXd dl = W_*s_ - def_grad;

  residual /= config_->h;
  double constraint_residual = dl.lpNorm<Infinity>();
  double constraint_tol = config_->constraint_tol;
  double update_zone_tol = config_->update_zone_tol; 

  // If residual from lagrangian minimization is below some tolerance
  // and constraint violation is above another tolerance, first
  // try increasing the kappa value. If kappa is already at the maximum,
  // then update lagrange multipliers.
  if (residual < update_zone_tol && constraint_residual > constraint_tol) {
    if (config_->kappa  < config_->max_kappa) {
      config_->kappa *= 2;
      config_->kappa = std::min(config_->kappa, config_->max_kappa);
    } else {
      la_ -= config_->kappa * dl;
    }
  }

  // TODO logger
  std::cout << "  [Update Lambda] constraint res: " << constraint_residual 
      << " residual: " << residual << std::endl;
  std::cout << "    LA norm: " << la_.norm() << " kappa: "
      << config_->kappa << std::endl;
}


double MixedADMMOptimizer::energy(const VectorXd& x, const VectorXd& s,
        const VectorXd& la) {
  double h = config_->h;
  double h2 = h*h;
  VectorXd xdiff = x - x0_ - h*vt_ - h*h*f_ext_;
  double Em = 0.5*xdiff.transpose()*M_*xdiff;

  VectorXd def_grad = J_*(P_.transpose()*x+b_);

  VectorXd e_R(nelem_);
  VectorXd e_L(nelem_);
  VectorXd e_Psi(nelem_);
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    Matrix3d F = Map<Matrix3d>(def_grad.segment(9*i,9).data());
    JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
    Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    Matrix<double,9,6> W;
    Wmat(R,W);
    const Vector6d& si = s.segment(6*i,6);
    Vector9d diff = W*si - def_grad.segment(9*i,9);

    e_R(i) = config_->kappa * 0.5 * diff.dot(diff) * vols_[i];
    e_L(i) = la.segment(9*i,9).dot(diff) * vols_[i];
    e_Psi(i) = object_->material_->energy(si) * vols_[i];
  }

  double Er = h2 * e_R.sum();
  double Ela = h2 * e_L.sum();
  double Epsi = h2 * e_Psi.sum();

  double e = Em + Epsi - Ela + Er;
  return e;
}

void MixedADMMOptimizer::reset() {
  // Reset variables
    // Initialize rotation matrices to identity
  nelem_ = object_->T_.rows();
  R_.resize(nelem_);
  Hs_.resize(nelem_);
  g_.resize(nelem_);
  dRS_.resize(nelem_);
  dRL_.resize(nelem_);
  dRe_.resize(nelem_);
  s_.resize(6 * nelem_);
  ds_.resize(6 * nelem_);
  ds_.setZero();

  // Make sure matrices are initially zero
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    R_[i].setIdentity();
    dRS_[i].setZero();
    dRL_[i].setZero();
    dRe_[i].setZero();
    Hs_[i].setIdentity();
    g_[i].setZero();
    s_.segment(6*i,6) = I_vec;
  }

  object_->V_ = object_->V0_;

  // Initialize lambdas
  la_.resize(9 * nelem_);
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
  tmp_r_ = dx_;
  tmp_z_ = dx_;
  tmp_p_ = dx_;
  tmp_Ap_ = dx_;


//   for (int i = 0; i < pinnedV_.size(); ++i) {
//     if (pinnedV_[i] == 1) {
//       vt_(3*i) = 1000;  
//     }
//   }
// pinnedV_.setZero();

  // Project out mass matrix pinned point
  M_ = P_ * M_ * P_.transpose();

  // External gravity force
  Vector3d ext = Map<Vector3f>(config_->ext).cast<double>();
  f_ext_ = P_ * ext.replicate(object_->V_.rows(),1);

  init_block_diagonal<9,6>(Whate_, nelem_);
  init_block_diagonal<9,6>(WhatL_, nelem_);
  init_block_diagonal<9,9>(WhatS_, nelem_);
  init_block_diagonal<9,6>(W_, nelem_);

  // Initialize volume sparse matrix
  A_.resize(nelem_*9, nelem_*9);
  std::vector<Triplet<double>> trips;
  for (int i = 0; i < nelem_; ++i) {
    for (int j = 0; j < 9; ++j) {
      trips.push_back(Triplet<double>(9*i+j, 9*i+j,vols_[i]));
    }
  }
  A_.setFromTriplets(trips.begin(),trips.end());

  // Initializing gradients and LHS
  update_system();
  
  // Compute preconditioner
  #if defined(SIM_USE_CHOLMOD)
  std::cout << "Using CHOLDMOD solver" << std::endl;
  #endif
  solver_.compute(Hx_);
  if(solver_.info()!=Success) {
    std::cerr << " KKT prefactor failed! " << std::endl;
  }
}
