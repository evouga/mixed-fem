#include "simulation_object.h"
#include "svd/svd3x3_sse.h"
#include "pcg.h"
#include "kkt.h"
#include "pinning_matrix.h"
#include <chrono>
#include "svd/dsvd.h"
#include "linesearch.h"

using namespace std::chrono;
using namespace Eigen;
using namespace mfem;

void SimObject::build_lhs() {
  double h2 = config_->h * config_->h;
  double k = config_->kappa;

  auto start = high_resolution_clock::now();
  //SparseMatrixd F = -WhatL_.transpose() * Jw_ * P_.transpose();
  // SparseMatrixd F = -WhatL_.transpose() * A_ * J2_ * P_.transpose();
  //SparseMatrixd G = WhatS_.transpose() * J2_;
  //G = J2_ - G.eval();
  G_ = WhatS_.transpose() * J2_ - J2_;
  // SparseMatrixd Fk = Whate_.transpose()*A_*J2_ - W_.transpose() * Gw;
  // Fk = Fk * P_.transpose();
  L_ = (P_* (G_.transpose() * A_ * G_) * P_.transpose());
  Hx_ = M_ + h2 * config_->kappa * L_;
  //SparseMatrixd L = P_* (J_.transpose() * A_ * J_) * P_.transpose();
  J_tilde_ = h2*(k*(Whate_.transpose()*A_*J2_ + W_.transpose() * A_ * G_)
      - WhatL_.transpose() * A_ * J2_) * P_.transpose();
  auto end = high_resolution_clock::now();
  double t_1 = duration_cast<nanoseconds>(end-start).count()/1e6;

  VectorXd kappa_vals(T_.rows());
  Vector6d tmp; tmp << 1,1,1,2,2,2;
  Matrix6d WTW = tmp.asDiagonal();
  start = high_resolution_clock::now();
  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) {
    g_[i] = material_->gradient(R_[i], S_[i]);
    Matrix6d H = material_->hessian(S_[i]);
    // SelfAdjointEigenSolver<Matrix6d> es(H);
    // kappa_vals(i) = es.eigenvalues().real().maxCoeff();
    Hs_[i] = vols_[i]*h2*(H + config_->kappa*WTW);
  }
  end = high_resolution_clock::now();
  double t_2 = duration_cast<nanoseconds>(end-start).count()/1e6;

  start = high_resolution_clock::now();
  //SparseMatrixd J = W_.transpose()  * Jw_ * P_.transpose();
  //fill_block_matrix(A, h2 * (F + 0 * config_->kappa * Fk), H_, lhs_);
  fill_block_matrix(Hx_, J_tilde_, Hs_, lhs_);
  end = high_resolution_clock::now();
  double t_3 = duration_cast<nanoseconds>(end-start).count()/1e6;
  std::cout << "  - Timing LHS [1]: " << t_1 << " [2]: "
      << t_2 << " [3]: " << t_3 << std::endl;
  //write out preconditioner to disk
  //bool did_it_write = saveMarket(lhs, "./preconditioner.txt");
  //exit(1);

  // MatrixXd lhs(lhs_);
  // EigenSolver<MatrixXd> es(lhs);
  // std::cout << "EVALS: \n" << es.eigenvalues().real() << std::endl;
  //std::cout << "kappa vals: " << kappa_vals << std::endl;
  //config_->kappa = kappa_vals.maxCoeff();
  // MatrixXd lhs(L);
  // EigenSolver<MatrixXd> es(lhs);
  // std::cout << "J^TJ Evals: \n" << es.eigenvalues().real() << std::endl;

}

void SimObject::build_rhs() {
  int sz = qt_.size() + T_.rows()*6;
  rhs_.resize(sz);
  rhs_.setZero();

  // Positional forces 
  double h = config_->h;
  double h2 = h*h;
  rhs_.segment(0, qt_.size()) = -M_*(qt_ - q0_ - h*vt_ - h*h*f_ext_);

  VectorXd reg_x(9*T_.rows());
  VectorXd def_grad = J_*(P_.transpose()*qt_+b_);

  // Lagrange multiplier forces
  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) {
    Matrix<double,9,6> W;
    Wmat(R_[i],W);
    Vector9d la = la_.segment(9*i,9);
    Vector9d diff = W*S_[i] - def_grad.segment(9*i,9);
    rhs_.segment(qt_.size() + 6*i, 6) = vols_(i) * h2 * (W.transpose()*la
        - g_[i] - config_->kappa * W.transpose()*diff);
    reg_x.segment(9*i,9) = /*vols_(i) **/ diff;
  }

  rhs_.segment(0, qt_.size()) += -h2 * P_ * (Jw_.transpose() 
     - Jw_.transpose() * WhatS_) * (la_ - config_->kappa*reg_x);
  //rhs_.segment(0, qt_.size()) -= P_ * (Jw_.transpose() * la_);
}

// Call after outer iteration
void SimObject::update_rotations() {
  VectorXd def_grad = J_*(P_.transpose()*qt_+b_);

  int N = (T_.rows() / 4) + int(T_.rows() % 4 != 0);

  #pragma omp parallel for 
  for (int ii = 0; ii < N; ++ii) {
    Matrix<float,12,3> F4,R4,U4,V4;
    Matrix<float,12,1> S4;
    // SSE implementation operates on 4 matrices at a time, so assemble
    // 12 x 3 matrices
    for (int jj = 0; jj < 4; ++jj) {
      int i = ii*4 +jj;
      if (i >= T_.rows())
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
      if (i >= T_.rows())
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

      Matrix<double, 9, 6> What;
      Matrix<double, 9, 6> W;
      Wmat(R_[i] , W);
      Vector9d error = W*S_[i] - def_grad.segment(9*i,9);

      for (int kk = 0; kk < 9; ++kk) {
        Wmat(dR_dF[kk] , What);
        dRS_[i].row(kk) = (What * S_[i]).transpose();
        dRL_[i].row(kk) = (What.transpose() * la_.segment(9*i,9)).transpose();
        dRe_[i].row(kk) = (What.transpose() * error).transpose();
      }
    }
  }
}

void SimObject::substep(bool init_guess, double& decrement) {
  t_rhs = 0;
  t_solve = 0;
  t_SR = 0;
  auto start = high_resolution_clock::now();
  build_rhs();
  auto end = high_resolution_clock::now();
  t_rhs += duration_cast<nanoseconds>(end-start).count()/1e6;
  start = end;

  if (config_->floor_collision) {
    VectorXd f_coll = collision_force();
    rhs_.segment(0,qt_.size()) += f_coll;
    end = high_resolution_clock::now();
    t_coll += duration_cast<nanoseconds>(end-start).count()/1e6;
    start = end;
  }

  // if (init_guess) {
  //   dq_ds_ = solver_.solve(rhs_);
  // }
  start=end;

  CholmodSupernodalLLT<SparseMatrixd> solver(lhs_);
  solver.compute(lhs_);
  if(solver.info()!=Success) {
   std::cerr << "!!!!!!!!!!!!!!!prefactor failed! " << std::endl;
   std::cout <<"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" << std::endl;
   exit(1);
  }
  dq_ds_ = solver.solve(rhs_);
  //dq_ds_ = solver_.solve(rhs_);
  //int niter = pcg(dq_ds_, lhs_ , rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_, solver_);
  // ConjugateGradient<SparseMatrix<double>, Lower|Upper> cg;
  // cg.compute(lhs_);
  // dq_ds_ = cg.solve(rhs_);
  // int niter = cg.iterations();
  // std::cout << "  - CG iters: " << niter << std::endl;
  //std::cout << "estimated error: " << cg.error()      << std::endl;
  double relative_error = (lhs_*dq_ds_ - rhs_).norm() / rhs_.norm(); // norm() is L2 norm

  decrement = dq_ds_.norm(); // if doing "full newton use this"
  //decrement = rhs_.norm();
  //std::cout << "  - # PCG iter: " << niter << std::endl;
  std::cout << "  - RHS Norm: " << rhs_.norm() << std::endl;
  std::cout << "  - Newton decrement: " << decrement  << std::endl;
  std::cout << "  - relative_error: " << relative_error << std::endl;

  end = high_resolution_clock::now();
  t_solve += duration_cast<nanoseconds>(end-start).count()/1e6;
  
  // Update per-element R & S matrices
  start = high_resolution_clock::now();
  dq_ = dq_ds_.segment(0, qt_.size());
  ds_ = dq_ds_.segment(qt_.size(), 6*T_.rows());

  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) {
    S_[i] += ds_.segment(6*i,6);
  }

  end = high_resolution_clock::now();
  t_SR += duration_cast<nanoseconds>(end-start).count()/1e6;
  ibeta_ = std::min(1e-8, 0.9*ibeta_);

  std::cout << "  - Timing Substep [RHS]: " << t_rhs << " [Solve]: "
      << t_solve << " [Update vars]: " << t_SR << std::endl;
}

VectorXd SimObject::collision_force() {

  //Vector3d N(plane(0),plane(1),plane(2));
  Vector3d N(.05,.99,0);
  //Vector3d N(0.,1.,0.);
  N = N / N.norm();
  double d = config_->plane_d;

  int n = qt_.size() / 3;
  VectorXd ret(qt_.size());
  ret.setZero();

  double k = 280; //20 for octopus ssliding

  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    Vector3d xi(qt_(3*i)+dq_ds_(3*i),
        qt_(3*i+1)+dq_ds_(3*i+1),
        qt_(3*i+2)+dq_ds_(3*i+2));
    double dist = xi.dot(N);
    if (dist < config_->plane_d) {
      ret.segment(3*i,3) = k*(config_->plane_d-dist)*N;
    }
  }
  return M_*ret;
}

void SimObject::reset_variables() {
  // Initialize rotation matrices to identity
  R_.resize(T_.rows());
  S_.resize(T_.rows());
  Hs_.resize(T_.rows());
  g_.resize(T_.rows());
  dRS_.resize(T_.rows());
  dRL_.resize(T_.rows());
  dRe_.resize(T_.rows());
  for (int i = 0; i < T_.rows(); ++i) {
    R_[i].setIdentity();
    S_[i] = I_vec;
    dRS_[i].setZero();
    dRL_[i].setZero();
    dRe_[i].setZero();
    Hs_[i].setIdentity();
    g_[i].setZero();
  }
  V_ = V0_;

  // Initialize lambdas
  dq_ds_.setZero();
  dq_.setZero();
  la_.resize(9 * T_.rows());
  la_.setZero();
  ds_.resize(6 * T_.rows());
  ds_.setZero();
  vt_.setZero();

  E_prev = 0;
}

void SimObject::init() {
  reset_variables();
  volumes(vols_);
  mass_matrix(M_);
  jacobian(J_, false);
  jacobian(Jw_, true);
  J2_ = J_;

  // Pinning matrices
  double min_x = V_.col(0).minCoeff();
  double max_x = V_.col(0).maxCoeff();
  double pin_x = min_x + (max_x-min_x)*0.2;
  double min_y = V_.col(1).minCoeff();
  double max_y = V_.col(1).maxCoeff();
  double pin_y = max_y - (max_y-min_y)*0.1;
  //double pin_y = min_y + (max_y-min_y)*0.1;
  //pinnedV_ = (V_.col(0).array() < pin_x).cast<int>(); 
  pinnedV_ = (V_.col(1).array() > pin_y).cast<int>();
  //pinnedV_ = (V_.col(0).array() < pin_x && V_.col(1).array() > pin_y).cast<int>();
  //pinnedV_.resize(V_.rows());
   pinnedV_.setZero();
   pinnedV_(0) = 1;

  P_ = pinning_matrix(V_, T_, pinnedV_, false);
  P_kkt_ = pinning_matrix(V_, T_, pinnedV_, true);

  MatrixXd tmp = V_.transpose();

  qt_ = Map<VectorXd>(tmp.data(), V_.size());

  b_ = qt_ - P_.transpose()*P_*qt_;
  qt_ = P_ * qt_;
  q0_ = qt_;
  dq_ds_.resize(qt_.size() + 6*T_.rows(),1);
  dq_ds_.setZero();
  tmp_r_ = dq_ds_;
  tmp_z_ = dq_ds_;
  tmp_p_ = dq_ds_;
  tmp_Ap_ = dq_ds_;
  dq_ = 0*qt_;
  vt_ = 0*qt_;

  // Project out mass matrix pinned point
  M_ = P_ * M_ * P_.transpose();

  // External gravity force
  Vector3d ext = Map<Vector3f>(config_->ext).cast<double>();
  f_ext_ = P_ * ext.replicate(V_.rows(),1);

  init_block_diagonal<9,6>(Whate_, T_.rows());
  init_block_diagonal<9,6>(WhatL_, T_.rows());
  init_block_diagonal<9,9>(WhatS_, T_.rows());
  init_block_diagonal<9,6>(W_, T_.rows());

  // Initialize volume sparse matrix
  A_.resize(T_.rows()*9, T_.rows()*9);
  std::vector<Triplet<double>> trips;
  for (int i = 0; i < T_.rows(); ++i) {
    for (int j = 0; j < 9; ++j) {
      trips.push_back(Triplet<double>(9*i+j, 9*i+j,vols_[i]));
    }
  }
  A_.setFromTriplets(trips.begin(),trips.end());

  // Initializing gradients and LHS
  update_gradients();
  
  // Compute preconditioner
  #if defined(SIM_USE_CHOLMOD)
  std::cout << "Using CHOLDMOD solver" << std::endl;
  #endif
  solver_.compute(lhs_);
  if(solver_.info()!=Success) {
    std::cerr << " KKT prefactor failed! " << std::endl;
  }
}

void SimObject::warm_start() {
  double h = config_->h;
  dq_ds_.segment(0, qt_.size()) =  h*vt_ + h*h*f_ext_;
  dq_ = h*vt_ + h*h*f_ext_;
  ibeta_ = 1. / config_->beta;

  qt_ += dq_;

  // VectorXd def_grad = J_*(P_.transpose()*dq_+b_);
  // for (int i = 0; i < T_.rows(); ++i) {
  //   Matrix3d F = Map<Matrix3d>(def_grad.segment(9*i,9).data());
  //   JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
  //   Matrix3d dS = svd.matrixV() * svd.singularValues().asDiagonal()
  //       * svd.matrixV().transpose();
  //   Vector6d ds;
  //   ds << dS(0,0), dS(1,1), dS(2,2), dS(1,0), dS(2,0), dS(2,1);
  //   S_[i] += ds;
  // }
}

bool SimObject::linesearch(VectorXd& q, const VectorXd& dq) {
  std::vector<Vector6d> s(S_.size());
 
  auto value = [&](const VectorXd& x)->double {
    return energy(x, S_, la_);
  };

  VectorXd xt = q;
  VectorXd tmp;
  SolverExitStatus status = linesearch_backtracking_bisection(xt, dq, value,
      tmp, config_->ls_iters, 1.0, 0.1, 0.5, E_prev);
  bool done = (status == MAX_ITERATIONS_REACHED ||
              (xt-dq).norm() < config_->ls_tol);
  q = xt;
  E_prev = energy(xt, S_, la_);
  return done;
}

bool SimObject::linesearch() {
  std::cout << "  - dq norm: " << dq_.norm() 
            << " ds norm: " << ds_.norm() << std::endl;
  return linesearch(qt_, dq_);
}

void SimObject::update_gradients() {
  
  ibeta_ = 1. / config_->beta;

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
  std::vector<Matrix<double,9,6>> Wvec(T_.rows());
  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) {
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
  std::cout << "  - Time [update rotations]: " << t_1 
      << " [update blocks]: " << t_2 << " [LHS]: " << t_3 << std::endl;
}

void SimObject::update_lambdas(int t, double residual) {
  VectorXd def_grad = J_*(P_.transpose()*qt_+b_);
  
  VectorXd dl = 0 * la_;

  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) {
    Matrix<double,9,6> W;
    Wmat(R_[i],W);
    dl.segment(9*i,9) = (W*S_[i] - def_grad.segment(9*i,9));
  }

  //residual /= config_->h;
  residual *= config_->h;
  double constraint_residual = dl.lpNorm<Infinity>();
  double max_kappa = 1e5;
  double constraint_tol = 1e-2;
  // update kappa and lambda if residual below this tolerance
  double update_zone_tol = 1e-1; 
  //dl *= -std::pow(10,t+2);
  //la_ = dl;
  std::cout << "kappa: " << config_->kappa << " max: " << max_kappa << std::endl;

  if (residual < update_zone_tol && constraint_residual > constraint_tol) {
    if (config_->kappa  < max_kappa) {
      config_->kappa *=2;
    } else {
      la_ -= config_->kappa * dl;
    }
  }
  //la_ -= config_-> kappa * dl;
  std::cout << "  [Update Lambda] constraint res: " << constraint_residual 
      << " residual: " << residual << std::endl;
  std::cout << "  - LA norm: " << la_.norm() << " kappa: " << config_->kappa << std::endl;
}

void SimObject::update_positions() {

  vt_ = (qt_ - q0_) / config_->h;
  q0_ = qt_;

  la_.setZero();

  VectorXd q = P_.transpose()*qt_ + b_;
  MatrixXd tmp = Map<MatrixXd>(q.data(), V_.cols(), V_.rows());
  V_ = tmp.transpose();

}

double SimObject::energy(VectorXd x, std::vector<Vector6d> s, VectorXd la) {
  double h = config_->h;
  double h2 = h*h;
  VectorXd xdiff = x - q0_ - h*vt_ - h*h*f_ext_;
  double Em = 0.5*xdiff.transpose()*M_*xdiff;

  VectorXd def_grad = J_*(P_.transpose()*x+b_);

  VectorXd e_R(T_.rows());
  VectorXd e_L(T_.rows());
  VectorXd e_Psi(T_.rows());
  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) {
    Matrix3d F = Map<Matrix3d>(def_grad.segment(9*i,9).data());
    JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
    Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    Matrix<double,9,6> W;
    Wmat(R,W);
    Vector9d diff = W*s[i] - def_grad.segment(9*i,9);

    e_R(i) = config_->kappa * 0.5 * diff.dot(diff) * vols_[i];
    e_L(i) = la.segment(9*i,9).dot(diff) * vols_[i];
    e_Psi(i) = material_->energy(s[i]) * vols_[i];
  }

  double Er = h2 * e_R.sum();
  double Ela = h2 * e_L.sum();
  double Epsi = h2 * e_Psi.sum();

  double e = Em + Epsi - Ela + Er;
  //std::cout << "E: " <<  e << " ";
  std::cout << "  - (Em: " << Em << " Epsi: " << Epsi 
     << " Ela: " << Ela << " Er: " << Er << " )" << std::endl;
  return e;
}

