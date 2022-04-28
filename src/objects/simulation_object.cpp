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

  std::vector<Triplet<double>> trips, trips_sim;

  SparseMatrixd precon; // diagonal preconditioner
  int sz = V_.size() + T_.rows()*9;
  kkt_lhs(M_, Jw_, config_->ih2, trips); 
  trips_sim = trips;

  // Just need the diagonal entries for the preconditioner's
  // compliance block so only initialize these.
  diagonal_compliance(vols_, material_config_->mu, V_.size(), trips);

  precon.resize(sz,sz);
  precon.setFromTriplets(trips.begin(), trips.end());
  precon = P_kkt_ * precon * P_kkt_.transpose();

  #if defined(SIM_USE_CHOLMOD)
  std::cout << "Using CHOLDMOD solver" << std::endl;
  #endif
  solver_.compute(precon);
  if(solver_.info()!=Success) {
    std::cerr << " KKT prefactor failed! " << std::endl;
  }

  //write out preconditioner to disk
  //bool did_it_write = saveMarket(lhs, "./preconditioner.txt");
  //exit(1);

  // The full compliance will be block diagonal, so initialize all the blocks
  init_compliance_blocks(T_.rows(), V_.size(), trips_sim);

  lhs_.resize(sz,sz);
  lhs_.setFromTriplets(trips_sim.begin(), trips_sim.end());
  lhs_ = P_kkt_ * lhs_ * P_kkt_.transpose();

}

void SimObject::build_rhs() {
  int sz = qt_.size() + T_.rows()*6;
  rhs_.resize(sz);
  rhs_.setZero();

  // Positional forces 
  double h = config_->h;
  double ih2 = config_->ih2;
  rhs_.segment(0, qt_.size()) = -ih2*M_*(qt_ - q0_ - h*vt_ - h*h*f_ext_);

  VectorXd reg_x(9*T_.rows());
  VectorXd def_grad = J_*(P_.transpose()*qt_+b_);

  // Lagrange multiplier forces
  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) {
    Matrix<double,9,6> W;
    Wmat(R_[i],W);
    Vector9d la = la_.segment(9*i,9);
    Vector9d diff = W*S_[i] - def_grad.segment(9*i,9);
    rhs_.segment(qt_.size() + 6*i, 6) = vols_(i) * (W.transpose()*la - g_[i]
        - config_->kappa * W.transpose()*diff);
    reg_x.segment(9*i,9) = vols_(i) * diff;
  }

  rhs_.segment(0, qt_.size()) += -P_ * (Jw_.transpose() 
     - Jw_.transpose() * WhatS_) * (la_ - config_->kappa*reg_x);
  //rhs_.segment(0, qt_.size()) -= P_ * (Jw_.transpose() * la_);
}

// Call after outer iteration
void SimObject::update_rotations() {
  VectorXd def_grad = J_*(P_.transpose()*qt_+b_);

  int N = (T_.rows() / 4) + int(T_.rows() % 4 != 0);

  #pragma omp parallel for 
  for (int ii = 0; ii < N; ++ii) {
    Matrix<float,12,3> F4,R4,U4,Vt4;
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
    svd3x3_sse(F4, U4, S4, Vt4);

    // Assign rotations to per-element matrices
    for (int jj = 0; jj < 4; jj++) {
      int i = ii*4 +jj;
      if (i >= T_.rows())
        break;
      //R_[i] = R4.block(3*jj,0,3,3).cast<double>();

      std::array<Matrix3d, 9> dR_dF;

      dsvd(F4.block(3*jj,0,3,3).cast<double>(),
          U4.block(3*jj,0,3,3).cast<double>(),
          S4.segment(3*jj,3).cast<double>(),
          Vt4.block(3*jj,0,3,3).cast<double>(),
          dR_dF
      );
      R_[i] = (U4.block(3*jj,0,3,3) 
          * Vt4.block(3*jj,0,3,3).transpose()).cast<double>();

      Matrix<double, 9, 6> What;
      for (int kk = 0; kk < 9; ++kk) {
        Wmat(dR_dF[kk] , What);
        dRS_[i].row(kk) = (What * S_[i]).transpose();
        dRL_[i].row(kk) = (What.transpose()* la_.segment(9*i,9)).transpose();
      }
 

    }
  }
}

// Only call within iter iteration
void SimObject::fit_rotations(VectorXd& dq, VectorXd& la) {

  VectorXd def_grad = J_*(P_.transpose()*(qt_+dq)+b_);

  int N = (T_.rows() / 4) + int(T_.rows() % 4 != 0);
  double fac = std::max((la.array().abs().maxCoeff() + 1e-6), 1.0);

  std::vector<Vector6d> S(S_.size());
  update_s(S, qt_+dq);

  #pragma omp parallel for 
  for (int ii = 0; ii < N; ++ii) {

    Matrix<float,12,3> Y4,R4;

    // SSE implementation operates on 4 matrices at a time, so assemble
    // 12 x 3 matrices
    for (int jj = 0; jj < 4; ++jj) {
      int i = ii*4 +jj;
      if (i >= T_.rows())
        break;
      Vector9d li = ibeta_*(la.segment(9*i,9)/fac) + def_grad.segment(9*i,9);
      Vector6d s = S[i];

      // Solve rotation matrices
      Matrix3d Cs;
      Cs << s(0), s(3), s(4), 
            s(3), s(1), s(5), 
            s(4), s(5), s(2);       
      //Matrix3d y4 = Map<Matrix3d>(li.data()).transpose()*Cs;
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

void SimObject::substep(bool init_guess) {

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

  start = high_resolution_clock::now();
  //material_->update_compliance(qt_.size(), T_.rows(), R_, Hinv_, vols_, lhs_);
  end = high_resolution_clock::now();
  t_asm += duration_cast<nanoseconds>(end-start).count()/1e6;
  start = end;

  // int niter;
  // if (!config_->local_global) 
  //   niter = pcg(dq_la_, lhs_ , rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_, solver_);
  // else
  //   niter = pcg(dq_la_, lhs_+lhs_rot_, rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_, solver_, 1e-8);

  ConjugateGradient<SparseMatrix<double>, Lower|Upper> cg;
  int n = 10000;
  cg.compute(lhs_);
  dq_ds_ = cg.solve(rhs_);
  std::cout << "#iterations:     " << cg.iterations() << std::endl;
  std::cout << "estimated error: " << cg.error()      << std::endl;
  double relative_error = (lhs_*dq_ds_ - rhs_).norm() / rhs_.norm(); // norm() is L2 norm

  //std::cout << "  - # PCG iter: " << niter << std::endl;
  std::cout << "  - RHS Norm: " << rhs_.norm() << std::endl;
  std::cout << "  - Newton decrement: " << dq_ds_.dot(dq_ds_) << std::endl;
  //std::cout << "  - relative_error: " << relative_error << std::endl;

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

  // if (config_->local_global) {
  //   fit_rotations(dq_, la_);
  // }
  end = high_resolution_clock::now();
  t_SR += duration_cast<nanoseconds>(end-start).count()/1e6;
  ibeta_ = std::min(1e-8, 0.9*ibeta_);
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
  Hinv_.resize(T_.rows());
  g_.resize(T_.rows());
  dRS_.resize(T_.rows());
  dRL_.resize(T_.rows());
  for (int i = 0; i < T_.rows(); ++i) {
    R_[i].setIdentity();
    S_[i] = I_vec;
    dRS_[i].setZero();
    dRL_[i].setZero();
    Hinv_[i].setIdentity();
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

  // Initialize KKT lhs
  build_lhs();

  // Project out mass matrix pinned point
  M_ = P_ * M_ * P_.transpose();

  // External gravity force
  Vector3d ext = Map<Vector3f>(config_->ext).cast<double>();
  f_ext_ = P_ * ext.replicate(V_.rows(),1);
}

void SimObject::warm_start() {
  double h = config_->h;
  dq_ds_.segment(0, qt_.size()) =  h*vt_ + h*h*f_ext_;
  dq_ = h*vt_ + h*h*f_ext_; // NEW
  ibeta_ = 1. / config_->beta;

  if (config_->local_global) {
  //  fit_rotations(dq_, la_);
  }

  qt_ += dq_;
  dq_ds_.setZero();
  dq_.setZero();

/////////////////
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

    svd3x3_sse(F4, U4, S4, V4);

    for (int jj = 0; jj < 4; ++jj) {
      int i = ii*4 +jj;
      if (i >= T_.rows())
        break;
      
      Matrix3d S = S4.segment(3*jj,3).cast<double>().asDiagonal();
      Matrix3d V = V4.block(3*jj,0,3,3).cast<double>();
      S = (V  * S * V.transpose());
      MatrixXd R = (U4.block(3*jj,0,3,3).cast<double>() * V.transpose());
      //S_[i] = Vector6d(S(0,0),S(1,1),S(2,2),S(1,0),S(2,0),S(2,1));
      // Matrix<double,9,6> W;
      // Wmat(R,W);
      //std::cout << "RS: \n " << R*S << " def grad: " << def_grad.segment(9*i,9) << "\n Ws: " << W*S_[i] << std::endl;
    }
  }
/////////////////

}

void SimObject::update_s(std::vector<Vector6d>& s, const VectorXd& q) {
  VectorXd Jdq = J_*(P_.transpose()*(q - qt_));
  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) {
    s[i] = S_[i] + material_->dS(R_[i], S_[i], la_.segment(9*i,9), Hinv_[i]);
    //std::cout << "g: " << material_->gradient(R_[i],S_[i]) << std::endl;

    if (!config_->local_global) {
      //std::cout << " ds 1: " << s[i]-S_[i] << " additional: " 
      //    << Hinv_[i] * dRL_[i].transpose() * Jdq.segment(9*i,9) << std::endl;
      s[i] = S_[i] + Hinv_[i] * dRL_[i].transpose() * Jdq.segment(9*i,9);
    }
  }
}


void SimObject::linesearch(VectorXd& q, const VectorXd& dq) {
  std::vector<Vector6d> s(S_.size());
 
  auto value = [&](const VectorXd& x)->double {
    //update_s(s, x);
    return energy(x, S_, la_);
  };

  VectorXd xt = q;
  VectorXd tmp;
  linesearch_backtracking_bisection(xt, dq, value, tmp,5,1.0,0.1,0.5,E_prev);
  //update_s(S_, xt);
  q = xt;
  E_prev = energy(xt, S_, la_);
}

void SimObject::linesearch() {
  std::cout << "  - dq norm: " << dq_.norm() 
            << " ds norm: " << ds_.norm() << std::endl;
  linesearch(qt_, dq_);
}


void SimObject::update_gradients() {
  
  ibeta_ = 1. / config_->beta;

  update_rotations();
    
  std::vector<Triplet<double>> trips1;
  for (int i = 0; i < T_.rows(); ++i) {
    // Assembly for the i-th lagrange multiplier matrix which
    // is associated with 4 vertices (for tetrahedra)
    for (int j = 0; j < 9; ++j) {
      // k-th vertex of the tetrahedra
      for (int k = 0; k < 6; ++k) {
        double val = dRL_[i](j, k) ;
        trips1.push_back(Triplet<double>(6*i+k, 9*i+j, val));
      }
    }
  }
  WhatL_.setZero();
  WhatL_.resize(6*T_.rows(), 9*T_.rows());
  WhatL_.setFromTriplets(trips1.begin(),trips1.end());

  trips1.clear();
  for (int i = 0; i < T_.rows(); ++i) {
    // Assembly for the i-th lagrange multiplier matrix which
    // is associated with 4 vertices (for tetrahedra)
    for (int j = 0; j < 9; ++j) {
      // k-th vertex of the tetrahedra
      for (int k = 0; k < 9; ++k) {
        double val = dRS_[i](j, k) ;
        trips1.push_back(Triplet<double>(9*i+j, 9*i+k, val));
      }
    }
  }
  WhatS_.setZero();
  WhatS_.resize(9*T_.rows(), 9*T_.rows());
  WhatS_.setFromTriplets(trips1.begin(),trips1.end());

  trips1.clear();
  for (int i = 0; i < T_.rows(); ++i) {
    Matrix<double,9,6> W;
    Wmat(R_[i],W);
    for (int j = 0; j < 9; ++j) {
      // k-th vertex of the tetrahedra
      for (int k = 0; k < 6; ++k) {
        trips1.push_back(Triplet<double>(9*i+j, 6*i+k, W(j,k)));
      }
    }
  }

  SparseMatrixdRowMajor Wk(9*T_.rows(), 6*T_.rows());
  Wk.setZero();
  Wk.setFromTriplets(trips1.begin(),trips1.end());


  MatrixXd V(T_.rows()*9, T_.rows()*9);
  V.setZero();
  for (int i = 0; i < T_.rows(); ++i) {
    for (int j = 0; j < 9; ++j) {
      V(9*i + j, 9*i + j) = vols_[i];
    }
  }
  MatrixXd F = -WhatL_ * Jw_ * P_.transpose();
  SparseMatrixdRowMajor G = WhatS_.transpose() * Jw_;
  G = Jw_ - G.eval();

  int n = qt_.size();
  int m = 6*T_.rows();
  MatrixXd lhs(n+m, n+m);
  lhs.setZero();
  //MatrixXd L = P_* (J_.transpose() * V * J_) * P_.transpose();
  MatrixXd L = P_* (G.transpose() * G) * P_.transpose();
  lhs.block(0,0,n,n) = config_->ih2*M_  + config_->kappa * L; 
  lhs.block(0,n,n,m) = F.transpose();
  lhs.block(n,0,m,n) = F;
  

  //lhs.block(0,n,n,m) -= config_->kappa * P_ * G.transpose() * Wk;
  //lhs.block(n,0,m,n) -= config_->kappa * Wk.transpose() * G * P_.transpose();

  VectorXd kappa_vals(T_.rows());
  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) {
    Hinv_[i] = material_->hessian_inv(R_[i],S_[i]);
    g_[i] = material_->gradient(R_[i], S_[i]);
    Matrix6d H = material_->hessian(S_[i]);

    Matrix<double,9,6> W;
    Wmat(R_[i],W);

    SelfAdjointEigenSolver<Matrix6d> es(H);
    kappa_vals(i) = vols_[i]*es.eigenvalues().real().maxCoeff();
    lhs.block(n+6*i,n+6*i,6,6) = vols_[i]*(H 
        + config_->kappa*W.transpose()*W);
  }
  std::cout << "kappa vals: " << kappa_vals << std::endl;
  config_->kappa = kappa_vals.maxCoeff() / 2;
  lhs_ = lhs.sparseView();
  //std::cout << "LA: \n" << la_ << std::endl;
  //std::cout << "LHS: \n" << lhs_ << std::endl;
}

void SimObject::update_lambdas(int t) {
  VectorXd def_grad = J_*(P_.transpose()*qt_+b_);
  
  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) {
    Matrix<double,9,6> W;
    Wmat(R_[i],W);
    la_.segment(9*i,9) = (W*S_[i] - def_grad.segment(9*i,9))* vols_[i];
  }
  la_ *= -std::pow(10,t+2);
  //la_.cwiseMax(0);
  std::cout << "  - LA norm: " << la_.norm() << std::endl;
}
void SimObject::update_positions() {

  vt_ = (qt_ - q0_) / config_->h;
  q0_ = qt_;

  dq_ds_.setZero(); // TODO should we do this?
  dq_.setZero();
  //la_.setZero();

  VectorXd q = P_.transpose()*qt_ + b_;
  MatrixXd tmp = Map<MatrixXd>(q.data(), V_.cols(), V_.rows());
  V_ = tmp.transpose();

}

double SimObject::energy(VectorXd x, std::vector<Vector6d> s, VectorXd la) {
//config_->ih2
  double h = config_->h;
  double h2 = config_->h * config_->h;
  VectorXd xdiff = x - q0_ - h*vt_ - h2*f_ext_;
  double Em = 0.5*config_->ih2* xdiff.transpose()*M_*xdiff;

  double Epsi = 0, Ela = 0, Er = 0;

  VectorXd def_grad = J_*(P_.transpose()*x+b_);

  for (int i = 0; i < T_.rows(); ++i) {
    Matrix3d F = Map<Matrix3d>(def_grad.segment(9*i,9).data());
    JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
    Matrix3d R = svd.matrixU() *  svd.matrixV().transpose();
    Matrix<double,9,6> W;
    Wmat(R,W);

    //std::cout << "s[i]: " << s[i].transpose() << std::endl;
    Epsi += material_->energy(s[i]) * vols_[i];
    // std::cout << "S_[i]: " << S_[i] << std::endl;
    //  std::cout << "   s[i] E_psi: " << material_->energy(s[i]) * vols_[i] << std::endl;
      //Matrix3d S = (svd.matrixV() * svd.singularValues().asDiagonal() * svd.matrixV().transpose());
      //Vector6d stmp = Vector6d(S(0,0),S(1,1),S(2,2),S(1,0),S(2,0),S(2,1));
      //std::cout << "   sSVD E_psi: " << material_->energy(stmp) * vols_[i] << std::endl;
    Vector9d diff = W*s[i] - def_grad.segment(9*i,9);
    Ela += la.segment(9*i,9).dot(diff) * vols_[i];
    Er += config_->kappa * 0.5 * diff.dot(diff) * vols_[i];
  }
  double e = Em + Epsi - Ela + Er;
  //std::cout << "E: " <<  e << " ";
  std::cout << "  - (Em: " << Em << " Epsi: " << Epsi 
      << " Ela: " << Ela << " Er: " << Er << " )" << std::endl;
  return e;

}
