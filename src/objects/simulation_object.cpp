#include "simulation_object.h"
#include "svd/svd3x3_sse.h"
#include "pcg.h"
#include "kkt.h"
#include "pinning_matrix.h"
#include <chrono>
#include "svd/dsvd.h"

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
  int sz = qt_.size() + T_.rows()*9;
  rhs_.resize(sz);
  rhs_.setZero();

  VectorXd q = P_.transpose()*qt_+b_;

  // Positional forces 
  rhs_.segment(0, qt_.size()) = f_ext_ + config_->ih2*M_*(q0_ - q1_);

  VectorXd la_rhs;
  //if (config_->regularizer) {
    la_rhs.resize(T_.rows() * 9);
  //}


  // Lagrange multiplier forces
  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) {
    Vector9d rhs = material_->rhs(R_[i], S_[i], Hinv_[i], g_[i]);
    Matrix<double,9,6> W;
    Wmat(R_[i],W);
    rhs += W*Hinv_[i]*W.transpose()*la_.segment(9*i,9);

    if (config_->regularizer) {
      Matrix<double, 9, 6> W;
      Wmat(R_[i], W);
      rhs -= config_->kappa * W * (Hinv_[i] * W.transpose() * W * S_[i]);
      la_rhs.segment(9*i, 9) = rhs;
    } else if (!config_->local_global) {
        
      Vector6d gs = material_->gradient(R_[i], S_[i]);
      Vector9d la = la_.segment(9*i,9);
      Vector9d tmp1 =  -dRL_[i] * Hinv_[i] * gs;
      Vector9d tmp2 = (dRL_[i] * Hinv_[i] * W.transpose() + dRS_[i].transpose()
          - Matrix9d::Identity()) * la;
      la_rhs.segment(9*i, 9) = tmp1 + tmp2;

      Vector9d tmp3 = -W * (Hinv_[i] * gs - S_[i] - Hinv_[i] * W.transpose() * la);
      rhs = tmp3;
    }
    rhs_.segment(qt_.size() + 9*i, 9) = vols_(i) * rhs;

  }

  if (config_->regularizer) {
    rhs_.segment(0, qt_.size()) += config_->kappa * P_ * Jw_.transpose() 
        * (la_rhs - (J_ - J_tilde_) * q);
  } else if (!config_->local_global) {
    rhs_.segment(0, qt_.size()) += Jw_.transpose() * la_rhs;
  }


  // Jacobian term
  if (config_->regularizer) {
    rhs_.segment(qt_.size(), 9*T_.rows()) -= (Jw_ - Jw_tilde_) * q;
  } else {
    rhs_.segment(qt_.size(), 9*T_.rows()) -= Jw_*q;
  }

  if (config_->local_global) {
    //VectorXd la = la_ + dq_la_.segment(qt_.size(), 9*T_.rows());
    rhs_.segment(0,qt_.size()) -= Jw_.transpose() * la_;
  }
}

void SimObject::update_SR2() {
  VectorXd def_grad = J_*(P_.transpose()*(qt_+dq_)+b_);

  int N = (T_.rows() / 4) + int(T_.rows() % 4 != 0);

  VectorXd la = la_ + dq_la_.segment(qt_.size(), 9*T_.rows());

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
        dRL_[i].row(kk) = (What.transpose()* la.segment(9*i,9)).transpose();
      }
 

    }
  }
}

void SimObject::update_SR() {

  VectorXd def_grad = J_*(P_.transpose()*(qt_+dq_)+b_);
  VectorXd Jdq = J_*(P_.transpose()*dq_);

  int N = (T_.rows() / 4) + int(T_.rows() % 4 != 0);

  VectorXd la = la_ + dq_la_.segment(qt_.size(), 9*T_.rows());

  double fac = std::max((la.array().abs().maxCoeff() + 1e-6), 1.0);

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
      if (config_->regularizer) {
        //li = ibeta_*(la_.segment(9*i,9)/fac)
        //   + def_grad.segment(9*i,9)*config_->kappa;
        li = (la.segment(9*i,9))/fac
           + def_grad.segment(9*i,9)*config_->kappa;
        // li = def_grad.segment(9*i,9)*config_->kappa;   
      }
      
      // Update S[i] using new lambdas
      if (config_->regularizer) {
        Matrix<double, 9, 6> W;
        Wmat(R_[i], W);
        dS_[i] = material_->dS(R_[i], S_[i], la.segment(9*i,9)
            + config_->kappa * def_grad.segment(9*i,9)
            - config_->kappa * W * S_[i]
            , Hinv_[i]);
      } else if (!config_->local_global) {
        dS_[i] = material_->dS(R_[i], S_[i], la.segment(9*i,9), Hinv_[i])
               + Hinv_[i] * dRL_[i].transpose() * Jdq.segment(9*i,9);
      } else {
        dS_[i] = material_->dS(R_[i], S_[i], la.segment(9*i,9), Hinv_[i]);
      }

      Vector6d s = S_[i] + dS_[i];

      // Solve rotation matrices
      Matrix3d Cs;
      // Cs << s(0), s(5), s(4), 
      //       s(5), s(1), s(3), 
      //       s(4), s(3), s(2);
      Cs << s(0), s(3), s(4), 
            s(3), s(1), s(5), 
            s(4), s(5), s(2);       
      //Matrix3d y4 = Map<Matrix3d>(li.data()).transpose()*Cs;
      Matrix3d y4 = Map<Matrix3d>(li.data())*Cs;
      Y4.block(3*jj, 0, 3, 3) = y4.cast<float>();
      //Matrix3d R4out;
      //eigen_svd(y4, R4out);
      //R[i] = R4out;
    }

    // Solve rotations
    polar_svd3x3_sse(Y4,R4);

    // Assign rotations to per-element matrices
    for (int jj = 0; jj < 4; jj++) {
      int i = ii*4 +jj;
      if (i >= T_.rows())
        break;

      if (config_->local_global)
        R_[i] = R4.block(3*jj,0,3,3).cast<double>();
    }
  }
}

void SimObject::update_gradients() {
  
  ibeta_ = 1. / config_->beta;

  if (!config_->local_global) {
    update_SR2();
    // jacobian_rotational(Jw_rot_, true);

    // SparseMatrixd M(J_.cols(), J_.cols());
    // int sz = V_.size() + T_.rows()*9;
    // lhs_rot_.setZero();
    // lhs_rot_.resize(sz, sz);
    // std::vector<Triplet<double>> trips;
    // kkt_lhs(M, -Jw_rot_, 1, trips); 
    // lhs_rot_.setFromTriplets(trips.begin(),trips.end());
    // lhs_rot_ = P_kkt_ * lhs_rot_ * P_kkt_.transpose();
  }

  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) {
    S_[i] += dS_[i];
    dS_[i].setZero();
    if (config_->regularizer) {
      Hinv_[i] = material_->hessian_inv(R_[i],S_[i],config_->kappa);
    } else {
      Hinv_[i] = material_->hessian_inv(R_[i],S_[i]);
    }
    g_[i] = material_->gradient(R_[i], S_[i]);
  }

  if (config_->regularizer) {
    jacobian_regularized(J_tilde_, false);
    jacobian_regularized(Jw_tilde_, true);

    SparseMatrixd M_reg = Jw_.transpose() * (J_ - J_tilde_);
    int sz = V_.size() + T_.rows()*9;
    lhs_reg_.setZero();
    lhs_reg_.resize(sz, sz);
    std::vector<Triplet<double>> trips;
    kkt_lhs(M_reg, -Jw_tilde_, config_->kappa, trips); 
    lhs_reg_.setFromTriplets(trips.begin(),trips.end());
    lhs_reg_ = P_kkt_ * lhs_reg_ * P_kkt_.transpose();
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

  if (init_guess) {
    dq_la_ = solver_.solve(rhs_);
  }
  start=end;

  start = high_resolution_clock::now();
  material_->update_compliance(qt_.size(), T_.rows(), R_, Hinv_, vols_, lhs_);
  end = high_resolution_clock::now();
  t_asm += duration_cast<nanoseconds>(end-start).count()/1e6;
  start = end;

  int niter;
  if (config_->regularizer)
    niter = pcg(dq_la_, lhs_ + lhs_reg_, rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_, solver_);
  else if (!config_->local_global) 
    niter = pcg(dq_la_, lhs_ , rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_, solver_);
  else
    niter = pcg(dq_la_, lhs_, rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_, solver_);
  std::cout << "# PCG iter: " << niter << std::endl;

  end = high_resolution_clock::now();
  t_solve += duration_cast<nanoseconds>(end-start).count()/1e6;
  
  // Update per-element R & S matrices
  start = high_resolution_clock::now();
  dq_ = dq_la_.segment(0, qt_.size());
  // la_ = dq_la_.segment(qt_.size(), 9*T_.rows());
  update_SR(); // TODO ds should be initialized but ds after outer loop not inner...

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
    Vector3d xi(qt_(3*i)+dq_la_(3*i),
        qt_(3*i+1)+dq_la_(3*i+1),
        qt_(3*i+2)+dq_la_(3*i+2));
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
  dS_.resize(T_.rows());
  Hinv_.resize(T_.rows());
  g_.resize(T_.rows());
  dRS_.resize(T_.rows());
  dRL_.resize(T_.rows());
  for (int i = 0; i < T_.rows(); ++i) {
    R_[i].setIdentity();
    S_[i] = I_vec;
    dS_[i].setZero();
    Hinv_[i].setIdentity();
    g_[i].setZero();
  }
  V_ = V0_;

  // Initialize lambdas
  la_.resize(9 * T_.rows());
  la_.setZero();
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
  q1_ = qt_;
  dq_la_ = 0*qt_;
  dq_la_.resize(V_.size() + 9*T_.size(),1);
  dq_la_.setZero();
  tmp_r_ = dq_la_;
  tmp_z_ = dq_la_;
  tmp_p_ = dq_la_;
  tmp_Ap_ = dq_la_;
  dq_ = 0*qt_;

  // Initialize KKT lhs
  build_lhs();

  // Project out mass matrix pinned point
  M_ = P_ * M_ * P_.transpose();

  // External gravity force
  Vector3d ext = Map<Vector3f>(config_->ext).cast<double>();
  f_ext_ = M_ * P_ * ext.replicate(V_.rows(),1);
  f_ext0_ = P_ * ext.replicate(V_.rows(),1);
}

void SimObject::warm_start() {
  double h2 = config_->h * config_->h;
  dq_la_.segment(0, qt_.size()) = (qt_ - q0_) + h2*f_ext0_;
  ibeta_ = 1. / config_->beta;
  la_.setZero();

  update_SR();
}

void SimObject::update_positions() {
  q1_ = q0_;
  q0_ = qt_;
  qt_ += dq_;

  la_ += dq_la_.segment(qt_.size(), 9*T_.rows());
  dq_la_.setZero();

  VectorXd q = P_.transpose()*qt_ + b_;
  MatrixXd tmp = Map<MatrixXd>(q.data(), V_.cols(), V_.rows());
  V_ = tmp.transpose();

  // #pragma omp parallel for
  // for (int i = 0; i < T_.rows(); ++i) {
  //   S_[i] += dS_[i];
  //   dS_[i].setZero();
  // }

}

void SimObject::energy() {
//config_->ih2
  VectorXd tmp1 = f_ext0_ + config_->ih2*M_*(q0_ - q1_);


}
