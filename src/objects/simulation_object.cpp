#include "simulation_object.h"
#include "svd/svd3x3_sse.h"
#include "pcg.h"
#include "kkt.h"
#include "pinning_matrix.h"
#include <chrono>

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

  // Positional forces 
  rhs_.segment(0, qt_.size()) = f_ext_ + config_->ih2*M_*(q0_ - q1_);

  // Lagrange multiplier forces
  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) {
    rhs_.segment(qt_.size() + 9*i, 9) = vols_(i) * material_->rhs(R_[i],
        S_[i], Hinv_[i], g_[i]);
  }

  // Jacobian term
  rhs_.segment(qt_.size(), 9*T_.rows()) -= Jw_*(P_.transpose()*qt_+b_);
}

void SimObject::update_SR() {

  VectorXd def_grad = J_*(P_.transpose()*(qt_+dq_)+b_);

  int N = (T_.rows() / 4) + int(T_.rows() % 4 != 0);

  double fac = std::max((la_.array().abs().maxCoeff() + 1e-6), 1.0);

  #pragma omp parallel for 
  for (int ii = 0; ii < N; ++ii) {

    Matrix<float,12,3> Y4,R4;

    // SSE implementation operates on 4 matrices at a time, so assemble
    // 12 x 3 matrices
    for (int jj = 0; jj < 4; ++jj) {
      int i = ii*4 +jj;
      if (i >= T_.rows())
        break;
      Vector9d li = ibeta_*(la_.segment(9*i,9)/fac) + def_grad.segment(9*i,9);
      
      // Update S[i] using new lambdas
      dS_[i] = material_->dS(R_[i], S_[i], la_.segment(9*i,9), Hinv_[i]);
      Vector6d s = S_[i] + dS_[i];

      // Solve rotation matrices
      Matrix3d Cs;
      Cs << s(0), s(5), s(4), 
            s(5), s(1), s(3), 
            s(4), s(3), s(2); 
      Matrix3d y4 = Map<Matrix3d>(li.data()).transpose()*Cs;
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
      R_[i] = R4.block(3*jj,0,3,3).cast<double>();
    }
  }
}

void SimObject::update_gradients() {
  
  ibeta_ = 1. / config_->beta;

  #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) {
    Hinv_[i] = material_->hessian_inv(R_[i],S_[i]);
    g_[i] = material_->gradient(R_[i], S_[i]);
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

  pcg(dq_la_, lhs_, rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_, solver_);
  end = high_resolution_clock::now();
  t_solve += duration_cast<nanoseconds>(end-start).count()/1e6;
  
  // Update per-element R & S matrices
  start = high_resolution_clock::now();
  dq_ = dq_la_.segment(0, qt_.size());
  la_ = dq_la_.segment(qt_.size(), 9*T_.rows());
  update_SR();

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
  for (int i = 0; i < T_.rows(); ++i) {
    R_[i].setIdentity();
    S_[i] = I_vec;
    dS_[i].setZero();
    Hinv_[i].setIdentity();
    g_[i].setZero();
  }

  // Initial lambdas
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
  double pin_x = min_x + (max_x-min_x)*0.15;
  double min_y = V_.col(1).minCoeff();
  double max_y = V_.col(1).maxCoeff();
  double pin_y = max_y - (max_y-min_y)*0.1;
  //double pin_y = min_y + (max_y-min_y)*0.1;
  //pinnedV = (V_.col(0).array() < pin_x).cast<int>(); 
  pinnedV_ = (V_.col(1).array() > pin_y).cast<int>();
  // pinnedV_.resize(V_.rows());
  // pinnedV_.setZero();
  // pinnedV_(0) = 1;

  P_ = pinning_matrix(V_, T_, pinnedV_, false);
  P_kkt_ = pinning_matrix(V_, T_, pinnedV_, true);

  MatrixXd tmp = V_.transpose();

  qt_ = Map<VectorXd>(tmp.data(), V_.size());

  b_ = qt_ - P_.transpose()*P_*qt_;
  qt_ = P_ * qt_;
  q0_ = qt_;
  q1_ = qt_;
  dq_la_ = 0*qt_;
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
  f_ext_ = M_ * P_ *Vector3d(0,config_->grav,0).replicate(V_.rows(),1);
  f_ext0_ = P_ *Vector3d(0,config_->grav,0).replicate(V_.rows(),1);
}

void SimObject::warm_start() {
  double h2 = config_->h * config_->h;
  dq_la_.segment(0, qt_.size()) = (qt_ - q0_) + h2*f_ext0_;
  ibeta_ = 1. / config_->beta;
  update_SR();
}

void SimObject::update_positions() {
  q1_ = q0_;
  q0_ = qt_;
  qt_ += dq_;
  dq_la_.setZero();

  VectorXd q = P_.transpose()*qt_ + b_;
  MatrixXd tmp = Map<MatrixXd>(q.data(), V_.cols(), V_.rows());
  V_ = tmp.transpose();
}