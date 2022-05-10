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

void MixedSQPOptimizer2::step() {

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
    substep(i==0, grad_norm);

    bool ls_done = linesearch(xt_, dx_);
    // energy(xt_, s_, la_);

    update_system();

    if (ls_done) {
      std::cout << "  - Linesearch done " << std::endl;
      // break;
    }
    ++i;
  } while (i < config_->outer_steps && grad_norm > config_->newton_tol);

  update_configuration();
}


void MixedSQPOptimizer2::build_lhs() {
  double h2 = config_->h * config_->h;

  // #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment(6*i,6);
    g_[i] = vols_[i]*h2*object_->material_->gradient(si);
    Matrix6d H = object_->material_->hessian(si);
    Hinv_[i] = H.inverse()/h2/vols_[i];
    //Hinv_[i] = vols_[i]*h2*H.inverse();
  }

  SparseMatrixd Hs;
  init_block_diagonal<6,6>(Hs, nelem_);
  update_block_diagonal<6,6>(Hinv_, Hs);
  //   MatrixXd Hs;
  // Hs.resize(6*nelem_,6*nelem_);
  // Hs.setZero();
  // int N = Hinv_.size();
  //   // #pragma omp parallel for
  // for (int i = 0; i < N; ++i) {
  //   Hs.block(6*i,6*i,6,6) = Hinv_[i];

  // }

  lhs_ = JC_.transpose() * MinvC_ + Hs;
  // lhs_ = Hs;

  SparseMatrixd reg(lhs_.rows(),lhs_.cols());
  reg.setIdentity();
  MatrixXd lhs(lhs_);
  // EigenSolver<MatrixXd> es(lhs);
  // std::cout << "LHS EVALS: \n" << es.eigenvalues().real() << std::endl;

  // lhs_ += reg;

  // lhs = MatrixXd(Hs);
  // es.compute(lhs);
  // std::cout << "Hs EVALS: \n" << es.eigenvalues().real() << std::endl;
  // std::cout << "LHS : " << lhs_ << std::endl;
}

void MixedSQPOptimizer2::build_rhs() {
  rhs_.resize(nelem_*6);
  rhs_.setZero();

  // Positional forces 
  double h = config_->h;
  double h2 = h*h;
  gx_ = M_*(xt_ - x0_ - h*vt_ - h*h*f_ext_);
  gs_.resize(nelem_*6);

  // Lagrange multiplier forces
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    Vector6d diff = S_[i] - s_.segment<6>(6*i);
    Vector6d gs = Hinv_[i] * g_[i];
    gs_.segment<6>(6*i) = gs;
    rhs_.segment<6>(6*i) =  diff + gs;
    // rhs_.segment<6>(6*i) =  diff - gs;
  }
  
  std::cout << "gs_.norm() : " << gs_.norm() << std::endl;
  // std::cout << "minv: " << MinvC_.rows() << " " << MinvC_.cols() << std::endl;
  // std::cout << "minv: " << gx_.rows() << "  rhs_" << rhs_.rows() << std::endl;
  rhs_ -= MinvC_.transpose() * gx_;
}

void MixedSQPOptimizer2::update_rotations() {
  dS_.resize(nelem_);
  VectorXd def_grad = J_*(P_.transpose()*xt_+b_);


  #pragma omp parallel for 
  for (int i = 0; i < nelem_; ++i) {

    std::array<Matrix3d, 9> dR_dF;
    JacobiSVD<Matrix3d> svd(Map<Matrix3d>(def_grad.segment(9*i,9).data()),
        ComputeFullU | ComputeFullV);

    //void dsvd(Eigen::Tensor3333x<Scalar> &dU, Eigen::Tensor333x<Scalar>  &dS, Eigen::Tensor3333x<Scalar> &dV, const Eigen::Matrix3x<UType> &U, const Eigen::Vector3x<SType> &S, const Eigen::Matrix3x<VType> &V);
    Tensor3333d dU, dV;
    Tensor333d dS;
    dsvd(dU, dS, dV, Map<Matrix3d>(def_grad.segment<9>(9*i).data()));

    Matrix3d S = svd.matrixV() * svd.singularValues().asDiagonal() 
        * svd.matrixV().transpose();
    Vector6d stmp; stmp << S(0,0), S(1,1), S(2,2), S(1,0), S(2,0), S(2,1);
    S_[i] = stmp;

    S = svd.singularValues().asDiagonal();
    Matrix3d V = svd.matrixV();
    std::array<Matrix3d, 9> dS_dF;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        dS_dF[3*c + r] = dV[r][c]*S*V.transpose() + V*dS[r][c].asDiagonal()*V.transpose()
            + V*S*dV[r][c].transpose();
      }
    }

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
    dS_[i] = Js.transpose();
  }
  
}

void MixedSQPOptimizer2::update_system() {
  
  auto start = high_resolution_clock::now();
  // Compute rotations and rotation derivatives
  update_rotations();
  auto end = high_resolution_clock::now();
  double t_1 = duration_cast<nanoseconds>(end-start).count()/1e6;

  // Assemble rotation derivatives into block matrices
  start = high_resolution_clock::now();
  update_block_diagonal(dS_, C_);
  JC_ = P_ * J_.transpose() * C_.eval();
  MinvC_ = Minv_ * JC_;
  end = high_resolution_clock::now();
  double t_2 = duration_cast<nanoseconds>(end-start).count()/1e6;

  // Assemble blocks for left and right hand side
  start = high_resolution_clock::now();
  build_lhs();
  end = high_resolution_clock::now();
  double t_3 = duration_cast<nanoseconds>(end-start).count()/1e6;
  build_rhs();

  // std::cout << "  - Timing [update rotations]: " << t_1 
      // << " [update blocks]: " << t_2 << " [LHS]: " << t_3 << std::endl;
}

bool MixedSQPOptimizer2::linesearch(VectorXd& x, const VectorXd& dx) {
 
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

void MixedSQPOptimizer2::substep(bool init_guess, double& decrement) {
  int niter = 0;
  // solver_.compute(lhs_);
  // if(solver_.info()!=Success) {
  //  std::cerr << "!!!!!!!!!!!!!!!prefactor failed! " << std::endl;
  //  exit(1);
  // }
  // la_ = solver_.solve(rhs_);
  niter = pcg(la_, lhs_ , rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_, solver_, 1e-8);
  std::cout << "  - CG iters: " << niter << std::endl;
  //std::cout << "estimated error: " << cg.error()      << std::endl;
  double relative_error = (lhs_*la_ - rhs_).norm() / rhs_.norm(); // norm() is L2 norm

  decrement = la_.norm(); // if doing "full newton use this"
  //decrement = rhs_.norm();
  //std::cout << "  - # PCG iter: " << niter << std::endl;
  std::cout << "  - RHS Norm: " << rhs_.norm() << std::endl;
  std::cout << "  - Newton decrement: " << decrement  << std::endl;
  std::cout << "  - relative_error: " << relative_error << std::endl;
  // Update per-element R & S matrices
  dx_ =- MinvC_ * la_  - Minv_ * gx_;
  ds_.resize(6*nelem_);

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    ds_.segment<6>(6*i) = -Hinv_[i]*la_.segment<6>(6*i) - gs_.segment<6>(6*i);
  }

  // xt_ += dx_;
  s_ += ds_; // currently not doing linesearch on s

  std::cout << "la norm: " << la_.norm() << " dx norm: "
      << dx_.norm() << " ds_.norm: " << ds_.norm() << std::endl;
// exit(1);

}

void MixedSQPOptimizer2::warm_start() {
  double h = config_->h;
  dx_ = h*vt_ + h*h*f_ext_;
  xt_ += dx_;
}

void MixedSQPOptimizer2::update_configuration() {

  vt_ = (xt_ - x0_) / config_->h;
  x0_ = xt_;
  la_.setZero();

  // Update mesh vertices
  VectorXd x = P_.transpose()*xt_ + b_;
  MatrixXd V = Map<MatrixXd>(x.data(), object_->V_.cols(), object_->V_.rows());
  object_->V_ = V.transpose();
}

double MixedSQPOptimizer2::energy(const VectorXd& x, const VectorXd& s,
        const VectorXd& la) {
  double h = config_->h;
  double h2 = h*h;
  VectorXd xdiff = x - x0_ - h*vt_ - h*h*f_ext_;
  
  double Em = 0.5*xdiff.transpose()*M_*xdiff;

  VectorXd def_grad = J_*(P_.transpose()*x+b_);

  VectorXd e_L(nelem_);
  VectorXd e_Psi(nelem_);
  // TODO parallel reduction ?
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    Matrix3d F = Map<Matrix3d>(def_grad.segment<9>(9*i).data());
    JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
    Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    Matrix3d S = svd.matrixV() * svd.singularValues().asDiagonal() 
        * svd.matrixV().transpose();
    Vector6d stmp; stmp << S(0,0), S(1,1), S(2,2), S(1,0), S(2,0), S(2,1);
    Matrix<double,9,6> W;
    Wmat(R,W);
    const Vector6d& si = s.segment<6>(6*i);
    Vector6d diff = stmp - si;

    e_L(i) = la.segment<6>(6*i).dot(diff) * vols_[i];
    e_Psi(i) = object_->material_->energy(si) * vols_[i];
  }

  double Ela = h2 * e_L.sum();
  double Epsi = h2 * e_Psi.sum();

  double e = Em + Epsi - Ela;
  //std::cout << "E: " <<  e << " ";
  std::cout << "  - (Em: " << Em << " Epsi: " << Epsi 
     << " Ela: " << Ela << " )" << std::endl;
  return e;
}

void MixedSQPOptimizer2::reset() {
  // Reset variables
    // Initialize rotation matrices to identity
  nelem_ = object_->T_.rows();
  R_.resize(nelem_);
  S_.resize(nelem_);
  Hinv_.resize(nelem_);
  g_.resize(nelem_);
  s_.resize(6 * nelem_);
  ds_.resize(6 * nelem_);
  ds_.setZero();

  // Make sure matrices are initially zero
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    R_[i].setIdentity();
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
  xt_ = Map<VectorXd>(tmp.data(), object_->V_.size());

  b_ = xt_ - P_.transpose()*P_*xt_;
  xt_ = P_ * xt_;
  x0_ = xt_;
  dx_ = 0*xt_;
  vt_ = 0*xt_;

  // Project out mass matrix pinned point
  M_ = P_ * M_ * P_.transpose();

  // External gravity force
  Vector3d ext = Map<Vector3f>(config_->ext).cast<double>();
  f_ext_ = P_ * ext.replicate(object_->V_.rows(),1);

  init_block_diagonal<9,6>(C_, nelem_);

  // Initialize volume sparse matrix
  A_.resize(nelem_*6, nelem_*6);
  std::vector<Triplet<double>> trips;
  for (int i = 0; i < nelem_; ++i) {
    for (int j = 0; j < 6; ++j) {
      trips.push_back(Triplet<double>(6*i+j, 6*i+j,vols_[i]));
    }
  }
  A_.setFromTriplets(trips.begin(),trips.end());


  Eigen::SimplicialLDLT<Eigen::SparseMatrixd> Msolver(M_);
  SparseMatrix<double> I(M_.rows(),M_.cols());
  I.setIdentity();
  Minv_ = Msolver.solve(I);
  std::cout << "bad stuff be here" << std::endl;

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


VectorXd MixedSQPOptimizer2::collision_force() {

  //Vector3d N(plane(0),plane(1),plane(2));
  Vector3d N(.05,.99,0);
  //Vector3d N(0.,1.,0.);
  N = N / N.norm();
  double d = config_->plane_d;

  int n = xt_.size() / 3;
  VectorXd ret(xt_.size());
  ret.setZero();

  double k = 280; //20 for octopus ssliding

  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    Vector3d xi(xt_(3*i+0)+dx_(3*i),
                xt_(3*i+1)+dx_(3*i+1),
                xt_(3*i+2)+dx_(3*i+2));
    double dist = xi.dot(N);
    if (dist < config_->plane_d) {
      ret.segment(3*i,3) = k*(config_->plane_d-dist)*N;
    }
  }
  return M_*ret;
}
