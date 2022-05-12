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


void MixedSQPOptimizer::setup_preconditioner() {

  //really soft preconditioner works better
  double mat_val = std::min(object_->config_->mu, 1e10);

  
  if(std::abs(mat_val - preconditioner_.mat_val()) > 1e-3) {

    std::cout<<"Rebuilding Preconditioner\n";
    std::vector<Matrix9d> C(nelem_);  
    #pragma parallel for
    for (int i = 0; i < nelem_; ++i) {
      
      C[i] = Eigen::Matrix9d::Identity()*(-vols_[i] / (mat_val* config_->h * config_->h));
    }
    
    SparseMatrixd P;
    SparseMatrixd J = Jw_;
    fill_block_matrix(M_, -J * P_.transpose(), C, P); 
    preconditioner_ = Eigen::CorotatedPreconditioner<double>(mat_val, P_.rows(), nelem_, P, dS_);

    //cg.preconditioner() = preconditioner_;
    //cg.setTolerance(1e-5);
    //cg.compute(lhs_);
  }

  preconditioner_.compute(lhs_);
  
}

void MixedSQPOptimizer::step() {

  setup_preconditioner();
  
  std::cout << "/////////////////////////////////////////////" << std::endl;
  std::cout << " GOOD? Simulation step " << std::endl;

  int i = 0;
  double grad_norm;
  q_.segment(P_.rows(), 6*nelem_).setZero();

  do {
    std::cout << "* Newton step: " << i << std::endl;
    update_system();
    substep(i==0, grad_norm);
    // 1. Try to doing linesearch over both, and also try only minimizing the "primal energy"
    // 2. Try removing h^2 from constraint term
    // 3. Try regular neohookean, flickering from SNH?
    // 1. Try diagonal only LHS as the preconditioner
    // 4. Find some wrapd failures
    // main objective
    // linesearch_x(x_, dx_);
    // linesearch_s(s_, ds_);

    linesearch(x_, dx_, s_, ds_);

    // x_ += dx_;
    // s_ += ds_;

    energy(x_, s_, la_);

    ++i;
  } while (i < config_->outer_steps && grad_norm > config_->newton_tol);

  update_configuration();
}

void MixedSQPOptimizer::build_lhs() {
  double ih2 = 1. / (config_->h * config_->h);

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment(6*i,6);
    Matrix6d H = object_->material_->hessian(si);
    Hinv_[i] = H.inverse();
    g_[i] = object_->material_->gradient(si);
    H_[i] = - ih2 * vols_[i] *  Sym * (Hinv_[i] + Matrix6d::Identity()*(1./(std::min(std::min(object_->config_->mu, object_->config_->la), 1e10)))) * Sym;
  }

  fill_block_matrix(M_, Gx_.transpose(), H_, lhs_);
}

void MixedSQPOptimizer::build_rhs() {
  size_t n = x_.size();
  size_t m = 6*nelem_;
  rhs_.resize(n + m);
  rhs_.setZero();
  double h = config_->h;
  double h2 = h*h;

  // -g_x
  rhs_.segment(0, n) = -M_*(x_ - x0_ - h*vt_ - h2*f_ext_);

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment(6*i,6);

    // W * c(x^k, s^k) + H^-1 * g_s
    rhs_.segment<6>(n + 6*i) = vols_[i]*Sym*(S_[i] - si + Hinv_[i] * g_[i]);
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
    R_[i] = svd.matrixU() * svd.matrixV().transpose();

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
    dS_[i] = Js.transpose() * Sym;
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

#include <unsupported/Eigen/IterativeSolvers>

void MixedSQPOptimizer::substep(bool init_guess, double& decrement) {
  // Factorize LHS (using SparseLU right now)
  // solver_.compute(lhs_);
  // if(solver_.info()!=Success) {
  //  std::cerr << "prefactor failed! " << std::endl;
  //  exit(1);
  // }

  // // Solve for update
  // q_ = solver_.solve(rhs_);
  
  //q_.setZero();

   //Gx_ = -P_ * J_.transpose() * C_.eval() * W_;

  int niter = pcr(q_, P_.rows(), nelem_, lhs_ , rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_, preconditioner_, 1e-4, config_->max_iterative_solver_iters);

    //int niter = pcr(q_, lhs_ , rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_, preconditioner_, 1e-4);

  //fill_block_matrix(M_, H_, P);
  //fill_block_matrix(M_, Gx0_.transpose(), H_, P);
  //solver_.compute(P);
  //q_.setZero();
  //int niter = pcg(q_, lhs_ , rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_, solver_, 1e-4);

  // Eigen CG
  //ConjugateGradient<SparseMatrix<double>, Lower|Upper, IncompleteLUT<double>> cg;
  // IDRS<SparseMatrix<double>, IncompleteLUT<double>> cg;
  // GMRES<SparseMatrixd, IncompleteLUT<double>> cg;
  //cg.compute(lhs_);
  //q_ = cg.solveWithGuess(rhs_, q_);
  //int niter = cg.iterations();
  //std::cout << "  - #iterations:     " << cg.iterations() << std::endl;
  //std::cout << "  - estimated error: " << cg.error()      << std::endl;

  double relative_error = (lhs_*q_ - rhs_).norm() / rhs_.norm();
  decrement = q_.norm(); // if doing "full newton use this"
  std::cout << "  - CG iter: " << niter << std::endl;
  std::cout << "  - RHS Norm: " << rhs_.norm() << std::endl;
  std::cout << "  - Newton decrement: " << decrement  << std::endl;
  std::cout << "  - relative_error: " << relative_error << std::endl;

  // Extract updates
  dx_ = q_.segment(0, x_.size());
  la_ = q_.segment(x_.size(), 6*nelem_);

  double ih2 = 1. / (config_->h * config_->h);

  #pragma omp parallel for 
  for (int i = 0; i < nelem_; ++i) {
    ds_.segment<6>(6*i) = -Hinv_[i] * (ih2 * Sym * la_.segment<6>(6*i)+ g_[i]);
  }

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
  
  double Em = 0.5*xdiff.transpose()*M_*xdiff;

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
    Vector6d diff = Sym * (stmp - si);

    e_L(i) = la.segment<6>(6*i).dot(diff) * vols_[i];
    e_Psi(i) = object_->material_->energy(si) * vols_[i];
  }

  double Ela = e_L.sum();
  double Epsi = h2 * e_Psi.sum();

  double e = Em + Epsi - Ela;
  //std::cout << "E: " <<  e << " ";
  // std::cout << "  - (Em: " << Em << " Epsi: " << Epsi 
  //    << " Ela: " << Ela << " )" << std::endl;
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

bool MixedSQPOptimizer::linesearch(Eigen::VectorXd& x, const Eigen::VectorXd& dx,
        Eigen::VectorXd& s, const Eigen::VectorXd& ds) {

  auto value = [&](const VectorXd& xs)->double {
    return energy(xs.segment(0,x.size()), xs.segment(x.size(),s.size()), la_);
  };

  VectorXd f(x.size() + s.size());
  f.segment(0,x.size()) = x;
  f.segment(x.size(),s.size()) = s;

  VectorXd g(x.size() + s.size());
  g.segment(0,x.size()) = dx;
  g.segment(x.size(),s.size()) = ds;

  VectorXd tmp;
  SolverExitStatus status = linesearch_backtracking_bisection(f, g, value,
      tmp, config_->ls_iters, 1.0, 0.1, 0.5, E_prev_);
  bool done = (status == MAX_ITERATIONS_REACHED);
  x = f.segment(0, x.size());
  s = f.segment(x.size(), s.size());
  E_prev_ = energy(x, s, la_);
  return done;
      
}

void MixedSQPOptimizer::reset() {
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

  // Initializing gradients and LHS
  update_system();

  Gx0_ = Gx_;
  
  // Compute preconditioner
  #if defined(SIM_USE_CHOLMOD)
  std::cout << "Using CHOLDMOD solver" << std::endl;
  #endif
  solver_.compute(lhs_);
  if(solver_.info()!=Success) {
    std::cerr << " KKT prefactor failed! " << std::endl;
  }

  setup_preconditioner();
}
