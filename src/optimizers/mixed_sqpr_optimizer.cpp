#include "mixed_sqpr_optimizer.h"

#include <chrono>
#include "sparse_utils.h"
#include "svd/dsvd.h"
#include "svd/svd3x3_sse.h"
#include "linesearch.h"
#include "pinning_matrix.h"
#include "pcg.h"
#include "svd/newton_procrustes.h"

#include <fstream>
#include <rigid_inertia_com.h>
#include "unsupported/Eigen/src/SparseExtra/MarketIO.h"

using namespace mfem;
using namespace Eigen;
using namespace std::chrono;

void MixedSQPROptimizer::step() {
  data_.clear();

  E_prev_ = 0;
  // setup_preconditioner();

  int i = 0;
  double grad_norm;
  double res;
  double E;
  VectorXd gx;
  VectorXd gs;

  do {
    update_system();

    E = energy(x_, s_, la_);
    res = std::abs((E - E_prev_) / (E+1));
    E_prev_ = E;
    double e_primal = primal_energy(x_,s_,gx,gs);

    // Solve system
    substep(i, grad_norm);

    // Linesearch on descent direction from substep
    linesearch_x(x_, dx_);
    // linesearch_s(s_, ds_);
    linesearch_s_local(s_,ds_);
    // linesearch(x_, dx_, s_, ds_);
    // x_ += dx_;
    // s_ += ds_;

    data_.add(" Iteration", i+1);
    data_.add("mixed E", E);
    data_.add("mixed E res", res);
    data_.add("mixed grad", grad_.norm());
    data_.add("Newton dec", grad_norm);
    data_.add("primal E", e_primal);
    data_.add("primal ||gx||", gx.norm());
    data_.add("primal ||gs||", gs.norm());

    ++i;
  } while (i < config_->outer_steps && grad_norm > config_->newton_tol
    && (res > 1e-12));

  data_.print_data(config_->show_timing);

  update_configuration();
}

void MixedSQPROptimizer::gradient(VectorXd& g, const VectorXd& x, const VectorXd& s,
    const VectorXd& la) {  
  VectorXd xt = P_.transpose()*x + b_;
  grad_.resize(xt.size() + 6*nelem_);
  double h = config_->h;

  VectorXd tmp(9*nelem_);
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    tmp.segment<9>(9*i) = dS_[i]*la.segment<6>(6*i);
    grad_.segment<6>(x_.size() + 6*i) = vols_[i] 
        * (g_[i] + Sym*la.segment<6>(6*i));
  }

  grad_.segment(0,xt.size()) = Mfull_*(xt - x0_ - h*vt_ - h*h*f_ext_)
      - Jw_.transpose() * tmp;
}

void MixedSQPROptimizer::build_lhs() {
  data_.timer.start("LHS");

  double ih2 = 1. / (config_->h * config_->h);
  double h2 = (config_->h * config_->h);

  static Eigen::Matrix6d Syminv = (Eigen::Vector6d() <<
    1, 1, 1, .5, .5, .5).finished().asDiagonal();


  data_.timer.start("Hinv");
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment(6*i,6);
    Matrix6d H = h2 * object_->material_->hessian(si);
    Hinv_[i] = H.inverse();
    g_[i] = h2 * object_->material_->gradient(si);
    H_[i] = (1.0 / vols_[i]) * (Syminv * H * Syminv);
  }
  data_.timer.stop("Hinv");

  // std::cout << "HX: " << std::endl;
  // SparseMatrixd Hs;
  // SparseMatrix<double, RowMajor> Gx = Gx_;
  // init_block_diagonal<6,6>(Hs, nelem_);
  // update_block_diagonal<6,6>(H_, Hs);
  // SparseMatrix<double, RowMajor> lhs0 =  Gx * Hs * Gx.transpose();
  // SparseMatrixd G = Gx * Hs * Gx.transpose();
  // saveMarket(M_, "M_.mkt");
  // saveMarket(Hs, "Hs.mkt");
  
  data_.timer.start("Local H");
  std::vector<Matrix12d> Hloc(nelem_); 
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    Hloc[i] = (Jloc_[i].transpose() * (dS_[i] * H_[i]
        * dS_[i].transpose()) * Jloc_[i]) * (vols_[i] * vols_[i]);
  }
  data_.timer.stop("Local H");
  // saveMarket(G, "GHG.mkt");
  //saveMarket(assembler_->A, "lhs2.mkt");
  //saveMarket(M_, "M_.mkt");
  data_.timer.start("Update LHS");
  assembler_->update_matrix(Hloc);
  data_.timer.stop("Update LHS");

  lhs_ = M_ + assembler_->A;
  // std::cout << "B\n" << lhs0 << std::endl;
  // std::cout << "lhs0: " << lhs0.rows() << " " << lhs0.cols();
  // std::cout << "lhs0: " << assembler_->A.rows() << " " << assembler_->A.cols();
  // std::cout << "A:\n" << assembler_->A;
  // saveMarket(assembler_->A, "GHG2.mkt");
  // std::cout << "DIFF: " << (assembler_->A - lhs0).norm() << std::endl;
  //   MatrixXd lhs(lhs_);
  //   EigenSolver<MatrixXd> es(lhs);
  //   std::cout << "LHS EVALS: \n" << es.eigenvalues().real() << std::endl;
  data_.timer.stop("LHS");

}

void MixedSQPROptimizer::build_rhs() {
  data_.timer.start("RHS");

  rhs_.resize(x_.size());
  rhs_.setZero();
  gl_.resize(6*nelem_);
  double h = config_->h;
  double h2 = h*h;

  VectorXd xt = P_.transpose()*x_ + b_;


  VectorXd tmp(9*nelem_);
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment<6>(6*i);

    // W * c(x^k, s^k) + H^-1 * g_s
    gl_.segment<6>(6*i) = vols_[i] * H_[i] * Sym * (S_[i] - si + Hinv_[i] * g_[i]);
    tmp.segment<9>(9*i) = dS_[i]*gl_.segment<6>(6*i);
  }

  rhs_ = -PJ_ * tmp - PM_*(xt - x0_ - h*vt_ - h2*f_ext_);
  data_.timer.stop("RHS");


  grad_.resize(x_.size() + 6*nelem_);
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    tmp.segment<9>(9*i) = dS_[i]*la_.segment<6>(6*i);
    grad_.segment<6>(x_.size() + 6*i) = vols_[i] * (g_[i] + Sym*la_.segment<6>(6*i));
  }
  grad_.segment(0,x_.size()) = PM_*(xt - x0_ - h*vt_ - h2*f_ext_) - PJ_ * tmp;
}

void MixedSQPROptimizer::update_system() {

  // Compute rotations and rotation derivatives
  update_rotations();

  // data_.timer.start("Gx");
  // update_block_diagonal(dS_, C_);
  // Gx_ = -P_ * J_.transpose() * C_.eval() * W_;
  // data_.timer.stop("Gx");

  // Assemble blocks for left and right hand side
  build_lhs();
  build_rhs();
}

void MixedSQPROptimizer::substep(int step, double& decrement) {
  int niter = 0;

  data_.timer.start("global");
  Eigen::Matrix<double, 12, 1> dx_affine;  
  // dx_affine = (T0_.transpose()*lhs_*T0_).lu().solve(T0_.transpose()*rhs_);
  // dx_ = T0_*dx_affine;
  // niter = pcg(dx_, lhs_ , rhs_, tmp_r_, tmp_z_, tmp_zm1_, tmp_p_, tmp_Ap_, solver_arap_, config_->itr_tol, config_->max_iterative_solver_iters);
  // std::cout << "  - CG iters: " << niter;
  // double relative_error = (lhs_*dx_ - rhs_).norm() / rhs_.norm(); 
  // std::cout << " rel error: " << relative_error << " abs error: " << (lhs_*dx_-rhs_).norm() << std::endl;

  solver_.compute(lhs_);
  if(solver_.info()!=Success) {
    std::cerr << "prefactor failed! " << std::endl;
    exit(1);
  }
  dx_ = solver_.solve(rhs_);
  data_.timer.stop("global");


  data_.timer.start("local");
  Jdx_ = - PJ_.transpose() * dx_;
  la_ = -gl_;

  // Update per-element R & S matrices
  ds_.resize(6*nelem_);

  #pragma omp parallel for 
  for (int i = 0; i < nelem_; ++i) {
    la_.segment<6>(6*i) += H_[i] * (dS_[i].transpose() * Jdx_.segment<9>(9*i));
    // ds_.segment<6>(6*i) = -Hinv_[i] * (Sym * la_.segment<6>(6*i)+ g_[i]);
    Vector6d si = s_.segment<6>(6*i);
    Vector6d gs = vols_[i]*(Sym * la_.segment<6>(6*i)+ g_[i]);
    ds_.segment<6>(6*i) = (vols_[i]*config_->h*config_->h*object_->material_->hessian(si,false)).completeOrthogonalDecomposition().solve(-gs);
  }
  data_.timer.stop("local");

  decrement = std::sqrt( dx_.squaredNorm() + ds_.squaredNorm());
  // decrement = std::sqrt(dx_.dot(grad_.segment(0,x_.size())) + ds_.dot(grad_.segment(x_.size(),6*nelem_)));
}

void MixedSQPROptimizer::reset() {
  MixedSQPOptimizer::reset();

  SparseMatrixdRowMajor A;
  A.resize(nelem_*9, nelem_*9);
  std::vector<Triplet<double>> trips;
  for (int i = 0; i < nelem_; ++i) {
    for (int j = 0; j < 9; ++j) {
      trips.push_back(Triplet<double>(9*i+j, 9*i+j, object_->config_->mu / vols_[i]));
    }
  }
  A.setFromTriplets(trips.begin(),trips.end());

  double h2 = config_->h * config_->h;
  SparseMatrixdRowMajor L = PJ_ * A * PJ_.transpose();
  SparseMatrixdRowMajor lhs = M_ + h2*L;
  solver_arap_.compute(lhs);

  //build up reduced space
  T0_.resize(3*object_->V0_.rows(), 12);

  //compute center of mass
  Eigen::Matrix3d I;
  Eigen::Vector3d c;
  double mass = 0;

  //std::cout<<"HERE 1 \n";
  sim::rigid_inertia_com(I, c, mass, object_->V0_, object_->T_, 1.0);

  for(unsigned int ii=0; ii<object_->V0_.rows(); ii++ ) {

    //std::cout<<"HERE 2 "<<ii<<"\n";
    T0_.block<3,3>(3*ii, 0) = Eigen::Matrix3d::Identity()*(object_->V0_(ii,0) - c(0));
    T0_.block<3,3>(3*ii, 3) = Eigen::Matrix3d::Identity()*(object_->V0_(ii,1) - c(1));
    T0_.block<3,3>(3*ii, 6) = Eigen::Matrix3d::Identity()*(object_->V0_(ii,2) - c(2));
    T0_.block<3,3>(3*ii, 9) = Eigen::Matrix3d::Identity();

  }

  T0_ = P_*T0_;
  //std::cout<<"c: "<<c.transpose()<<"\n";
  //std::cout<<"T0: \n"<<T0_<<"\n";

  Matrix<double, 12,12> tmp_pre_affine = T0_.transpose()*lhs*T0_; 
  pre_affine_ = tmp_pre_affine.inverse();


}

