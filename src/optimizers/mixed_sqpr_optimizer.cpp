#include "mixed_sqpr_optimizer.h"

#include <chrono>
#include "sparse_utils.h"
#include "svd/dsvd.h"
#include "svd/svd3x3_sse.h"
#include "linesearch.h"
#include "pinning_matrix.h"
#include "pcg.h"

#include <fstream>
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
  // q_.setZero();
  // q_.segment(0, x_.size()) = x_ - x0_;

  do {
    data_.timer.start("step");
    update_system();
    substep(i==0, grad_norm);

     linesearch_x(x_, dx_);
    // linesearch_s(s_, ds_);
     linesearch_s_local(s_,ds_);
    //linesearch(x_, dx_, s_, ds_);

    // x_ += dx_;
    // s_ += ds_;

    double E = energy(x_, s_, la_);
    double res = std::abs((E - E_prev_) / E);
    data_.egrad_.push_back(grad_norm);
    data_.energies_.push_back(E);
    data_.energy_residuals_.push_back(res);
    E_prev_ = E;
    data_.timer.stop("step");


    ++i;
  } while (i < config_->outer_steps && grad_norm > config_->newton_tol);

  data_.print_data();
  update_configuration();
}

void MixedSQPROptimizer::build_lhs() {
  data_.timer.start("LHS");

  double ih2 = 1. / (config_->h * config_->h);
  double h2 = (config_->h * config_->h);

  static Eigen::Matrix6d Syminv = (Eigen::Vector6d() <<
    1, 1, 1, .5, .5, .5).finished().asDiagonal();


  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment(6*i,6);
    Matrix6d H = h2 * object_->material_->hessian(si);
    Hinv_[i] = H.inverse();
    g_[i] = h2 * object_->material_->gradient(si);
    // H_[i] = - ih2 * vols_[i] *  Sym * Hinv_[i] * Sym;
    H_[i] = (1.0 / vols_[i]) * (Syminv * H * Syminv);

    // Matrix6d Htmp = -vols_[i] *  Sym * Hinv_[i] * Sym;
    // H_[i] = -Htmp.inverse();
  }

  // std::cout << "HX: " << std::endl;
  SparseMatrixd Hs;
  SparseMatrix<double, RowMajor> Gx = Gx_;
  init_block_diagonal<6,6>(Hs, nelem_);
  update_block_diagonal<6,6>(H_, Hs);

  lhs_ = M_;
  lhs_ += Gx * Hs * Gx.transpose();
   SparseMatrixd G = Gx * Hs * Gx.transpose();
  // saveMarket(lhs_, "lhs.mkt");
  // saveMarket(M_, "M_.mkt");
  // saveMarket(Hs, "Hs.mkt");
  // saveMarket(G, "GHG.mkt");

  // lhs_.makeCompressed();
  // std::cout << "rows: " << lhs_.rows() << " R*C: " << lhs_.size() << " nnz: " << lhs_.nonZeros() << std::endl;
  // lhs_.makeCompressed();
  // std::cout << "rows: " << lhs_.rows() << " R*C: " << lhs_.size() << " nnz: " << lhs_.nonZeros() << std::endl;

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

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment(6*i,6);

    // W * c(x^k, s^k) + H^-1 * g_s
    gl_.segment<6>(6*i) = vols_[i] * H_[i] * Sym * (S_[i] - si + Hinv_[i] * g_[i]);

  }
  rhs_ = Gx_ * gl_ - M_*(x_ - x0_ - h*vt_ - h2*f_ext_);
  data_.timer.stop("RHS");

}


void MixedSQPROptimizer::substep(bool init_guess, double& decrement) {
  int niter = 0;

  if (init_guess) {
    data_.timer.start("prefactor");

    solver_.compute(lhs_);
    if(solver_.info()!=Success) {
    std::cerr << "prefactor failed! " << std::endl;
    exit(1);
    }
    dx_ = solver_.solve(rhs_);
    data_.timer.stop("prefactor");

  } else {
    data_.timer.start("global");

    SparseMatrix<double,RowMajor> lhs = lhs_;
    BiCGSTAB<SparseMatrix<double>> cg;
    cg.compute(lhs);
    cg.setTolerance(1e-4);
    cg.setMaxIterations(10);
    dx_ = solver_.solve(rhs_);
    dx_ = cg.solveWithGuess(rhs_, dx_);
    std::cout << "estimated error: " << cg.error()  << " iters: " << cg.iterations() << std::endl;

    // niter = pcg(dx_, lhs_ , rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_, solver_, 1e-8);
    // std::cout << "  - CG iters: " << niter << std::endl;

    data_.timer.stop("global");

  }
  // std::cout << "  - CG iters: " << niter << std::endl;
  //std::cout << "estimated error: " << cg.error()      << std::endl;
  double relative_error = (lhs_*dx_ - rhs_).norm() / rhs_.norm(); // norm() is L2 norm

  decrement = dx_.norm(); // if doing "full newton use this"
  //decrement = rhs_.norm();
  //std::cout << "  - # PCG iter: " << niter << std::endl;
  // std::cout << "  - RHS Norm: " << rhs_.norm() << std::endl;
  // std::cout << "  - Newton decrement: " << decrement  << std::endl;
  std::cout << "  - relative_error: " << relative_error << std::endl;

  data_.timer.start("local");

  VectorXd Gdx = Gx_.transpose() * dx_;
  la_ = -gl_;

  // Update per-element R & S matrices
  ds_.resize(6*nelem_);

  #pragma omp parallel for 
  for (int i = 0; i < nelem_; ++i) {
    la_.segment<6>(6*i) += H_[i] * Gdx.segment<6>(6*i);
    // ds_.segment<6>(6*i) = -Hinv_[i] * (Sym * la_.segment<6>(6*i)+ g_[i]);
    Vector6d si = s_.segment<6>(6*i);
    Vector6d gs = vols_[i]*(Sym * la_.segment<6>(6*i)+ g_[i]);
    ds_.segment<6>(6*i) = (vols_[i]*config_->h*config_->h*object_->material_->hessian(si,false)).completeOrthogonalDecomposition().solve(-gs);
  }
  data_.timer.stop("local");

  // #pragma omp parallel for 
  // for (int i = 0; i < nelem_; ++i) {
  //   // ds_.segment<6>(6*i) = -Hinv_[i] * (Sym * la_.segment<6>(6*i)+ g_[i]);
  //   if (ds_.segment<6>(6*i).norm() > 100) {
  //     Vector6d si = s_.segment<6>(6*i);
  //     Matrix6d H = object_->material_->hessian(si);
  //     std::cout << "si: \n " << si << std::endl;
  //     std::cout << "H[s]: " << H << std::endl;
  //     std::cout << "H[s]2: " << object_->material_->hessian(si,false) << std::endl;
  //     std::cout << "evals: " << H.eigenvalues().real() << std::endl;
  //     std::cout << "evals2: " << object_->material_->hessian(si,false).eigenvalues().real() << std::endl;

  //     std::cout << "i: " << i << std::endl;
  //     Vector6d gs = (Sym * la_.segment<6>(6*i)+ g_[i]);
  //     std::cout << "DS tmp: \n" << (config_->h*config_->h*object_->material_->hessian(si,false)).completeOrthogonalDecomposition().solve(-gs);
  //     std::cout << "ds i: \n" << ds_.segment<6>(6*i) << std::endl;
  //     std::cout << "l i: \n" << la_.segment<6>(6*i) << std::endl;
  //     //std::cout << "Hinv_ i: \n" << Hinv_[i] << std::endl;
  //     std::cout << "g_ i: \n" << g_[i] << std::endl;
  //     std::cout << "gs \n" << gs << std::endl;

  //     SparseMatrixd Hs;
  //     SparseMatrix<double, RowMajor> Gx = Gx_;
  //     init_block_diagonal<6,6>(Hs, nelem_);
  //     update_block_diagonal<6,6>(H_, Hs);
  // SparseMatrixd G = Gx * Hs * Gx.transpose();
  // saveMarket(lhs_, "lhs.mkt");
  // // saveMarket(M_, "M_.mkt");
  // saveMarket(Hs, "Hs.mkt");
  // saveMarket(G, "GHG.mkt");

  //     exit(1);
    // }
  // }
  // std::cout << "dsinf: " << ds_.lpNorm<Infinity>() << " la inf: " << la_.lpNorm<Infinity>() << std::endl;
// exit(1);
  decrement = std::sqrt(dx_.squaredNorm() + ds_.squaredNorm());
  data_.egrad_x_.push_back(dx_.norm());
  data_.egrad_s_.push_back(ds_.norm());
  data_.egrad_la_.push_back(la_.norm());

}
