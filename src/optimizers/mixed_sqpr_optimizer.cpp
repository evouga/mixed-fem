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

void MixedSQPROptimizer::build_lhs() {
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
  // SparseMatrixd G = Gx * Hs * Gx.transpose();
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
}

void MixedSQPROptimizer::build_rhs() {
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
}


void MixedSQPROptimizer::substep(bool init_guess, double& decrement) {
  int niter = 0;

  Eigen::SimplicialLLT<SparseMatrixd> solver(lhs_);
  if(solver.info()!=Success) {
   std::cerr << "!!!!!!!!!!!!!!!prefactor failed! " << std::endl;
   exit(1);
  }
  dx_ = solver.solve(rhs_);
  // BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double>> cg;
  // cg.compute(lhs_);
  // cg.setTolerance(1e-6);
  // dx_ = cg.solveWithGuess(rhs_, dx_);
  // niter = pcg(la_, lhs_ , rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_, solver_, 1e-8);
  // std::cout << "  - CG iters: " << niter << std::endl;
  //std::cout << "estimated error: " << cg.error()      << std::endl;
  double relative_error = (lhs_*dx_ - rhs_).norm() / rhs_.norm(); // norm() is L2 norm

  decrement = dx_.norm(); // if doing "full newton use this"
  //decrement = rhs_.norm();
  //std::cout << "  - # PCG iter: " << niter << std::endl;
  // std::cout << "  - RHS Norm: " << rhs_.norm() << std::endl;
  // std::cout << "  - Newton decrement: " << decrement  << std::endl;
  // std::cout << "  - relative_error: " << relative_error << std::endl;

  VectorXd Gdx = Gx_.transpose() * dx_;
  la_ = -gl_;

  // Update per-element R & S matrices
  ds_.resize(6*nelem_);
  //ds_.setZero();

  #pragma omp parallel for 
  for (int i = 0; i < nelem_; ++i) {
    la_.segment<6>(6*i) += H_[i] * Gdx.segment<6>(6*i);
    //ds_.segment<6>(6*i) = -Hinv_[i] * (ih2 * Sym * la_.segment<6>(6*i)+ g_[i]);
    Vector6d si = s_.segment<6>(6*i);
    Vector6d gs = vols_[i]*(Sym * la_.segment<6>(6*i)+ g_[i]);
    ds_.segment<6>(6*i) = (vols_[i]*config_->h*config_->h*object_->material_->hessian(si,false)).completeOrthogonalDecomposition().solve(-gs);

    // Matrix6d H = vols_[i]*config_->h*config_->h*object_->material_->hessian(si,false);
    // JacobiSVD<MatrixXd, FullPivHouseholderQRPreconditioner> svd(H);
    // ds_.segment<6>(6*i) = svd.solve(-gs);
    // ds_.segment<6>(6*i) = (config_->h*config_->h*object_->material_->hessian(si,false)).jacobiSvd().solve(-gs);
  }
  // #pragma omp parallel for 
  for (int i = 0; i < nelem_; ++i) {
    // ds_.segment<6>(6*i) = -Hinv_[i] * (Sym * la_.segment<6>(6*i)+ g_[i]);
    if (ds_.segment<6>(6*i).norm() > 100) {
      Vector6d si = s_.segment<6>(6*i);
      Matrix6d H = object_->material_->hessian(si);
      std::cout << "si: \n " << si << std::endl;
      std::cout << "H[s]: " << H << std::endl;
      std::cout << "H[s]2: " << object_->material_->hessian(si,false) << std::endl;
      std::cout << "evals: " << H.eigenvalues().real() << std::endl;

      std::cout << "i: " << i << std::endl;
      Vector6d gs = (Sym * la_.segment<6>(6*i)+ g_[i]);
      std::cout << "DS tmp: \n" << (config_->h*config_->h*object_->material_->hessian(si,false)).completeOrthogonalDecomposition().solve(-gs);
      std::cout << "ds i: \n" << ds_.segment<6>(6*i) << std::endl;
      std::cout << "l i: \n" << la_.segment<6>(6*i) << std::endl;
      //std::cout << "Hinv_ i: \n" << Hinv_[i] << std::endl;
      std::cout << "g_ i: \n" << g_[i] << std::endl;
      std::cout << "gs \n" << gs << std::endl;

      SparseMatrixd Hs;
      SparseMatrix<double, RowMajor> Gx = Gx_;
      init_block_diagonal<6,6>(Hs, nelem_);
      update_block_diagonal<6,6>(H_, Hs);
  SparseMatrixd G = Gx * Hs * Gx.transpose();
  saveMarket(lhs_, "lhs.mkt");
  // saveMarket(M_, "M_.mkt");
  saveMarket(Hs, "Hs.mkt");
  saveMarket(G, "GHG.mkt");

      exit(1);
    }
  }
  // std::cout << "dsinf: " << ds_.lpNorm<Infinity>() << " la inf: " << la_.lpNorm<Infinity>() << std::endl;
// exit(1);
  data_.egrad_x_.push_back(dx_.norm());
  data_.egrad_s_.push_back(ds_.norm());
  data_.egrad_la_.push_back(la_.norm());

}

void MixedSQPROptimizer::reset() {
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

  Eigen::SimplicialLDLT<Eigen::SparseMatrixd> Msolver(M_);
  SparseMatrix<double> I(M_.rows(),M_.cols());
  I.setIdentity();
  Minv_ = Msolver.solve(I);
  std::cout << "bad stuff be here" << std::endl;

  // Initializing gradients and LHS
  update_system();
  
  // Compute preconditioner
  // #if defined(SIM_USE_CHOLMOD)
  // std::cout << "Using CHOLDMOD solver" << std::endl;
  // #endif
  // solver_.compute(lhs_);
  // if(solver_.info()!=Success) {
  //   std::cerr << " KKT prefactor failed! " << std::endl;
  // }

}
