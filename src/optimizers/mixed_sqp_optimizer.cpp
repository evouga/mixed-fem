#include "mixed_sqp_optimizer.h"

#include <chrono>
#include "sparse_utils.h"
#include "svd/dsvd.h"
#include "svd/svd3x3_sse.h"
#include "pcg.h"
// #include "linsolver/nasoq_lbl_eigen.h"
#include "svd/svd_eigen.h"
#include "svd/newton_procrustes.h"

#include <fstream>
#include "unsupported/Eigen/src/SparseExtra/MarketIO.h"

using namespace mfem;
using namespace Eigen;
using namespace std::chrono;


void MixedSQPOptimizer::setup_preconditioner() {

  //really soft preconditioner works better
  double mat_val = std::min(object_->config_->mu, 1e10);

  
  if(std::abs(mat_val - preconditioner_.mat_val()) > 1e-3) {

    std::cout<<"Rebuilding Preconditioner\n";
    std::vector<Matrix9d> C(nelem_);  
    #pragma omp parallel for
    for (int i = 0; i < nelem_; ++i) {
      
      C[i] = Eigen::Matrix9d::Identity()*(-vols_[i] / (mat_val
          * config_->h * config_->h));
    }
    
    SparseMatrixd P;
    SparseMatrixd J = -Jw_ * P_.transpose();
    fill_block_matrix(M_, J, C, P); 
    preconditioner_ = Eigen::CorotatedPreconditioner<double>(mat_val,
        P_.rows(), nelem_, P, dS_);

    //cg.preconditioner() = preconditioner_;
    //cg.setTolerance(1e-5);
    //cg.compute(lhs_);
  }

  preconditioner_.compute(lhs_);
  
}

void MixedSQPOptimizer::build_lhs() {
  data_.timer.start("LHS");

  double ih2 = 1. / (config_->h * config_->h);

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment(6*i,6);
    Matrix6d H = object_->material_->hessian(si);
    Hinv_[i] = H.inverse();
    g_[i] = object_->material_->gradient(si);
    H_[i] = - ih2 * vols_[i] *  Sym * (Hinv_[i] + Matrix6d::Identity()
        *(1./(std::min(std::min(object_->config_->mu, object_->config_->la),
        1e10)))) * Sym;
    // H_[i] = - ih2 * vols_[i] *  Sym * (Hinv_[i]) * Sym;
  }

  SparseMatrixd GxT = Gx_.transpose();
  fill_block_matrix(M_, GxT, H_, lhs_);

  data_.timer.stop("LHS");
}

void MixedSQPOptimizer::build_rhs() {
  data_.timer.start("RHS");

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
  data_.timer.stop("RHS");

  double gradx = (rhs_.segment(0, n) - Gx_ * la_).norm();
  double grads = 0;
  double gradl = 0;
  #pragma omp parallel for reduction(+ : grads)
  for (int i = 0; i < nelem_; ++i) {
    Vector6d g = h2*g_[i] + Sym*la_.segment<6>(6*i);
    grads += (vols_[i]*g).squaredNorm();
  }

  #pragma omp parallel for reduction(+ : gradl)
  for (int i = 0; i < nelem_; ++i) {
    const Vector6d& si = s_.segment(6*i,6);
    gradl += (vols_[i]*Sym*(S_[i] - si)).squaredNorm();
  }
  data_.egrad_x_.push_back(gradx);
  data_.egrad_s_.push_back(std::sqrt(grads));
  data_.egrad_la_.push_back(std::sqrt(gradl));
}

void MixedSQPOptimizer::update_system() {

  // Compute rotations and rotation derivatives
  update_rotations();

  // Assemble rotation derivatives into block matrices
  data_.timer.start("Gx");
  update_block_diagonal(dS_, C_);
  Gx_ = -P_ * J_.transpose() * C_.eval() * W_;
  data_.timer.stop("Gx");

  // Assemble blocks for left and right hand side
  build_lhs();
  build_rhs();
}


#include <amgcl/backend/eigen.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/idrs.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/relaxation/ilut.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/chebyshev.hpp>
#include <amgcl/solver/gmres.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/fgmres.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/preconditioner/schur_pressure_correction.hpp>
#include <amgcl/solver/preonly.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/make_block_solver.hpp>

void MixedSQPOptimizer::substep(int step, double& decrement) {
  //SparseMatrix<double,RowMajor> lhsr = lhs_;

  data_.timer.start("substep");
  // // Solve for update
  solver_.compute(lhs_);
  q_ = solver_.solve(rhs_);

  // SparseMatrixd test = lhs_.triangularView<Eigen::Lower>();
  // test.makeCompressed();
  // q_.segment(P_.rows(), 6*nelem_).setZero();
  // nasoq::linear_solve(test, rhs_, q_);

  //int niter = pcg(q_, lhs_ , rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_,
  //    preconditioner_, 1e-4, config_->max_iterative_solver_iters);
  //std::cout << "niter: " << niter << std::endl;
  // int niter = pcr(q_, lhs_ , rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_,
  //    preconditioner_, 1e-4);

  // SparseMatrix<double,RowMajor> A = lhs_;
  // //A.makeCompressed();
  // typedef amgcl::backend::eigen<double> Backend;
  // typedef amgcl::backend::eigen<Matrix<double,6,6>> BlockBackend;
  // typedef amgcl::make_solver<
  //     amgcl::amg<
  //         Backend,
  //         amgcl::coarsening::aggregation,
  //         amgcl::relaxation::ilut
  //         >,
  //     amgcl::solver::preonly<Backend>
  //     > Usolver;
  // typedef amgcl::make_solver<
  //     amgcl::relaxation::as_preconditioner<
  //         Backend,
  //         amgcl::relaxation::ilut
  //         >,
  //     amgcl::solver::preonly<Backend>
  //     > Psolver;

  // typedef amgcl::make_solver<
  //     amgcl::preconditioner::schur_pressure_correction<
  //         Usolver, Psolver>,
  //     amgcl::solver::idrs<amgcl::backend::eigen<double>>
  //     > Solver;

  // // Solver parameters
  // Solver::params prm;
  // prm.solver.s=5;
  // prm.precond.adjust_p = 1;
  // prm.precond.approx_schur = false;
  // prm.solver.maxiter = 300;
  // // prm.precond.verbose=1;
  // // prm.precond.simplec_dia = false;
  // // prm.precond.usolver.solver.maxiter=8;
  // // prm.precond.psolver.solver.maxiter=8;
  // // prm.precond.simplec_dia = false;
  // prm.precond.pmask.resize(A.rows());
  // for(ptrdiff_t i = 0; i < A.rows(); ++i) prm.precond.pmask[i] = (i >= x_.size());
  
  // typedef amgcl::make_solver<
  //     amgcl::relaxation::as_preconditioner<Backend, amgcl::relaxation::ilut>,
  //     amgcl::solver::fgmres<Backend>
  // > SolverGMRES;
  
  // SolverGMRES::params prm2;
  // prm2.solver.verbose = 1;
  // prm2.solver.tol = 1e-6;
  // //prm2.solver.pside = amgcl::preconditioner::side::left;
  // prm2.solver.maxiter = 100;
  // prm2.precond.tau = 1e-6;
  // prm2.precond.p = 10;
  // //SolverGMRES solve(A, prm2);
  // Solver solve(A, prm);
  // // std::cout << solve << std::endl;


  // // Solve the system for the given RHS:
  // int    iters;
  // double error;
  // q_.setZero();
  // std::tie(iters, error) = solve(rhs_, q_);
  // std::cout << iters << " " << error << std::endl;
  //fill_block_matrix(M_, H_, P);
  //fill_block_matrix(M_, Gx0_.transpose(), H_, P);
  //solver_.compute(P);
  // q_.setZero();
  //int niter = pcg(q_, lhs_ , rhs_, tmp_r_, tmp_z_, tmp_p_, tmp_Ap_, solver_, 1e-4);
  // q_.segment(x_.size(),6*nelem_).setZero();
  // (4)
  // Eigen CG
  // BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double>> cg;
  // IDRS<SparseMatrix<double>, IncompleteLUT<double>> cg;
  // GMRES<SparseMatrixd, IncompleteLUT<double>> cg;
  // cg.compute(lhs_);
  // cg.setTolerance(1e-6);
  // q_ = cg.solveWithGuess(rhs_, q_);
  // //int niter = cg.iterations();
  // std::cout << "  - #iterations:     " << cg.iterations() << std::endl;
  // std::cout << "  - estimated error: " << cg.error()      << std::endl;

  double relative_error = (lhs_*q_ - rhs_).norm() / rhs_.norm();
  decrement = q_.norm();
  /*std::cout << "  - CG iter: " << niter << std::endl;
  std::cout << "  - RHS Norm: " << rhs_.norm() << std::endl;
  std::cout << "  - Newton decrement: " << decrement  << std::endl;
  std::cout << "  - relative_error: " << relative_error << std::endl;*/

  // Extract updates
  dx_ = q_.segment(0, x_.size());
  la_ = q_.segment(x_.size(), 6*nelem_);

  double ih2 = 1. / (config_->h * config_->h);

  #pragma omp parallel for 
  for (int i = 0; i < nelem_; ++i) {
    ds_.segment<6>(6*i) = -Hinv_[i] * (ih2 * Sym * la_.segment<6>(6*i)+ g_[i]);
  }
  data_.timer.stop("substep");

  std::cout << "  - la norm: " << la_.norm() << " dx norm: "
     << dx_.norm() << " ds_.norm: " << ds_.norm() << std::endl;
}

double MixedSQPOptimizer::energy(const VectorXd& x, const VectorXd& s,
        const VectorXd& la) {
  double h = config_->h;
  double h2 = h*h;
  // data_.timer.start("1");
  VectorXd xdiff = x - x0_ - h*vt_ - h*h*f_ext_;
  
  double Em = 0.5*xdiff.transpose()*M_*xdiff;

  VectorXd def_grad = J_*(P_.transpose()*x+b_);

  VectorXd e_L(nelem_);
  VectorXd e_Psi(nelem_);
  // data_.timer.stop("1");

  //svd3x3_sse(def_grad, U_, sigma_, V_);

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
  
  //   std::cout<<"F: \n"<<sim::unflatten<3,3>(def_grad.segment<9>(9*i))<<"\n";
    newton_procrustes(R_[i], Matrix3d::Identity(), Eigen::Map<Matrix3d>(def_grad.segment<9>(9*i).data()), 1e-6);
    Matrix3d S = R_[i].transpose()*Eigen::Map<Matrix3d>(def_grad.segment<9>(9*i).data());
  
    //std::cout<<"R: \n"<<R_[i]<<"\n";
    //std::cout<<"S: \n"<<S<<"\n";
    //Matrix3f S = V_[i] * sigma_[i].asDiagonal() * V_[i].transpose();
    Vector6d stmp; stmp << S(0,0), S(1,1), S(2,2), 0.5*(S(1,0) + S(0,1)), 0.5*(S(2,0) + S(0,2)), 0.5*(S(2,1) + S(1,2));
    const Vector6d& si = s.segment<6>(6*i);
    Vector6d diff = Sym * (stmp - si);
    e_L(i) = la.segment<6>(6*i).dot(diff) * vols_[i];
    e_Psi(i) = object_->material_->energy(si) * vols_[i];
  }
  double Ela = e_L.sum();
  double Epsi = h2 * e_Psi.sum();
  double e = Em + Epsi - Ela;
  return e;
}


void MixedSQPOptimizer::gradient(VectorXd& g, const VectorXd& x, const VectorXd& s,
    const VectorXd& la) {
  std::cerr << "MixedSQPOptimizer: gradient() unimplemented" << std::endl;
}



void MixedSQPOptimizer::reset() {
  MixedOptimizer::reset();
 
  object_->jacobian(Jw_, vols_, true);
  object_->jacobian(Jloc_);
  PJ_ = P_ * Jw_.transpose();

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

  //init R
  #pragma omp parallel for
  for(int ii=0; ii<nelem_; ++ii) {
    R_[ii] = Matrix3d::Identity();
  }

  // Building FEM matrix assembler
  int curr = 0;
  std::vector<int> free_map(pinnedV_.size(), -1);  
  for (int i = 0; i < pinnedV_.size(); ++i) {
    if (pinnedV_(i) == 0) {
      free_map[i] = curr++;
    }
  }
  assembler_ = std::make_shared<Assembler<double,4,3>>(object_->T_, free_map);

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
  //setup_preconditioner();
}
