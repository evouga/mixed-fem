#include "mixed_sqp_pd_optimizer.h"

#include <chrono>
#include "sparse_utils.h"
#include "svd/dsvd.h"
#include "svd/svd3x3_sse.h"
#include "linesearch.h"
#include "pcg.h"
#include "svd/newton_procrustes.h"
#include "energies/material_model.h"
#include "mesh/mesh.h"

#include <iomanip>
#include <fstream>
#include <rigid_inertia_com.h>
#include "unsupported/Eigen/SparseExtra"

#include "time_integrators/bdf.h"
#include "linear_solvers/solver_factory.h"

using namespace mfem;
using namespace Eigen;
using namespace std::chrono;

void MixedSQPPDOptimizer::step() {
  data_.clear();

  E_prev_ = 0;
  // setup_preconditioner();

  int i = 0;
  double grad_norm;
  double res;
  double E;
  VectorXd gx;
  step_x.clear();
  do {
    if (config_->save_substeps) {
      VectorXd x = P_.transpose()*x_ + b_;
      step_x.push_back(x);
      step_v = vt_;
      step_x0 = x0_;
    }

    // Update gradient and hessian
    update_system();

    // Record initial energies
    E = energy(x_, s_, la_);
    res = std::abs((E - E_prev_) / (E+1));
    E_prev_ = E;

    // Solve system
    substep(i, grad_norm);

    // Linesearch on descent direction
    double alpha = 1.0;
    SolverExitStatus status = linesearch_backtracking_cubic(xvar_, {svar_}, alpha,
        config_->ls_iters);

    // Record some data
    data_.add(" Iteration", i+1);
    data_.add("mixed E", E);
    data_.add("mixed E res", res);
    data_.add("mixed grad", grad_.norm());
    data_.add("Newton dec", grad_norm);
    ++i;

  } while (i < config_->outer_steps && grad_norm > config_->newton_tol
    /*`&& (res > 1e-12)*/);

  if (config_->show_data) {
    data_.print_data(config_->show_timing);
  }

  xvar_->update(gx,0.);
  svar_->lambda().setZero();
  //update_configuration();
}

void MixedSQPPDOptimizer::build_lhs() {}
void MixedSQPPDOptimizer::build_rhs() {}
void MixedSQPPDOptimizer::update_system() {

  VectorXd x = xvar_->value();
  xvar_->unproject(x);

  if (!mesh_->fixed_jacobian()) {
    mesh_->update_jacobian(x);
  }

  svar_->update(x, wdt_*config_->h); 

  // Assemble blocks for left and right hand side
  lhs_ = xvar_->lhs() + svar_->lhs();
  rhs_ = xvar_->rhs() + svar_->rhs();
}

void MixedSQPPDOptimizer::substep(int step, double& decrement) {
  int niter = 0;

  data_.timer.start("global");
  // Eigen::Matrix<double, 12, 1> dx_affine;  
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
  svar_->solve(dx_);
  xvar_->delta() = dx_;
  data_.timer.stop("local");

  decrement = std::max(dx_.lpNorm<Infinity>(), svar_->delta().lpNorm<Infinity>());
}

void MixedSQPPDOptimizer::reset() {
  MixedSQPOptimizer::reset();

  svar_ = std::make_shared<Stretch<3>>(mesh_);
  svar_->reset();
  xvar_ = std::make_shared<Displacement<3>>(mesh_, config_);
  xvar_->reset();

  SparseMatrixdRowMajor A;
  A.resize(nelem_*9, nelem_*9);
  std::vector<Triplet<double>> trips;
  for (int i = 0; i < nelem_; ++i) {
    for (int j = 0; j < 9; ++j) {
      trips.push_back(Triplet<double>(9*i+j, 9*i+j, mesh_->config_->mu / vols_[i]));
    }
  }
  A.setFromTriplets(trips.begin(),trips.end());

  double h2 = wdt_*wdt_*config_->h * config_->h;
  SparseMatrixdRowMajor L = PJ_ * A * PJ_.transpose();
  SparseMatrixdRowMajor lhs = M_ + h2*L;
  solver_arap_.compute(lhs);

  //build up reduced space
  T0_.resize(3*mesh_->V0_.rows(), 12);

  //compute center of mass
  Eigen::Matrix3d I;
  Eigen::Vector3d c;
  double mass = 0;

  //std::cout<<"HERE 1 \n";
  // TODO wrong? should be F_ not T_ for tetrahedra
  sim::rigid_inertia_com(I, c, mass, mesh_->V0_, mesh_->T_, 1.0);

  for(unsigned int ii=0; ii<mesh_->V0_.rows(); ii++ ) {
    //std::cout<<"HERE 2 "<<ii<<"\n";
    T0_.block<3,3>(3*ii, 0) = Eigen::Matrix3d::Identity()*(mesh_->V0_(ii,0) - c(0));
    T0_.block<3,3>(3*ii, 3) = Eigen::Matrix3d::Identity()*(mesh_->V0_(ii,1) - c(1));
    T0_.block<3,3>(3*ii, 6) = Eigen::Matrix3d::Identity()*(mesh_->V0_(ii,2) - c(2));
    T0_.block<3,3>(3*ii, 9) = Eigen::Matrix3d::Identity();
  }

  T0_ = P_*T0_;
  //std::cout<<"c: "<<c.transpose()<<"\n";
  //std::cout<<"T0: \n"<<T0_<<"\n";

  Matrix<double, 12,12> tmp_pre_affine = T0_.transpose()*lhs*T0_; 
  pre_affine_ = tmp_pre_affine.inverse();
}
