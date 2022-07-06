#include "mixed_sqp_pd_optimizer.h"

#include "linesearch.h"
#include "mesh/mesh.h"
#include "linear_solvers/solver_factory.h"
#include "factories/integrator_factory.h"

using namespace mfem;
using namespace Eigen;

void MixedSQPPDOptimizer::step() {
  data_.clear();

  E_prev_ = 0;

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
  solver_->compute(lhs_);
  dx_ = solver_->solve(rhs_);
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

  SolverFactory solver_factory;
  solver_ = solver_factory.create(mesh_, config_);
}
