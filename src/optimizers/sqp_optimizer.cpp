#include "sqp_optimizer.h"

#include "linesearch.h"
#include "mesh/mesh.h"
#include "factories/solver_factory.h"
#include "factories/integrator_factory.h"
#include <unsupported/Eigen/SparseExtra>

using namespace mfem;
using namespace Eigen;

template <int DIM>
void SQPOptimizer<DIM>::step() {
  data_.clear();

  int i = 0;
  double grad_norm;
  double E = 0, E_prev = 0;
  // step_x.clear();
  //config_->kappa = 1e0;
  double kappa0 = config_->kappa;
  do {

    // Update gradient and hessian
    update_system();

    // Record initial energies
    // E = energy(x_, s_, la_);
    double res = std::abs((E - E_prev) / (E+1));
    E_prev = E;

    // Solve system
    substep(grad_norm);

    // Linesearch on descent direction
    double alpha = 1.0;
    // SolverExitStatus status = linesearch_backtracking_cubic(x_,
    //    {svar_}, alpha, config_->ls_iters);
    SolverExitStatus status = linesearch_backtracking(x_,
        vars_, alpha, config_->ls_iters, 0.0, 0.9);

    // Record some data
    data_.add(" Iteration", i+1);
    data_.add("mixed E", E);
    data_.add("mixed E res", res);
    // data_.add("mixed grad", grad_.norm());
    data_.add("Newton dec", grad_norm);
    data_.add("alpha ", alpha);
    data_.add("kappa ", config_->kappa);
    //config_->kappa *= 2;
    ++i;
    // Base::callback(vars_);

  } while (i < config_->outer_steps && grad_norm > config_->newton_tol
    /*`&& (res > 1e-12)*/);

  if (config_->show_data) {
    data_.print_data(config_->show_timing);
  }

  this->BCs_.step_script(mesh_, config_->h);
  x_->post_solve();
  for (auto& var : vars_) {
    var->post_solve();
  }
  config_->kappa = kappa0;
}

template <int DIM>
void SQPOptimizer<DIM>::update_system() {

  VectorXd x = x_->value();
  x_->unproject(x);

  if (!mesh_->fixed_jacobian()) {
    mesh_->update_jacobian(x);
  }

  lhs_ = x_->lhs();
  rhs_ = x_->rhs();

  // NOTE changed 1.0 -> dt() for cvar
  for (auto& var : vars_) {
    var->update(x, x_->integrator()->dt());
    lhs_ += var->lhs();
    rhs_ += var->rhs();
  }
}

template <int DIM>
void SQPOptimizer<DIM>::substep(double& decrement) {
  data_.timer.start("global");
  solver_->compute(lhs_);
  x_->delta() = solver_->solve(rhs_);
  data_.timer.stop("global");

  decrement = x_->delta().template lpNorm<Infinity>();

  for (auto& var : vars_) {
    var->solve(x_->delta());
    decrement = std::max(decrement, var->delta().template lpNorm<Infinity>());
  }
}

template <int DIM>
void SQPOptimizer<DIM>::reset() {
  Optimizer<DIM>::reset();

  x_->reset();
  for (auto& var : vars_) {
    var->reset();
  }

  SolverFactory solver_factory;
  solver_ = solver_factory.create(config_->solver_type, mesh_, config_);
}

template class mfem::SQPOptimizer<3>;
template class mfem::SQPOptimizer<2>;
