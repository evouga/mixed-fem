#include "newton_optimizer.h"

#include "linesearch.h"
#include "mesh/mesh.h"
#include "factories/solver_factory.h"
#include <unsupported/Eigen/SparseExtra>
#include "utils/max_possible_step.h"

using namespace mfem;
using namespace Eigen;

template <int DIM>
void NewtonOptimizer<DIM>::step() {
  state_.data_.clear();

  int i = 0;
  double grad_norm;
  double E = 0, E_prev = 0;
  // step_x.clear();
  //config_->kappa = 1e0;
  double kappa0 = state_.config_->kappa;
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

    // If collisions enabled truncate the initial step size to avoid
    // intersections
    // TODO move to linesearch
    if (state_.config_->enable_collisions) {
      VectorXd x1 = state_.x_->value();
      VectorXd x2 = state_.x_->value() + alpha * state_.x_->delta();
      state_.x_->unproject(x1);
      state_.x_->unproject(x2);
      alpha = max_possible_step<DIM>(x1, x2, state_.mesh_->F_);
      std::cout << std::setprecision(10) << "alpha0: " << alpha << std::endl;
    }

    // SolverExitStatus status = linesearch_backtracking_cubic(x_,
    //    {svar_}, alpha, config_->ls_iters);
    //SolverExitStatus status = linesearch_backtracking(state_.x_,
    //    state_.mixed_vars_, alpha, state_.config_->ls_iters, 0.0, 0.9);
    SolverExitStatus status = linesearch_backtracking(state_, alpha, 0.0, 0.9);

    // Record some data
    state_.data_.add(" Iteration", i+1);
    state_.data_.add("Energy", E);
    state_.data_.add("Energy res", res);
    // state_.data_.add("mixed grad", grad_.norm());
    state_.data_.add("Newton dec", grad_norm);
    state_.data_.add("alpha ", alpha);
    state_.data_.add("kappa ", state_.config_->kappa);
    //config_->kappa *= 2;
    ++i;
    Base::callback(state_);

  } while (i < state_.config_->outer_steps
      && grad_norm > state_.config_->newton_tol
    /*`&& (res > 1e-12)*/);

  if (state_.config_->show_data) {
    state_.data_.print_data(state_.config_->show_timing);
  }

  state_.BCs_.step_script(state_.mesh_, state_.config_->h);
  state_.x_->post_solve();
  for (auto& var : state_.mixed_vars_) {
    var->post_solve();
  }
  state_.config_->kappa = kappa0;
}

template <int DIM>
void NewtonOptimizer<DIM>::update_system() {

  VectorXd x = state_.x_->value();
  state_.x_->unproject(x);

  if (!state_.mesh_->fixed_jacobian()) {
    state_.mesh_->update_jacobian(x);
  }

  lhs_ = state_.x_->lhs();
  rhs_ = state_.x_->rhs();

  for (auto& var : state_.vars_) {
    var->update(x, state_.x_->integrator()->dt());
    lhs_ += var->lhs();
    rhs_ += var->rhs();
  }
  for (auto& var : state_.mixed_vars_) {
    var->update(x, state_.x_->integrator()->dt());
    lhs_ += var->lhs();
    rhs_ += var->rhs();
  }
}

template <int DIM>
void NewtonOptimizer<DIM>::substep(double& decrement) {
  state_.data_.timer.start("global");
  state_.solver_->compute(lhs_);
  state_.x_->delta() = state_.solver_->solve(rhs_);
  state_.data_.timer.stop("global");

  decrement = state_.x_->delta().template lpNorm<Infinity>();

  for (auto& var : state_.mixed_vars_) {
    var->solve(state_.x_->delta());
    decrement = std::max(decrement, var->delta().template lpNorm<Infinity>());
  }
  // state_.data_.add("||x delta||", xvar_->delta().norm());
  // state_.data_.add("||s delta||", svar_->delta().norm());
  // state_.data_.add("||c delta||", cvar_->delta().norm());
}

template <int DIM>
void NewtonOptimizer<DIM>::reset() {
  Optimizer<DIM>::reset();

  state_.x_->reset();
  for (auto& var : state_.vars_) {
    var->reset();
  }
  for (auto& var : state_.mixed_vars_) {
    var->reset();
  }

  SolverFactory solver_factory;
  state_.solver_ = solver_factory.create(state_.config_->solver_type,
      state_.mesh_, state_.config_);
}

template class mfem::NewtonOptimizer<3>;
template class mfem::NewtonOptimizer<2>;
