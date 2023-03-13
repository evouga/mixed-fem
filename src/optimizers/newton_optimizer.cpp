#include "newton_optimizer.h"

#include "linesearch.h"
#include "mesh/mesh.h"
#include "factories/linear_solver_factory.h"
#include <unsupported/Eigen/SparseExtra>
#include "utils/additive_ccd.h"
#include "ipc/ipc.hpp"
#include "igl/edges.h"

using namespace mfem;
using namespace Eigen;

template <int DIM> void NewtonOptimizer<DIM>::step() {
  
  OptimizerData::get().clear();

  // Pre operations for variables
  state_.x_->pre_solve();
  for (auto& var : state_.mixed_vars_) {
    var->pre_solve();
  }
  for (auto& var : state_.vars_) {
    var->pre_solve();
  }

  int i = 0;
  double decrement_norm;
  double E = 0, E_prev = 0, res = 0;

  do {
    Base::callback(state_);

    // Update gradient and hessian
    update_system();

    // Solve linear system
    substep(decrement_norm);

    double alpha = 1.0;

    // If collisions enabled, perform CCD
    // NOTE: this is the chunk we want to delete :) 
    if (state_.config_->enable_ccd) {
      OptimizerData::get().timer.start("ACCD");
      VectorXd x1 = state_.x_->value();
      state_.x_->unproject(x1);
      VectorXd p = state_.mesh_->projection_matrix().transpose() 
          * state_.x_->delta();
      
      // Conservative CCD (scale result by 0.9)
      alpha = 0.9 * ipc::additive_ccd<DIM>(x1, p,
          state_.mesh_->collision_mesh(),
          state_.mesh_->collision_candidates(),
          state_.config_->dhat);
      OptimizerData::get().add("ACCD ", alpha);
      OptimizerData::get().timer.stop("ACCD");
    }

    auto energy_func = [&state = state_](double a) {
      double h2 = std::pow(state.x_->integrator()->dt(), 2);
      Eigen::VectorXd x0 = state.x_->value() + a * state.x_->delta();
      double val = state.x_->energy(x0);
      state.x_->unproject(x0);

      for (const auto& var : state.mixed_vars_) {
        const Eigen::VectorXd si = var->value() + a * var->delta();
        val += h2 * var->energy(x0, si) + var->constraint_value(x0, si);  
      }
      for (const auto& var : state.vars_) {
        val += h2 * var->energy(x0);  
      }
      return val;
    };

    // Record initial energies
    E = energy_func(0.0);
    res = std::abs((E - E_prev) / (E+1e-6));
    E_prev = E;

    // Linesearch on descent direction
    OptimizerData::get().timer.start("LS");
    auto status = linesearch_backtracking(state_, alpha, energy_func,0.0,0.5);
    OptimizerData::get().timer.stop("LS");

    // Record some data
    OptimizerData::get().add(" Iteration", i+1);
    OptimizerData::get().add("Energy", E);
    OptimizerData::get().add("Energy res", res);
    OptimizerData::get().add("Decrement", decrement_norm);
    OptimizerData::get().add("alpha ", alpha);
    ++i;

  } while (i < state_.config_->outer_steps
      && decrement_norm > state_.config_->newton_tol
      && (res > 1e-12)); 

  if (state_.config_->show_data) {
    OptimizerData::get().print_data(state_.config_->show_timing);
  }

  // Update dirichlet boundary conditions
  state_.mesh_->update_bcs(state_.config_->h);

  // Post solve update nodal and mixed variables
  state_.x_->post_solve();
  for (auto& var : state_.mixed_vars_) {
    var->post_solve();
  }
  for (auto& var : state_.vars_) {
    var->post_solve();
  }
}

template <int DIM>
void NewtonOptimizer<DIM>::update_system() {
  OptimizerData::get().timer.start("update");

  // Get full-space configuration
  VectorXd x = state_.x_->value();
  state_.x_->unproject(x);

  // Update derivatives, and assemble hessians
  // and gradients
  for (auto& var : state_.vars_) {
    var->update(x, state_.x_->integrator()->dt());
  }
  for (auto& var : state_.mixed_vars_) {
    var->update(x, state_.x_->integrator()->dt());
  }
  OptimizerData::get().timer.stop("update");
}

template <int DIM>
void NewtonOptimizer<DIM>::substep(double& decrement) {
  // Solve linear system
  OptimizerData::get().timer.start("linsolve");
  linear_solver_->solve();
  OptimizerData::get().timer.stop("linsolve");

  // Use infinity norm of deltas as termination criteria
  decrement = state_.x_->delta().template lpNorm<Infinity>();
  for (auto& var : state_.mixed_vars_) {
    decrement = std::max(decrement, var->delta().template lpNorm<Infinity>());
  }
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

  LinearSolverFactory<DIM> solver_factory;
  linear_solver_ = solver_factory.create(state_.config_->solver_type, &state_);
}

template class mfem::NewtonOptimizer<3>;
template class mfem::NewtonOptimizer<2>;
