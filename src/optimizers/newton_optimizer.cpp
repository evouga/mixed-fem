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

template <int DIM> 
void NewtonOptimizer<DIM>::step() {
  
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
  double grad_norm;
  double E = 0, E_prev = 0, res = 0;

  // TODO 
  // Need to construct collision candidates if it's
  // not been run yet. Happens for first timestep, so
  // we may end up missing collisions

  do {
    Base::callback(state_);

    // Update gradient and hessian
    update_system();

    // Solve linear system
    substep(grad_norm);

    double alpha = 1.0;

    // If collisions enabled, perform CCD
    // TODO enable_ccd in simulate_state initialization
    if (state_.config_->enable_ccd) {
      OptimizerData::get().timer.start("ACCD");
      VectorXd x1 = state_.x_->value();
      state_.x_->unproject(x1);
      VectorXd p = state_.mesh_->projection_matrix().transpose() 
          * state_.x_->delta();
      alpha = 0.9 * ipc::additive_ccd<DIM>(x1, p,
          state_.mesh_->collision_mesh(),
          state_.mesh_->collision_candidates(),
          state_.config_->dhat);
      OptimizerData::get().add("ACCD ", alpha);
      OptimizerData::get().timer.stop("ACCD");
    }

    auto energy_func = [&state = state_](double a) {
      double h2 = std::pow(state.x_->integrator()->dt(), 2);
      //TODO bunch of unneccessary copies
      Eigen::VectorXd x0 = state.x_->value() + a * state.x_->delta();
      double val = state.x_->energy(x0);
      state.x_->unproject(x0);
      std::cout << "Displacement energy: " << val << std::endl;

      for (const auto& var : state.mixed_vars_) {
        Eigen::VectorXd si = var->value() + a * var->delta();
        std::cout << "Mixed energy: " <<  h2 * var->energy(x0, si)  << std::endl;
        std::cout << "Mixed constraint: " << var->constraint_value(x0, si) << std::endl;
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
    OptimizerData::get().add("Decrement", grad_norm);
    OptimizerData::get().add("alpha ", alpha);
    ++i;
    //Base::callback(state_);

  } while (i < state_.config_->outer_steps
      && grad_norm > state_.config_->newton_tol
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

  // Get full configuration vector
  VectorXd x = state_.x_->value();
  state_.x_->unproject(x);
  //std::cout << std::setprecision(10) << std::endl;
  //std::cout << "x norm: " << x.norm() << std::endl;

  if (!state_.mesh_->fixed_jacobian()) {
    state_.mesh_->update_jacobian(x);
  }

  for (auto& var : state_.vars_) {
    var->update(x, state_.x_->integrator()->dt());
  }
  for (auto& var : state_.mixed_vars_) {
    var->update(x, state_.x_->integrator()->dt());
  }
  OptimizerData::get().timer.stop("update");

  // TODO bullshit exporting
  // saveMarket(state_.x_->lhs(), "lhs_M.mkt");
  // saveMarket(state_.x_->rhs(), "rhs_x.mkt");
  //SparseMatrix<double> A, B, C;
  //VectorXd gs,gl;
  //state_.mixed_vars_[0]->hessian(A);
  //state_.mixed_vars_[0]->jacobian_x(B);
  //state_.mixed_vars_[0]->jacobian_mixed(C);
  //gs = state_.mixed_vars_[0]->gradient_mixed();
  //state_.mixed_vars_[0]->evaluate_constraint(x, gl);
  //saveMarket(A, "lhs_K.mkt");
  //saveMarket(B, "lhs_Gx.mkt");
  //saveMarket(C, "lhs_Gs.mkt");
  //saveMarket(-gs, "rhs_gs.mkt");
  //saveMarket(-gl, "rhs_gl.mkt");
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
