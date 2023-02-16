#include "newton_optimizer_gpu.h"

#include "linesearch.h"
#include "mesh/mesh.h"
#include "factories/linear_solver_factory.h"
#include <unsupported/Eigen/SparseExtra>
#include "utils/additive_ccd.h"
#include "ipc/ipc.hpp"
#include "igl/edges.h"
#include <thrust/inner_product.h>
#include "utils/ccd_gpu.h"

using namespace mfem;
using namespace Eigen;
using namespace thrust::placeholders;

template <int DIM> 
void NewtonOptimizerGpu<DIM>::step() {
  
  OptimizerData::get().clear();

  // Pre solve operations for variables
  OptimizerData::get().timer.start("pre_solve");
  state_.x_->pre_solve();
  for (auto& var : state_.mixed_vars_) {
    var->pre_solve();
  }
  for (auto& var : state_.vars_) {
    var->pre_solve();
  }
  OptimizerData::get().timer.stop("pre_solve");


  int i = 0;
  double grad_norm;
  double E = 0, E_prev = 0, res = 0;

  // TODO 
  // Need to construct collision candidates if it's
  // not been run yet. Happens for first timestep, so
  // we may end up missing collisions

  std::vector<thrust::device_vector<double>> tmps(state_.mixed_vars_.size());

  // spdlog::set_level(spdlog::level::trace);

  // Compute z = x + a * y
  auto add_vec = [](const thrust::device_vector<double>& x, double a,
      const thrust::device_vector<double>& y,
      thrust::device_vector<double>& z) {
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), _1 + a * _2);
  };

  bool ls_done = false;
  do {
    Base::callback(state_);

    // Update gradient and hessian
    state_.x_->to_full(state_.x_->value(), x_full_);

    update_system();

    // Solve linear system
    substep(grad_norm);

    double alpha = 1.0;

    // If collisions enabled, perform CCD
    if (state_.config_->enable_ccd) {
      OptimizerData::get().timer.start("CCD");

      // Compute x2 = x + a * dx, and use this for CCD
      add_vec(state_.x_->value(), 1.0, state_.x_->delta(), x_);
      state_.x_->to_full(x_, x2_full_);
      alpha = ipc::additive_ccd<DIM>(x_full_, x2_full_,
          state_.mesh_->collision_mesh(),
          state_.mesh_->collision_candidates(),
          state_.config_->dhat);
      OptimizerData::get().add("CCD ", alpha);
      OptimizerData::get().timer.stop("CCD");
    }
    for (size_t i = 0; i < state_.mixed_vars_.size(); ++i) {
      tmps[i].resize(state_.mixed_vars_[i]->size());
    }
    auto energy_func = [&](double a) {
      double h2 = std::pow(state_.x_->integrator()->dt(), 2);
      // x = x0 + a * p
      add_vec(state_.x_->value(), a, state_.x_->delta(), x_);

      // From reduced coordinates to full (with dirichlet BCs)
      state_.x_->to_full(x_, x_full_);

      // Inertial energy
      double val = state_.x_->energy(x_full_);
      // std::cout << "Displacement energy: " << val << std::endl;

      for (size_t i = 0; i < state_.mixed_vars_.size(); ++i) {
        // si = s + a * p
        add_vec(state_.mixed_vars_[i]->value(), a,
                  state_.mixed_vars_[i]->delta(), tmps[i]);
        // std::cout << "  Mixed energy: " << 
        //     h2 * state_.mixed_vars_[i]->energy(x_full_, tmps[i]) << std::endl;
        val += h2 * state_.mixed_vars_[i]->energy(x_full_, tmps[i]);
      }
      return val;
    };

    // // Record initial energies
    E = energy_func(0.0);
    res = std::abs((E - E_prev) / (E+1e-6));
    E_prev = E;

    // // Linesearch on descent direction
    OptimizerData::get().timer.start("LS");
    auto status = linesearch_backtracking2(state_, alpha, energy_func, 0.0,0.5);
    OptimizerData::get().timer.stop("LS");

    if (true || status == SolverExitStatus::CONVERGED) {
      // x += alpha * dx
      add_vec(state_.x_->value(), alpha, state_.x_->delta(),
                state_.x_->value());
      for (auto& var : state_.mixed_vars_) {
        // s += alpha * ds
        add_vec(var->value(), alpha, var->delta(), var->value());
      }
    } else {
      // ls_done = true;
    }

    // check for error
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

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
      && (res > 1e-12)
      && !ls_done); 

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
void NewtonOptimizerGpu<DIM>::update_system() {
  OptimizerData::get().timer.start("update");
  
  for (auto& var : state_.vars_) {
    std::cout << "NewtonOptimizer doesn't support position"
        " dependent vars yet" << std::endl;
  }
  for (auto& var : state_.mixed_vars_) {
    var->update(x_full_, state_.x_->integrator()->dt());
  }
  OptimizerData::get().timer.stop("update");
}

template <int DIM>
void NewtonOptimizerGpu<DIM>::substep(double& decrement) {
  // Solve linear system
  OptimizerData::get().timer.start("linsolve");
  linear_solver_->solve();
  OptimizerData::get().timer.stop("linsolve");

  decrement = thrust::inner_product(state_.x_->delta().begin(),
      state_.x_->delta().end(), state_.x_->delta().begin(), 0.0);
  // // Use infinity norm of deltas as termination criteria
  // decrement = state_.x_->delta().template lpNorm<Infinity>();
  // for (auto& var : state_.mixed_vars_) {
  //   decrement = std::max(decrement, var->delta().template lpNorm<Infinity>());
  // }
}

template <int DIM>
void NewtonOptimizerGpu<DIM>::reset() {
  Base::reset();
  state_.x_->reset();
  for (auto& var : state_.vars_) {
    var->reset();
  }
  for (auto& var : state_.mixed_vars_) {
    var->reset();
  }

  x_.resize(state_.x_->value().size());
  x_full_.resize(state_.mesh_->V_.size());
  x2_full_.resize(state_.mesh_->V_.size());

  LinearSolverFactory<DIM,STORAGE_THRUST> solver_factory;
  linear_solver_ = solver_factory.create(state_.config_->solver_type, &state_);
}

template class mfem::NewtonOptimizerGpu<3>;