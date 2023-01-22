#include "newton_optimizer_gpu.h"

#include "linesearch.h"
#include "mesh/mesh.h"
#include "factories/linear_solver_factory.h"
#include <unsupported/Eigen/SparseExtra>
#include "utils/additive_ccd.h"
#include "ipc/ipc.hpp"
#include "igl/edges.h"
#include <thrust/inner_product.h>

using namespace mfem;
using namespace Eigen;
using namespace thrust::placeholders;

template <int DIM> 
void NewtonOptimizerGpu<DIM>::step() {
  
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


  // Copy to thrust vector
  thrust::device_vector<double> x(state_.x_->value().size()); 
  thrust::device_vector<double> x_full(state_.mesh_->V_.size());
  std::vector<thrust::device_vector<double>> tmps(state_.mixed_vars_.size());
  for (size_t i = 0; i < state_.mixed_vars_.size(); ++i) {
    tmps[i].resize(state_.mixed_vars_[i]->size());
  }

  do {
    Base::callback(state_);

    // Update gradient and hessian
    update_system();

    // Solve linear system
    substep(grad_norm);

    double alpha = 1.0;

    auto energy_func = [&](double a) {
      double h2 = std::pow(state_.x_->integrator()->dt(), 2);
      // x = x0 + a * p
      thrust::transform(state_.x_->value().begin(), state_.x_->value().end(),
          state_.x_->delta().begin(), x.begin(), _1 + a * _2);

      double* x_full_ptr = state_.x_->to_full(x);

      // Copy cuda x_full to thrust vector
      cudaMemcpy(thrust::raw_pointer_cast(x_full.data()), x_full_ptr,
          state_.mesh_->V_.size() * sizeof(double), cudaMemcpyDeviceToDevice);

      double val = state_.x_->energy(x_full);

      for (size_t i = 0; i < state_.mixed_vars_.size(); ++i) {
        // si = s + a * p
        auto& var = state_.mixed_vars_[i];
        thrust::transform(var->value().begin(), var->value().end(),
            var->delta().begin(), tmps[i].begin(), _1 + a * _2);
        val += h2 * var->energy(x_full, tmps[i]);  
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

    if (status == SolverExitStatus::CONVERGED) {
      // x->value() += alpha * x->delta()
      thrust::transform(state_.x_->value().begin(), state_.x_->value().end(),
          state_.x_->delta().begin(), state_.x_->value().begin(),
           _1 + alpha * _2);
      for (auto& var : state_.mixed_vars_) {
        // si->value() += alpha * si->delta()
        thrust::transform(var->value().begin(), var->value().end(),
            var->delta().begin(), var->value().begin(),
             _1 + alpha * _2);
      }
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
void NewtonOptimizerGpu<DIM>::update_system() {
  OptimizerData::get().timer.start("update");

  double* x_full = state_.x_->to_full(state_.x_->value());

  // Copy to thrust vector
  thrust::device_vector<double> x(state_.mesh_->V_.size());

  // Copy cuda x_full to thrust vector
  cudaMemcpy(thrust::raw_pointer_cast(x.data()), x_full,
      state_.mesh_->V_.size() * sizeof(double), cudaMemcpyDeviceToDevice);

  for (auto& var : state_.vars_) {
    std::cout << "NewtonOptimizer doesn't support position"
        " dependent vars yet" << std::endl;
  }
  for (auto& var : state_.mixed_vars_) {
    var->update(x, state_.x_->integrator()->dt());
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
std::cout << "reset" << std::endl;
  state_.x_->reset();
std::cout << "2" << std::endl;
  for (auto& var : state_.vars_) {
    var->reset();
  }
  for (auto& var : state_.mixed_vars_) {
    var->reset();
  }
std::cout << "3" << std::endl;

  LinearSolverFactory<DIM,STORAGE_THRUST> solver_factory;
  linear_solver_ = solver_factory.create(state_.config_->solver_type, &state_);
std::cout << "4" << std::endl;

}

template class mfem::NewtonOptimizerGpu<3>;