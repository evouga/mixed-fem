#pragma once

#include "factory.h"
#include "config.h"
#include "linear_solvers/linear_solver.h"
#include "simulation_state.h"

namespace mfem {

  template<int DIM, StorageType STORAGE = STORAGE_EIGEN>
  class LinearSolverFactory : public Factory<LinearSolverType,
      LinearSolver<double,DIM,STORAGE>, SimState<DIM,STORAGE>*> {
  public:
    LinearSolverFactory();
  private:
    // Register positive-definite Schur-complement based solvers
    void register_pd_solvers();

    // Register indefinite system solvers
    void register_indefinite_solvers();
  };
}
