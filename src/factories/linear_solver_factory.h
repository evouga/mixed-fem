#pragma once

#include "factory.h"
#include "config.h"
#include "linear_solvers/linear_solver.h"
#include "simulation_state.h"

namespace mfem {

  template<int DIM>
  class LinearSolverFactory : public Factory<LinearSolverType,
      LinearSolver<double,DIM>, SimState<DIM>*> {
  public:
    LinearSolverFactory();
  };
}
