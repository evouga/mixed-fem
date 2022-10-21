#pragma once

#include "factory.h"
#include "config.h"
#include "linear_solvers/linear_solver.h"
#include "simulation_state.h"

namespace mfem {

  template<int DIM>
  class SolverFactory : public Factory<SolverType,
      LinearSolver<double,DIM>, SimState<DIM>*> {
  public:
    SolverFactory();
  };
}
