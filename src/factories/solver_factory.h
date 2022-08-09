#pragma once

#include "factory.h"
#include "config.h"
#include "linear_solvers/linear_solver.h"

namespace mfem {

  class Mesh;

  class SolverFactory : public Factory<SolverType,
      LinearSolver<double,Eigen::RowMajor>, 
      std::shared_ptr<Mesh>, std::shared_ptr<SimConfig>> {
  public:
    SolverFactory();
  };
}
