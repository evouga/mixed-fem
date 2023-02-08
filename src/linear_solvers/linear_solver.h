#pragma once

#include "EigenTypes.h"
#include "simulation_state.h"

namespace mfem {

  // Linear solver wrapper class
  template <typename Scalar, int DIM, StorageType STORAGE = STORAGE_EIGEN>
  class LinearSolver {

  public:

    LinearSolver(SimState<DIM,STORAGE>* state) : state_(state) {}
    
    virtual void solve() = 0;

    virtual ~LinearSolver() = default;

    // Get the relative residuals from the solve. If direct solver,
    // there will only be one entry in the returned vector, but if
    // iterative, there will be one entry per iteration.
    virtual std::vector<double> residuals() { return std::vector<double>(); }

  protected:
    SimState<DIM,STORAGE>* state_;
  };

}
