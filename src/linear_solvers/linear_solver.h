#pragma once

#include "EigenTypes.h"
#include "simulation_state.h"

namespace mfem {

  template <typename Scalar, int DIM, StorageType STORAGE = STORAGE_EIGEN>
  class LinearSolver {

  public:

    LinearSolver(SimState<DIM,STORAGE>* state) : state_(state) {}
    
    virtual void solve() = 0;

    virtual ~LinearSolver() = default;

  protected:
    SimState<DIM,STORAGE>* state_;
  };

}
