#pragma once

#include <memory>
#include <functional>
#include <EigenTypes.h>
#include "simulation_state.h"
#include "linear_solvers/linear_solver.h"

namespace mfem {

  class Mesh;
  class SimConfig;

  auto default_optimizer_callback = [](auto& state){};

  template <int DIM, StorageType STORAGE = STORAGE_EIGEN>
  class Optimizer {
  public:
    Optimizer(SimState<DIM,STORAGE>& state) : state_(std::move(state)) {
      callback = default_optimizer_callback;
    }

    virtual ~Optimizer() = default;
          
    static std::string name() {
      return "base";
    }

    virtual void reset();
    virtual void step() = 0;
    
    SimState<DIM,STORAGE>& state() {
      return state_;
    }

    std::function<void(const SimState<DIM,STORAGE>& state)> callback;

  protected:

    SimState<DIM,STORAGE> state_;
    
  };        
  
}
