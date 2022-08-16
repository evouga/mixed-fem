#pragma once

#include <memory>
#include <functional>
#include <EigenTypes.h>
#include "simulation_state.h"

namespace mfem {

  class Mesh;
  class SimConfig;

  auto default_optimizer_callback = [](auto& state){};

  template <int DIM>
  class Optimizer {
  public:
    Optimizer(SimState<DIM>& state) : state_(std::move(state)) {
      callback = default_optimizer_callback;
    }

    virtual ~Optimizer() = default;
          
    static std::string name() {
      return "base";
    }

    virtual void reset();
    virtual void step() = 0;
    

    virtual void update_vertices(const Eigen::MatrixXd& V) {
      std::cerr << "Update vertices not implemented!" << std::endl;
    }
    virtual void set_state(const Eigen::VectorXd& x,
        const Eigen::VectorXd& v) {
      std::cerr << "Update state not implemented!" << std::endl;
    }

    SimState<DIM>& state() {
      return state_;
    }
    
    // Temporary. Should be a part of a callback function instead.
    // Used to save per substep vertices;
    std::vector<Eigen::MatrixXd> step_x;
    Eigen::VectorXd step_x0;
    Eigen::VectorXd step_v;

    std::function<void(const SimState<DIM>& state)> callback;

  protected:

    SimState<DIM> state_;
  };        
  
}
