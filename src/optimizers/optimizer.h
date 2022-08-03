#pragma once

#include <memory>
#include <functional>
#include <EigenTypes.h>
#include "optimizer_data.h"
#include "boundary_conditions.h"
#include "variables/displacement.h"
#include "variables/mixed_variable.h"
#include "linear_solvers/linear_solver.h"
#include "time_integrators/implicit_integrator.h"

namespace mfem {

  class Mesh;
  class SimConfig;

  template <int DIM>
  struct SimState {
    // For reporting simulation data and timing
    OptimizerData data_;

    // Tracks verticies and applies a selected dirichlet boundary condition
    BoundaryConditions<DIM> BCs_;

    // Simulation mesh
    std::shared_ptr<Mesh> mesh_;

    // Scene parameters
    std::shared_ptr<SimConfig> config_;

    // Time Integrator
    // std::shared_ptr<ImplicitIntegrator> integrator_;

    // Nodal displacement primal variable
    std::shared_ptr<Displacement<DIM>> x_;

    // Mixed variables
    std::vector<std::shared_ptr<MixedVariable<DIM>>> vars_;

    // Linear solver to be used in substep of method
    std::shared_ptr<LinearSolver<double, Eigen::RowMajor>> solver_;
  };

  auto default_optimizer_callback = [](auto &state){};

  
  template <int DIM>
  class Optimizer {
  public:
    Optimizer(const SimState<DIM>& state) : state_(state) {
      callback = default_optimizer_callback;
    }
          
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
