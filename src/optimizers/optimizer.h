#pragma once

#include <EigenTypes.h>
#include <memory>
#include <functional>
#include "optimizer_data.h"
#include "boundary_conditions.h"
#include "time_integrators/implicit_integrator.h"
#include "variables/variable.h"

namespace mfem {

  class Mesh;
  class SimConfig;
  
  template <int DIM>
  class Optimizer {
  public:
    Optimizer(std::shared_ptr<Mesh> object,
          std::shared_ptr<SimConfig> config)
          : mesh_(object), config_(config) {}
          
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

    std::function<void(const std::vector<std::shared_ptr<Variable<DIM>>>&)> callback;

  protected:

    OptimizerData data_;
    std::shared_ptr<Mesh> mesh_;
    std::shared_ptr<SimConfig> config_;
    std::shared_ptr<ImplicitIntegrator> integrator_;

    // Debug timing variables (timings in milliseconds)
    std::map<std::string, double> timings;

    BoundaryConditions<DIM> BCs_;

    int nelem_;             // number of elements
  };        
  
}
