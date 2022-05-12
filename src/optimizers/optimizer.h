#pragma once

#include <EigenTypes.h>
#include <memory>
#include "objects/simulation_object.h"
#include "config.h"
#include "optimizer_data.h"

namespace mfem {

  static Eigen::Vector6d I_vec = (Eigen::Vector6d() <<
      1, 1, 1, 0, 0, 0).finished();

  static Eigen::Matrix6d Sym = (Eigen::Vector6d() <<
      1, 1, 1, 2, 2, 2).finished().asDiagonal();

  class Optimizer {
  public:
    Optimizer(std::shared_ptr<SimObject> object,
          std::shared_ptr<SimConfig> config)
          : object_(object), config_(config) {}

    virtual void reset() = 0;
    virtual void step() = 0;
  
  protected:

    OptimizerData data_;
    std::shared_ptr<SimConfig> config_;
    std::shared_ptr<SimObject> object_;

    // Debug timing variables (timings in milliseconds)
    std::map<std::string, double> timings;
  };        
  
}
