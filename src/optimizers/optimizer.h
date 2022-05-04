#pragma once

#include <EigenTypes.h>
#include <memory>
#include "objects/simulation_object.h"
#include "config.h"


namespace mfem {

  class Optimizer {
  public:
    Optimizer(std::shared_ptr<SimObject> object,
          std::shared_ptr<SimConfig> config)
          : object_(object), config_(config) {}

    virtual void reset() = 0;
    virtual void step() = 0;
  
  protected:

    std::shared_ptr<SimConfig> config_;
    std::shared_ptr<SimObject> object_;

    // Debug timing variables (timings in milliseconds)
    std::map<std::string, double> timings;
  };        
  
}