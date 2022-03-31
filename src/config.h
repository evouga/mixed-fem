#pragma once

#include <EigenTypes.h>

namespace mfem {

  // Global parameters for the simulation
  struct SimConfig {

  };

  // Simple config for material parameters for a single object
  struct MaterialConfig {
    double mu;
    double la;
  };


}