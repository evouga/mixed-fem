#pragma once

#include <EigenTypes.h>

namespace mfem {

  // Global parameters for the simulation
  struct SimConfig {
    double h = 0.034; 
    double density = 1000.0;
    double ih2 = 1.0/h/h;
    double grav = -9.8;
    double beta = 5.;
    bool warm_start = true;
    bool floor_collision = true;
    int outer_steps = 2;
    int inner_steps = 7;
    double plane_d = 0;
  };

  // Simple config for material parameters for a single object
  struct MaterialConfig {
    double ym = 1e4;
    double pr = 0.45;
    double mu = ym/(2.0*(1.0+pr));
    double la = (ym*pr)/((1.0+pr)*(1.0-2.0*pr));
  };

}