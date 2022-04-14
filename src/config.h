#pragma once

#include <EigenTypes.h>

namespace mfem {

  // Converts Young's modulus and Poisson's ratio to Lame parameters
  // E      - young's modulus (in pascals)
  // nu     - poisson's ratio
  // lambda - lame first parameter
  // mu     - lame second parameter
  constexpr void Enu_to_lame(double E, double nu, double& lambda, double& mu) {
    mu = E/(2.0*(1.0+nu));
    lambda = (E*nu)/((1.0+nu)*(1.0-2.0*nu));
  }

  // Global parameters for the simulation
  struct SimConfig {
    double h = 0.034; 
    double density = 1000.0;
    double ih2 = 1.0/h/h;
    double grav = -9.8;
    float ext[3] = {0., -9.8, 0.};
    double beta = 5.;
    bool warm_start = true;
    bool floor_collision = false;
    bool regularizer = false;
    int outer_steps = 2;
    int inner_steps = 7;
    double plane_d = 0;
    double thickness = 1e-3;
    double kappa = 1.0;
  };

  // Simple config for material parameters for a single object
  struct MaterialConfig {
    double ym = 1e5;
    double pr = 0.45;
    double mu = ym/(2.0*(1.0+pr));
    double la = (ym*pr)/((1.0+pr)*(1.0-2.0*pr));
  };

}
