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

  enum OptimizerType {
      OPTIMIZER_ALM,
      OPTIMIZER_ADMM,
      OPTIMIZER_LBFGS,
      OPTIMIZER_SQP,
      OPTIMIZER_SQP_FULL,
      OPTIMIZER_SQP_LA,
  };

  enum MaterialModelType {
      MATERIAL_SNH,
      MATERIAL_FCR,
      MATERIAL_ARAP
  };
  
  // Global parameters for the simulation
  struct SimConfig {
    double h = 0.034; 
    double h2 = h*h;
    double ih2 = 1.0/h/h;
    double grav = -9.8;
    float ext[3] = {0., -9.8, 0.};
    double beta = 5.;
    bool warm_start = false;
    bool floor_collision = false;
    bool regularizer = false;
    bool local_global = true;
    int outer_steps = 5;
    int inner_steps = 7;
    double plane_d = 0;
    double kappa = 1000.0;
    double max_kappa = 1e6;
    double constraint_tol = 1e-2;
    
    // update kappa and lambda if residual below this tolerance
    double update_zone_tol = 1e-1; 

    double newton_tol = 1e-10;
    double ls_tol = 1e-4;
    int ls_iters = 6;
    OptimizerType optimizer = OPTIMIZER_SQP;

  };

  // Simple config for material parameters for a single object
  struct MaterialConfig {
    MaterialModelType material_model = MATERIAL_SNH;
    double ym = 1e6;
    double pr = 0.45;
    double mu = ym/(2.0*(1.0+pr));
    double la = (ym*pr)/((1.0+pr)*(1.0-2.0*pr));
    double density = 1000.0;
    double thickness = 1e-3;
  };

}
