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

  // TODO shouldn't list everything here, use a factory or whatever the fuck
  // that OOP design is called
  enum OptimizerType {
      OPTIMIZER_ALM,
      OPTIMIZER_ADMM,
      OPTIMIZER_SQP,
      OPTIMIZER_SQP_PD,
      OPTIMIZER_NEWTON,
  };

  enum MaterialModelType {
      MATERIAL_SNH,   // Stable neohookean
      MATERIAL_NH,    // neohookean
      MATERIAL_COROT, // corotational
      MATERIAL_ARAP,  // as-rigid-as possible
      MATERIAL_FUNG   // exponential
  };

  enum BCScriptType {
      BC_NULL,
      BC_SCALEF,
      BC_HANG,
      BC_HANGENDS,
      BC_STRETCH,
      BC_SQUASH,
      BC_STRETCHNSQUASH,
      BC_BEND,
      BC_TWIST,
      BC_TWISTNSTRETCH,
      BC_TWISTNSNS,
      BC_TWISTNSNS_OLD,
      BC_RUBBERBANDPULL,
      BC_ONEPOINT,
      BC_RANDOM,
      BC_FALL,
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
    bool show_timing = true;
    bool show_data = true;
    bool save_substeps = false;
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
    int ls_iters = 20;
    OptimizerType optimizer = OPTIMIZER_SQP_PD;
    int max_iterative_solver_iters = 500;
    double itr_tol = 1e-4;
    BCScriptType bc_type = BC_ONEPOINT;

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
