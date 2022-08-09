#include "simulation_state.h"
#include "config.h"
#include <fstream>
#include <sstream>
#include "json/json.hpp"

using json = nlohmann::json;
using namespace mfem;
using namespace Eigen;

template <int DIM>
void SimState<DIM>::load() {

using namespace nlohmann;

  // Default args (any null values will be initalized below)
  json args = R"({
    "optimizer": "newton",
    "time_integrator": "bdf1",
    "linear_solver": "cholmod",
    "timestep": 0.0333,
    "body_force": [0.0, -9.8, 0.0],
    "print_stats": true,
    "print_timing": true,
    "max_iterations": 5,
    "max_linesearch_iterations": 20,
    "kappa": 1,
    "objects": [
      {
        name: "../models/obj/T.xy
      }
    ],
    "material_models": [
      {
        "youngs_modulus": 1000000,
        "poissons_ratio": 0.45,
        "density": 1000,
        "energy": "Stable-Neohookean"
      },
    ],
    "mixed_variables": ["mixed-stretch","mixed-collision"],
    "variables": [],
    "enable_ccd": true,
    "boundary_condition": "hang",
    "dhat": 0.01
  })"_json;
}