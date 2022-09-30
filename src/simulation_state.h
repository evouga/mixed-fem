#pragma once

#include "optimizers/optimizer_data.h"
#include "variables/displacement.h"
#include "variables/mixed_variable.h"
#include "linear_solvers/linear_solver.h"
#include "energies/material_model.h"
#include "time_integrators/implicit_integrator.h"
#include "json/json.hpp"

namespace mfem {  

  // Class to maintain simulation state
  template <int DIM>
  struct SimState {
    // For reporting simulation data and timing
    OptimizerData data_;

    // Simulation mesh
    std::shared_ptr<Mesh> mesh_;

    // Scene parameters
    std::shared_ptr<SimConfig> config_;

    // Material models
    std::vector<std::shared_ptr<MaterialModel>> material_models_;

    // Nodal displacement primal variable
    std::unique_ptr<Displacement<DIM>> x_;

    // Mixed variables
    std::vector<std::unique_ptr<MixedVariable<DIM>>> mixed_vars_;

    // Displacement-based variables
    // These don't maintain the state of the nodal displacements, but
    // compute some energy (stretch, bending, contact) dependent on
    // displacements.
    std::vector<std::shared_ptr<Variable<DIM>>> vars_;

    // Linear solver to be used in substep of method
    std::unique_ptr<LinearSolver<double, Eigen::RowMajor>> solver_;

    bool load(const std::string& json_file);
    bool load(const nlohmann::json& args);
  
  private:
    // Load triangle mesh
    static void load_mesh(const std::string& path, Eigen::MatrixXd& V,
        Eigen::MatrixXi& T);

    // Loads simulation parameters
    void load_params(const nlohmann::json& args);
  };
}
