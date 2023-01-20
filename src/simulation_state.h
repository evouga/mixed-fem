#pragma once

#include "optimizers/optimizer_data.h"
#include "variables/displacement.h"
#include "variables/displacement_gpu.h"
#include "variables/mixed_variable.h"
#include "energies/material_model.h"
#include "time_integrators/implicit_integrator.h"
#include "json/json.hpp"

namespace mfem {  

  // Class to maintain simulation state
  template <int DIM, StorageType STORAGE = STORAGE_EIGEN>
  struct SimState {
    
    // Set traits based on storage type
    template <StorageType _storage, class = void>
    struct Traits;

    // If using Eigen storage, use CPU Position variable
    template <StorageType _storage>
    struct Traits<_storage, std::enable_if_t<(_storage == STORAGE_EIGEN)>> { 
      using PositionType = Displacement<DIM>;
    };

    // If using Thrust storage, use GPU Position variable
    template <StorageType _storage>
    struct Traits<_storage, std::enable_if_t<(_storage == STORAGE_THRUST)>> { 
      using PositionType = DisplacementGpu<DIM>;
    };

    using PositionType = typename Traits<STORAGE>::PositionType;

    // Simulation mesh
    std::shared_ptr<Mesh> mesh_;

    // Scene parameters
    std::shared_ptr<SimConfig> config_;

    // Material models
    std::vector<std::shared_ptr<MaterialModel>> material_models_;

    // Nodal displacement primal variable
    std::unique_ptr<PositionType> x_;

    // Mixed variables
    std::vector<std::unique_ptr<MixedVariable<DIM, STORAGE>>> mixed_vars_;

    // Displacement-based variables
    // These don't maintain the state of the nodal displacements, but
    // compute some energy (stretch, bending, contact) dependent on
    // displacements.
    std::vector<std::shared_ptr<Variable<DIM>>> vars_;

    // The total number of degrees of freedom
    int size() const {
      int n = x_->size();
      for (const auto& var : mixed_vars_) {
        n += var->size() + var->size_dual();
      }
      return n;
    }

    bool load(const std::string& json_file);
    bool load(const nlohmann::json& args);
  
  private:
    // Load triangle mesh
    static void load_mesh(const std::string& path, Eigen::MatrixXd& V,
        Eigen::MatrixXi& T, bool normalize_vertices);

    // Loads simulation parameters
    void load_params(const nlohmann::json& args);
  };

  // Traits for storage type. Used to get the return data types of functions
  // for a particular storage. Same function as traits in Eigen
  // namespace internal {
  //   struct traits<MixedVariable<3>> {
  //     typedef double Scalar;
  //     typedef Eigen::VectorXd VectorType;
  //   };
  // }
}
