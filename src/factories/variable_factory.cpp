#include "variable_factory.h"
#include "variables/mixed_collision.h"
#include "variables/mixed_stretch.h"
#include "variables/mixed_stretch_gpu.h"
#include "variables/stretch.h"
#include "variables/collision.h"
#include "variables/friction.h"
#include "variables/displacement_gpu.h"
#include "mesh/mesh.h"

using namespace mfem;
using namespace Eigen;

template<int DIM, StorageType STORAGE>
MixedVariableFactory<DIM,STORAGE>::MixedVariableFactory() {
  // Volumetric stretch
  if constexpr (STORAGE == STORAGE_EIGEN) {
    this->register_type(VariableType::VAR_MIXED_STRETCH,
        MixedStretch<DIM>::name(),
        [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
        ->std::unique_ptr<MixedVariable<DIM>>
        {return std::make_unique<MixedStretch<DIM>>(mesh);});

    // Log-barrier collisions
    this->register_type(VariableType::VAR_MIXED_COLLISION,
        MixedCollision<DIM>::name(),
        [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
        ->std::unique_ptr<MixedVariable<DIM>>
        {return std::make_unique<MixedCollision<DIM>>(mesh, config);}); 
  }
  // GPU versions only supported in 3D
  if constexpr (DIM == 3) {
    // If using thrust storage type, just use the CPU name since the
    // CPU mixed thrust cannot be used with thrust. This makes all the
    // existing scene jsons still compatible.
    std::string name = MixedStretchGpu<DIM,STORAGE>::name();
    if constexpr (STORAGE == STORAGE_THRUST) {
      name = MixedStretch<DIM>::name();
    }
    this->register_type(VariableType::VAR_MIXED_STRETCH_GPU, name,
        [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
        ->std::unique_ptr<MixedVariable<DIM,STORAGE>>
        {return std::make_unique<MixedStretchGpu<DIM,STORAGE>>(mesh,config);});
  }


}

template<int DIM>
VariableFactory<DIM>::VariableFactory() {
  // Volumetric stretch
  this->register_type(VariableType::VAR_STRETCH,
      Stretch<DIM>::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Variable<DIM>>
      {return std::make_unique<Stretch<DIM>>(mesh);});

  // Log-barrier collisions
  this->register_type(VariableType::VAR_COLLISION,
      Collision<DIM>::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Variable<DIM>>
      {return std::make_unique<Collision<DIM>>(mesh, config);});

  // IPC Friction
  this->register_type(VariableType::VAR_FRICTION,
      Friction<DIM>::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Variable<DIM>>
      {return std::make_unique<Friction<DIM>>(mesh, config);});
}

template class mfem::MixedVariableFactory<3, STORAGE_THRUST>;
template class mfem::MixedVariableFactory<3, STORAGE_EIGEN>;
template class mfem::MixedVariableFactory<2, STORAGE_EIGEN>;

template class mfem::VariableFactory<3>;
template class mfem::VariableFactory<2>;
