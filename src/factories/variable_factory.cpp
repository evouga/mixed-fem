#include "variable_factory.h"
#include "variables/collision.h"
#include "variables/stretch.h"
#include "mesh/mesh.h"

using namespace mfem;
using namespace Eigen;

template<int DIM>
MixedVariableFactory<DIM>::MixedVariableFactory() {
  // Volumetric stretch
  this->register_type(VariableType::VAR_STRETCH, Stretch<DIM>::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<MixedVariable<DIM>>
      {return std::make_unique<Stretch<DIM>>(mesh);});

  // Log-barrier collisions
  this->register_type(VariableType::VAR_COLLISION, Collision<DIM>::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<MixedVariable<DIM>>
      {return std::make_unique<Collision<DIM>>(mesh, config);}); 
}

template class mfem::MixedVariableFactory<3>;
template class mfem::MixedVariableFactory<2>;
