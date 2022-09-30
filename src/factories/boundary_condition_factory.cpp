#include "boundary_condition_factory.h"
#include "boundary_conditions/fixed_boundary_conditions.h"
#include "boundary_conditions/bend_boundary_condition.h"
#include "boundary_conditions/stretch_boundary_condition.h"

using namespace mfem;
using namespace Eigen;

BoundaryConditionFactory::BoundaryConditionFactory() {

  register_type(BCScriptType::BC_NULL, "null", 
      [](std::shared_ptr<Mesh> m, BoundaryConditionConfig cfg)
      ->std::unique_ptr<BoundaryCondition>
      {return std::make_unique<NullBC>(m, cfg);});

  register_type(BCScriptType::BC_SCALEF, "scale", 
      [](std::shared_ptr<Mesh> m, BoundaryConditionConfig cfg)
      ->std::unique_ptr<BoundaryCondition>
      {return std::make_unique<ScaleBC>(m, cfg);});

  register_type(BCScriptType::BC_RANDOM, "randomize", 
      [](std::shared_ptr<Mesh> m, BoundaryConditionConfig cfg)
      ->std::unique_ptr<BoundaryCondition>
      {return std::make_unique<RandomizeBC>(m, cfg);});

  register_type(BCScriptType::BC_ONEPOINT, "onepoint", 
      [](std::shared_ptr<Mesh> m, BoundaryConditionConfig cfg)
      ->std::unique_ptr<BoundaryCondition>
      {return std::make_unique<OnePointBC>(m, cfg);});

  register_type(BCScriptType::BC_HANG, "hang", 
      [](std::shared_ptr<Mesh> m, BoundaryConditionConfig cfg)
      ->std::unique_ptr<BoundaryCondition>
      {return std::make_unique<HangBC>(m, cfg);});

  register_type(BCScriptType::BC_HANGENDS, "hangends", 
      [](std::shared_ptr<Mesh> m, BoundaryConditionConfig cfg)
      ->std::unique_ptr<BoundaryCondition>
      {return std::make_unique<HangEndsBC>(m, cfg);});
  



}