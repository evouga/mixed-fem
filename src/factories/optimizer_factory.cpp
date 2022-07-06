#include "optimizer_factory.h"
#include "optimizers/optimizer.h"
#include "optimizers/mixed_alm_optimizer.h"
#include "optimizers/mixed_admm_optimizer.h"
#include "optimizers/mixed_sqp_optimizer.h"
#include "optimizers/mixed_sqp_pd_optimizer.h"
#include "optimizers/newton_optimizer.h"
#include "optimizers/mixed_sqp_bending.h"
#include "mesh/mesh.h"

using namespace mfem;
using namespace Eigen;

OptimizerFactory::OptimizerFactory() {

  // ADMM
  register_type(OptimizerType::OPTIMIZER_ADMM, MixedADMMOptimizer::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Optimizer>
      {return std::make_unique<MixedADMMOptimizer>(mesh, config);});

  // ALM
  register_type(OptimizerType::OPTIMIZER_ALM, MixedALMOptimizer::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Optimizer>
      {return std::make_unique<MixedALMOptimizer>(mesh, config);});

  // SQP Indefinite
  register_type(OptimizerType::OPTIMIZER_SQP, MixedSQPOptimizer::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Optimizer>
      {return std::make_unique<MixedSQPOptimizer>(mesh, config);});

  // SQP Positive Definite
  register_type(OptimizerType::OPTIMIZER_SQP_PD, MixedSQPPDOptimizer::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Optimizer>
      {return std::make_unique<MixedSQPPDOptimizer>(mesh, config);});

  // Newton's
  register_type(OptimizerType::OPTIMIZER_NEWTON, NewtonOptimizer::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Optimizer>
      {return std::make_unique<NewtonOptimizer>(mesh, config);});

  // SQP with Bending Energy
  register_type(OptimizerType::OPTIMIZER_SQP_BENDING, MixedSQPBending::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Optimizer>
      {return std::make_unique<MixedSQPBending>(mesh, config);});
}
