#include "optimizers/mixed_alm_optimizer.h"
#include "optimizers/mixed_admm_optimizer.h"
#include "optimizers/mixed_sqp_optimizer.h"
#include "optimizers/mixed_sqp_pd_optimizer.h"
#include "optimizers/newton_optimizer.h"
#include "optimizers/optimizer_factory.h"

using namespace mfem;

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
  register_type(OptimizerType::OPTIMIZER_SQP_PD, MixedSQPROptimizer::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Optimizer>
      {return std::make_unique<MixedSQPROptimizer>(mesh, config);});

  // Newton's
  register_type(OptimizerType::OPTIMIZER_NEWTON, NewtonOptimizer::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Optimizer>
      {return std::make_unique<NewtonOptimizer>(mesh, config);});     
}

std::unique_ptr<Optimizer> OptimizerFactory::create(
    std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config) {

  if (auto it = type_creators_.find(config->optimizer);
      it !=  type_creators_.end()) {
    return it->second(mesh, config);
  }
  std::cout << "OptimizerFactory create: type not found" << std::endl;
  return nullptr;
}

std::unique_ptr<Optimizer> OptimizerFactory::create(const std::string& type,
    std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config) {

  if (auto it = str_type_creators_.find(type); it !=  str_type_creators_.end())
  {
    return it->second(mesh, config);
  }
  return nullptr;
}

void OptimizerFactory::register_type(OptimizerType type,
    const std::string& name, TypeCreator func) {
  type_creators_.insert(std::pair<OptimizerType, TypeCreator>(type, func));
  str_type_creators_.insert(std::pair<std::string, TypeCreator>(name, func));
}