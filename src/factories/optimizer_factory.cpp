#include "optimizer_factory.h"
#include "optimizers/optimizer.h"
#include "optimizers/mixed_admm_optimizer.h"
#include "optimizers/mixed_sqp_optimizer.h"
#include "optimizers/mixed_sqp_pd_optimizer.h"
#include "optimizers/newton_optimizer.h"
#include "optimizers/mixed_sqp_bending.h"
#include "mesh/mesh.h"

using namespace mfem;
using namespace Eigen;

template<>
OptimizerFactory<3>::OptimizerFactory() {

  // ADMM
  register_type(OptimizerType::OPTIMIZER_ADMM, MixedADMMOptimizer::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Optimizer<3>>
      {return std::make_unique<MixedADMMOptimizer>(mesh, config);});

  // SQP Indefinite
  register_type(OptimizerType::OPTIMIZER_SQP, MixedSQPOptimizer::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Optimizer<3>>
      {return std::make_unique<MixedSQPOptimizer>(mesh, config);});

  // SQP Positive Definite
  register_type(OptimizerType::OPTIMIZER_SQP_PD,MixedSQPPDOptimizer<3>::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Optimizer<3>>
      {return std::make_unique<MixedSQPPDOptimizer<3>>(mesh, config);});

  // Newton's
  register_type(OptimizerType::OPTIMIZER_NEWTON, NewtonOptimizer<3>::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Optimizer<3>>
      {return std::make_unique<NewtonOptimizer<3>>(mesh, config);});

  // SQP with Bending Energy
  register_type(OptimizerType::OPTIMIZER_SQP_BENDING, MixedSQPBending::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Optimizer<3>>
      {return std::make_unique<MixedSQPBending>(mesh, config);});
}

template<>
OptimizerFactory<2>::OptimizerFactory() {
  // Newton's
  register_type(OptimizerType::OPTIMIZER_NEWTON, NewtonOptimizer<2>::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Optimizer<2>>
      {return std::make_unique<NewtonOptimizer<2>>(mesh, config);});

  register_type(OptimizerType::OPTIMIZER_SQP_PD,MixedSQPPDOptimizer<2>::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Optimizer<2>>
      {return std::make_unique<MixedSQPPDOptimizer<2>>(mesh, config);});
}

template class mfem::OptimizerFactory<3>;
template class mfem::OptimizerFactory<2>;