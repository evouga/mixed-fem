#include "optimizer_factory.h"
#include "optimizers/optimizer.h"
#include "optimizers/newton_optimizer.h"
#include "optimizers/newton_optimizer_gpu.h"
#include "mesh/mesh.h"

using namespace mfem;
using namespace Eigen;

template<int DIM, StorageType STORAGE>
OptimizerFactory<DIM,STORAGE>::OptimizerFactory() {
  // Newton's
  if constexpr (STORAGE == STORAGE_EIGEN) {
    this->register_type(OptimizerType::OPTIMIZER_NEWTON,
        NewtonOptimizer<DIM>::name(),
        [](SimState<DIM,STORAGE>& state)->std::unique_ptr<Optimizer<DIM,STORAGE>>
        {return std::make_unique<NewtonOptimizer<DIM>>(state);});
  } else if constexpr (STORAGE == STORAGE_THRUST) {
    this->register_type(OptimizerType::OPTIMIZER_NEWTON,
        NewtonOptimizerGpu<DIM>::name(),
        [](SimState<DIM,STORAGE>& state)->std::unique_ptr<Optimizer<DIM,STORAGE>>
        {return std::make_unique<NewtonOptimizerGpu<DIM>>(state);});
  }
}

template class mfem::OptimizerFactory<3, STORAGE_THRUST>;
template class mfem::OptimizerFactory<3, STORAGE_EIGEN>;
template class mfem::OptimizerFactory<2, STORAGE_EIGEN>;
