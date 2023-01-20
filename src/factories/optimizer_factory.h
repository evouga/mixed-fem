#pragma once

#include "factory.h"
#include "config.h"
#include "optimizers/optimizer.h"

namespace mfem {

  class Mesh;

  template<int DIM, StorageType STORAGE = STORAGE_EIGEN>
  class OptimizerFactory : public Factory<OptimizerType,
      Optimizer<DIM,STORAGE>, SimState<DIM,STORAGE>&> {
  public:
    OptimizerFactory();
  };
}
