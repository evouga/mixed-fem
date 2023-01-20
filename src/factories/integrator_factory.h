#pragma once

#include "factory.h"
#include "config.h"
#include "time_integrators/implicit_integrator.h"

namespace mfem {
  template <StorageType STORAGE>
  class IntegratorFactory : public Factory<TimeIntegratorType,
      ImplicitIntegrator<STORAGE>,
      typename ImplicitIntegrator<STORAGE>::Vector,
      typename ImplicitIntegrator<STORAGE>::Vector, double> {
  public:
    IntegratorFactory();
  };
}
