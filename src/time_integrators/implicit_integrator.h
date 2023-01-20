#pragma once

#include "EigenTypes.h"
#include "config.h"
#include <deque>

namespace mfem {

  template <StorageType STORAGE = STORAGE_EIGEN>
  class ImplicitIntegrator {

    // Implicit integrator custom traits
    template <StorageType _storage, class = void>
    struct IntegratorTraits;

    template <StorageType _storage>
    struct IntegratorTraits<_storage,
        std::enable_if_t<(_storage == STORAGE_EIGEN)>> { 
      using RetType = Eigen::VectorXd;
    };

    template <StorageType _storage>
    struct IntegratorTraits<_storage,
        std::enable_if_t<(_storage == STORAGE_THRUST)>> { 
      using RetType = double*;
    };

  public:
    using Vector = typename Traits<STORAGE>::VectorType;
    using RetType = typename IntegratorTraits<STORAGE>::RetType;

    ImplicitIntegrator(double h) : h_(h) {}
        
    virtual ~ImplicitIntegrator() = default;
    virtual const Vector& x_tilde() const = 0;
    virtual double dt() const = 0;
    virtual void update(const Vector& x) = 0;
    virtual void reset() = 0;
    virtual const Vector& x_prev() const = 0;
    virtual const Vector& v_prev() const = 0;
  protected:

    double h_;

  };

}
