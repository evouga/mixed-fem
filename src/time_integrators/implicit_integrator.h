#pragma once

#include "EigenTypes.h"
#include "config.h"
#include <deque>

namespace mfem {

  template <StorageType STORAGE = STORAGE_EIGEN>
  class ImplicitIntegrator {
  public:
    using Vector = typename Traits<STORAGE>::VectorType;

    ImplicitIntegrator(Vector x0, Vector, double h) : h_(h) {}
        
    virtual ~ImplicitIntegrator() = default;
    virtual Vector x_tilde() const = 0;
    virtual double dt() const = 0;
    virtual void update(const Vector& x) = 0;
    virtual void reset();
    virtual const Vector& x_prev() const;
    virtual const Vector& v_prev() const;
  protected:

    double h_;

  };

}
