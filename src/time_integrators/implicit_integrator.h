#pragma once

#include "EigenTypes.h"
#include <deque>

namespace mfem {

  class ImplicitIntegrator {
  public:

    ImplicitIntegrator(Eigen::VectorXd x0, Eigen::VectorXd v0, double h)
        : h_(h) {}
        
    virtual ~ImplicitIntegrator() = default;
    virtual Eigen::VectorXd x_tilde() const = 0;
    virtual double dt() const = 0;
    virtual void update(const Eigen::VectorXd& x) = 0;
    virtual void reset() {
      x_prevs_.clear();
      v_prevs_.clear();
    }
  protected:

    double h_;
    std::deque<Eigen::VectorXd> x_prevs_;
    std::deque<Eigen::VectorXd> v_prevs_;
  };

}
