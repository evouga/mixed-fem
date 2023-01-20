#pragma once

#include "implicit_integrator.h"
#include <cstdio>

namespace mfem {

  template<int I>
  class BDF : public ImplicitIntegrator<STORAGE_EIGEN> {

  public:

    static std::string name() {
      char buf[8];
      sprintf(buf,"BDF%d",I);
      return std::string(buf);
    }

    BDF(Eigen::VectorXd x0, Eigen::VectorXd v0, double h)
        : ImplicitIntegrator<STORAGE_EIGEN>(x0, v0, h) {
      static_assert(I >= 1 && I <= 6, "Only BDF1 - BDF6 are supported");
      for (int i = 0; i < I; ++i) {
        x_prevs_.push_front(x0);
        v_prevs_.push_front(v0);
      }
    }

    Eigen::VectorXd x_tilde() const override;
    double dt() const override;
    void update(const Eigen::VectorXd& x) override;

    virtual void reset() override {
      x_prevs_.clear();
      v_prevs_.clear();
    }

    virtual const Eigen::VectorXd& x_prev() const override {
      return x_prevs_.front();
    }
    virtual const Eigen::VectorXd& v_prev() const override {
      return v_prevs_.front();
    }

  private:

    Eigen::VectorXd weighted_sum(const std::deque<Eigen::VectorXd>& x) const;
    constexpr std::array<double,I> alphas() const;
    constexpr double beta() const;

    std::deque<Eigen::VectorXd> x_prevs_;
    std::deque<Eigen::VectorXd> v_prevs_;
  };

}
