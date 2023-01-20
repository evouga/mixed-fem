#pragma once

#include "implicit_integrator.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <cstdio>

namespace mfem {

  template<int I>
  class BDFGpu : public ImplicitIntegrator<STORAGE_THRUST> {

  public:

    static std::string name() {
      char buf[8];
      sprintf(buf,"BDF%d",I);
      return std::string(buf);
    }

    BDFGpu(thrust::device_vector<double> x0,
        thrust::device_vector<double> v0, double h);

    const thrust::device_vector<double>& x_tilde() const override;
    double dt() const override;
    void update(const thrust::device_vector<double>& x) override;
    void reset() override;

    const thrust::device_vector<double>& x_prev() const override;
    const thrust::device_vector<double>& v_prev() const override;

  private:

    // Eigen::VectorXd weighted_sum(const std::deque<Eigen::VectorXd>& x) const;
    constexpr std::array<double,I> alphas() const;
    constexpr double beta() const;
    thrust::device_vector<double> x_tilde_;

    thrust::device_vector<double> x0_;
    thrust::device_vector<double> x1_;
    thrust::device_vector<double> v0_;
    thrust::device_vector<double> v1_;
    thrust::device_vector<double> wx_;
    thrust::device_vector<double> wv_;

    // std::deque<Eigen::VectorXd> x_prevs_;
    // std::deque<Eigen::VectorXd> v_prevs_;
  };

}
