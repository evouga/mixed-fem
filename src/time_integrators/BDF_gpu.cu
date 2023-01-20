#include "BDF_gpu.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

using namespace mfem;
using namespace Eigen;

using namespace thrust::placeholders;

template <int I>
BDFGpu<I>::BDFGpu(thrust::device_vector<double> x0,
    thrust::device_vector<double> v0, double h) 
    : ImplicitIntegrator<STORAGE_THRUST>(h) {
    //   static_assert(I >= 1 && I <= 6, "Only BDF1 - BDF2 are supported");
    //   for (int i = 0; i < I; ++i) {
    //     x_prevs_.push_front(x0);
    //     v_prevs_.push_front(v0);
    //   }
    // }
    // x0_ = x0;
    // x1_ = x0;
    // v0_ = v0;
    // v1_ = v0;
}

template <int I>
const thrust::device_vector<double>& BDFGpu<I>::x_tilde() const {
  return x_tilde_;
}

template <int I>
double BDFGpu<I>::dt() const {
  return beta() * h_;
}

template <int I>
void BDFGpu<I>::update(const thrust::device_vector<double>& x) {
  // BDF1
  if constexpr (I == 1) {
    // wx_ = x0_;

    // wx = x0
    // x0 = x
    // v0 = (x - wx) / h
    // x_tilde = x0 + dt * v0

  } else {
    std::cout << "BDF2 currently unsupported! ! ! Exiting " << std::endl;
    exit(1);
    // BDF2
    // wx = 4/3 * x0 - 1/3 * x1
    // x1 = x0
    // x0 = x
    // v1 = v0
    // v0 = (x - wx) / h

    // wx = 4/3 * x0 - 1/3 * x1
    // wv = 4/3 * v0 - 1/3 * v1

    // x_tilde = wx + dt * wv
  }
}

template <int I>
void BDFGpu<I>::reset() {
  std::cout << "BDFGpu::reset() not implemented! ! !" << std::endl;
}

template <int I>
const thrust::device_vector<double>& BDFGpu<I>::x_prev() const {
  return x0_;
}

template <int I>
const thrust::device_vector<double>& BDFGpu<I>::v_prev() const {
  return v0_;
}

// Specializations for each BDF integrator
template <>
constexpr std::array<double,1> BDFGpu<1>::alphas() const {
  return {1.0};
}

template <int I>
constexpr double BDFGpu<I>::beta() const {
  switch(I) {
    case 1:
      return 1.0;
    case 2:
      return 2.0 / 3.0;
  }
}

template class mfem::BDFGpu<1>;
// template class mfem::BDFGpu<2>;