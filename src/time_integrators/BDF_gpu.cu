#include "BDF_gpu.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

using namespace mfem;
using namespace Eigen;

using namespace thrust::placeholders;

template <int I>
BDFGpu<I>::BDFGpu(double* x0, double* v0, int size, double h) 
    : ImplicitIntegrator<STORAGE_THRUST>(h), size_(size) {

  // Init device vectors
  wx_.resize(size);
  wv_.resize(size);
  x0_.resize(size);
  x1_.resize(size);
  v0_.resize(size);
  v1_.resize(size);
  x0_ptr = thrust::raw_pointer_cast(x0_.data());
  v0_ptr = thrust::raw_pointer_cast(v0_.data());

  // Copy x0 to x0_ and x1_
  cudaMemcpy(thrust::raw_pointer_cast(x0_.data()), x0,
      size * sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpy(thrust::raw_pointer_cast(x1_.data()), x0,
      size * sizeof(double), cudaMemcpyDeviceToDevice);

  // Copy v0 to v0_ and v1_
  cudaMemcpy(thrust::raw_pointer_cast(v0_.data()), v0,
      size * sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpy(thrust::raw_pointer_cast(v1_.data()), v0,
      size * sizeof(double), cudaMemcpyDeviceToDevice);
  
  // Initialize x_tilde_
  x_tilde_.resize(size);
  x_tilde_ptr_ = thrust::raw_pointer_cast(x_tilde_.data());
  thrust::transform(x0_.begin(), x0_.end(), v0_.begin(), x_tilde_.begin(),
      _1 + dt() * _2);
}

template <int I>
double* const & BDFGpu<I>::x_tilde() const {
  return x_tilde_ptr_;
}

template <int I>
double BDFGpu<I>::dt() const {
  return beta() * h_;
}

template <int I>
void BDFGpu<I>::update(double* const& x) {
  // BDF1
  if constexpr (I == 1) {
    wx_ = x0_;

    // Copy x to x0_
    cudaMemcpy(thrust::raw_pointer_cast(x0_.data()), x,
        size_ * sizeof(double), cudaMemcpyDeviceToDevice);

    // v0 = (x - wx) / h
    thrust::transform(x0_.begin(), x0_.end(), wx_.begin(), v0_.begin(),
        (_1 - _2) / h_);

    // x_tilde = x0 + dt * v0
    thrust::transform(x0_.begin(), x0_.end(), v0_.begin(), x_tilde_.begin(),
        _1 + dt() * _2);

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
double* const& BDFGpu<I>::x_prev() const {
  return x0_ptr;
}

template <int I>
double* const& BDFGpu<I>::v_prev() const {
  return v0_ptr;
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