#include "linear_system.h"

using namespace mfem;
using namespace Eigen;


template <typename Scalar>
void SystemMatrixThrustCpu<Scalar>::pre_solve(
    const SimState<3,STORAGE_THRUST>* state) {
  // Add LHS and RHS from each variable
  lhs_ = state->x_->lhs();

  auto& rhs = state->x_->rhs();
  rhs_.resize(rhs.size());
  rhs_.setZero();
  // std::cout << "x rhs size: " << rhs.size() << std::endl;

  // Copy from device rhs to host rhs_
  cudaMemcpy(rhs_.data(), thrust::raw_pointer_cast(rhs.data()),
      rhs.size()*sizeof(double), cudaMemcpyDeviceToHost);
  std::cout << "DisplacementGpu rhs: " << rhs_.norm() << std::endl;

  // std::cout << "lhs rows: " << lhs_.rows() << " cols: " << lhs_.cols() << std::endl;
  // std::cout << "x rhs: " << rhs_.norm() << "size: " << rhs.size() << std::endl;
  VectorXd rhs_tmp(rhs_.size());

  for (auto& var : state->mixed_vars_) {
    lhs_ += var->lhs();

    auto& rhs_s = var->rhs();
    // std::cout << "rhs s size: " << rhs_s.size() << std::endl;
    cudaMemcpy(rhs_tmp.data(), thrust::raw_pointer_cast(rhs_s.data()),
        rhs_s.size()*sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Mixed rhs: " << rhs_tmp.norm() << std::endl;
    rhs_ += rhs_tmp;
  //   std::cout << "rhs tmp size: " << rhs_tmp.size() << std::endl;
  //   std::cout << "s rhs: " << rhs_tmp.norm() << std::endl;
  }
}

template <typename Scalar>
void SystemMatrixThrustCpu<Scalar>::post_solve(
    const SimState<3,STORAGE_THRUST>* state, VectorXd& dx) {

  // std::cout << "dx norm: " << dx.norm() << std::endl;
  // state->x_->delta() = dx;
  double* dx_ptr = thrust::raw_pointer_cast(state->x_->delta().data());

  // Copy from host dx to device dx_ptr
  cudaMemcpy(dx_ptr, dx.data(), dx.size()*sizeof(double),
      cudaMemcpyHostToDevice);

  for (auto& var : state->mixed_vars_) {
    var->solve(state->x_->delta());
  }
}

template void mfem::SystemMatrixThrustCpu<double>::pre_solve(const SimState<3, STORAGE_THRUST>*);
template void mfem::SystemMatrixThrustCpu<double>::post_solve(const SimState<3, STORAGE_THRUST>*, VectorXd&);
