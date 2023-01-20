#include "displacement_gpu.h"

using namespace mfem;
using namespace Eigen;

// typedef DisplacementGpu<DIM>::VectorType VectorType;

template<int DIM>
DisplacementGpu<DIM>::DisplacementGpu(std::shared_ptr<Mesh> mesh,
    std::shared_ptr<SimConfig> config)
    : Variable<DIM,STORAGE_THRUST>(mesh), config_(config) {
}

template<int DIM>
double DisplacementGpu<DIM>::energy(VectorType& x) {
  return 0;
}

template<int DIM>
void DisplacementGpu<DIM>::post_solve() {
}

template<int DIM>
void DisplacementGpu<DIM>::update(VectorType&, double) {}

template<int DIM>
DisplacementGpu<DIM>::VectorType DisplacementGpu<DIM>::rhs() {
  OptimizerData::get().timer.start("rhs", "DisplacementGpu");
  // rhs_ = -gradient();
  OptimizerData::get().timer.stop("rhs", "DisplacementGpu");
  return rhs_;
}

template<int DIM>
DisplacementGpu<DIM>::VectorType DisplacementGpu<DIM>::gradient() {
  OptimizerData::get().timer.start("gradient", "DisplacementGpu");
  std::cout << "Displacement gradient not implemented yet" << std::endl;
  OptimizerData::get().timer.stop("gradient", "DisplacementGpu");
  return grad_;
}

template<int DIM>
void DisplacementGpu<DIM>::reset() {

//   MatrixXd tmp = mesh_->V_.transpose();
//   x_ = Map<VectorXd>(tmp.data(), mesh_->V_.size());

//   tmp = mesh_->initial_velocity_.transpose();
//   VectorXd v0 = Map<VectorXd>(tmp.data(), mesh_->V_.size());

//   IntegratorFactory factory;
//   integrator_ = factory.create(config_->ti_type, x_, v0, config_->h);  
  
//   // Project out Dirichlet boundary conditions
//   const auto& P = mesh_->projection_matrix();
//   b_ = x_ - P.transpose()*P*x_;
//   x_ = P * x_;
//   dx_ = 0*x_;

//   lhs_ = mesh_->template mass_matrix<MatrixType::PROJECTED>();
}

template class mfem::DisplacementGpu<3>; // 3D
