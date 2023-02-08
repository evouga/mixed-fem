#include "displacement.h"

#include "mesh/mesh.h"
#include "energies/material_model.h"
#include "svd/newton_procrustes.h"
#include "utils/pinning_matrix.h"
#include "config.h"
#include "factories/integrator_factory.h"
#include "time_integrators/implicit_integrator.h"

using namespace Eigen;
using namespace mfem;

template<int DIM>
Displacement<DIM>::Displacement(std::shared_ptr<Mesh> mesh,
    std::shared_ptr<SimConfig> config)
    : Variable<DIM>(mesh), config_(config) {
}

template<int DIM>
double Displacement<DIM>::energy(VectorXd& x) {

  double h = integrator_->dt();
  const auto& P = mesh_->projection_matrix();
  VectorXd xt = P.transpose()*x + b_;
  VectorXd diff = xt - integrator_->x_tilde() 
      - (h*h) * mesh_->external_force();
  const auto& MM = mesh_->template mass_matrix<MatrixType::FULL>();
  double e = 0.5*diff.transpose()*MM*diff;
  return e;
}

template<int DIM>
void Displacement<DIM>::post_solve() {
  // TODO b_,/BCs should not be here, boundary conditions
  // probably should be owned by either optimizer or mesh
  #pragma omp parallel for
  for (int i = 0; i < mesh_->V_.rows(); ++i) {
    if (mesh_->is_fixed_(i)) {
      b_.segment<DIM>(DIM*i) = mesh_->V_.row(i).transpose();
    }
  }

  const auto& P = mesh_->projection_matrix();
  VectorXd x = P.transpose()*x_ + b_;
  integrator_->update(x);

  // Update mesh vertices
  MatrixXd V = Map<MatrixXd>(x.data(), mesh_->V_.cols(), mesh_->V_.rows());
  mesh_->V_ = V.transpose();
}

template<int DIM>
void Displacement<DIM>::update(Eigen::VectorXd&, double) {}

template<int DIM>
VectorXd& Displacement<DIM>::rhs() {
  OptimizerData::get().timer.start("rhs", "Displacement");
  rhs_ = -gradient();
  OptimizerData::get().timer.stop("rhs", "Displacement");
  return rhs_;
}

template<int DIM>
VectorXd Displacement<DIM>::gradient() {
  OptimizerData::get().timer.start("gradient", "Displacement");

  double h = integrator_->dt();
  const auto& P = mesh_->projection_matrix();
  VectorXd xt = P.transpose()*x_ + b_;
  VectorXd diff = xt - integrator_->x_tilde() 
      - (h*h) * mesh_->external_force();

  const auto& PM = mesh_->template mass_matrix<MatrixType::PROJECT_ROWS>();
  grad_ = PM * diff;
  OptimizerData::get().timer.stop("gradient", "Displacement");
  return grad_;
}

template<int DIM>
void Displacement<DIM>::reset() {

  MatrixXd tmp = mesh_->V_.transpose();
  x_ = Map<VectorXd>(tmp.data(), mesh_->V_.size());

  tmp = mesh_->initial_velocity_.transpose();
  VectorXd v0 = Map<VectorXd>(tmp.data(), mesh_->V_.size());

  IntegratorFactory<STORAGE_EIGEN> factory;
  integrator_ = factory.create(config_->ti_type, x_, v0, x_.size(),
      config_->h);  
  
  // Project out Dirichlet boundary conditions
  const auto& P = mesh_->projection_matrix();
  b_ = x_ - P.transpose()*P*x_;
  x_ = P * x_;
  dx_ = 0*x_;

  lhs_ = mesh_->template mass_matrix<MatrixType::PROJECTED>();
}

template class mfem::Displacement<3>; // 3D
template class mfem::Displacement<2>; // 2D

