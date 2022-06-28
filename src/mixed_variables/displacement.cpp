#include "displacement.h"

#include "mesh/mesh.h"
#include "energies/material_model.h"
#include "svd/newton_procrustes.h"
#include "config.h"
#include "time_integrators/BDF.h"
#include "pinning_matrix.h"

using namespace Eigen;
using namespace mfem;

template class mfem::Displacement<3>; // 3D
//template class mfem::Displacement<2>; // 2D

template<int DIM>
Displacement<DIM>::Displacement(std::shared_ptr<Mesh> mesh,
    std::shared_ptr<SimConfig> config)
    : MixedVariable<DIM>(mesh), config_(config) {

  integrator_ = std::make_shared<BDF<1>>(config->h);
}

template<int DIM>
double Displacement<DIM>::energy(const VectorXd& x) {
  double h = integrator_->dt();

  VectorXd xt = P_.transpose()*x + b_;
  VectorXd diff = xt - integrator_->x_tilde() - h*h*f_ext_;
  double e = 0.5*diff.transpose()*M_*diff;
  return e;
}

template<int DIM>
double Displacement<DIM>::constraint_value(const VectorXd& x,
    const VectorXd& s) {
  return 0;
}

template<int DIM>
void Displacement<DIM>::update(const Eigen::VectorXd& x, double dt) {
  // If non-mixed compute derivatives & assemble
}

template<int DIM>
VectorXd Displacement<DIM>::rhs() {
  data_.timer.start("RHS - x");
  double h = integrator_->dt();

  VectorXd xt = P_.transpose()*x_ + b_;
  VectorXd diff = xt - integrator_->x_tilde() - h*h*f_ext_;
  rhs_ = - PM_ * diff;

  data_.timer.stop("RHS - x");
  return rhs_;
}

template<int DIM>
VectorXd Displacement<DIM>::gradient() {
  double h = integrator_->dt();
  VectorXd xt = P_.transpose()*x_ + b_;
  VectorXd diff = xt - integrator_->x_tilde() - h*h*f_ext_;
  grad_ = M_ * diff;
  return grad_;
}

template<int DIM>
void Displacement<DIM>::solve(const VectorXd& dx) {
}

template<int DIM>
void Displacement<DIM>::reset() {
  nelem_ = mesh_->T_.rows();

  H_.resize(nelem_);
  g_.resize(nelem_);
  Aloc_.resize(nelem_);
  assembler_ = std::make_shared<Assembler<double,DIM>>(
      mesh_->T_, mesh_->free_map_);

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    H_[i].setIdentity();
    g_[i].setZero();
  }

  P_ = pinning_matrix(mesh_->V_, mesh_->T_, mesh_->is_fixed_, false);

  mesh_->mass_matrix(M_, mesh_->volumes());

  MatrixXd tmp = mesh_->V_.transpose();
  x_ = Map<VectorXd>(tmp.data(), mesh_->V_.size());
  b_ = x_ - P_.transpose()*P_*x_;
  x_ = P_ * x_;
  dx_ = 0*x_;


  // Project out mass matrix pinned point
  PMP_ = P_ * M_ * P_.transpose();
  PM_ = P_ * M_;

  // External gravity force
  Vector3d ext = Map<Vector3f>(config_->ext).cast<double>();
  f_ext_ = P_.transpose()*P_*ext.replicate(mesh_->V_.rows(),1);
}

