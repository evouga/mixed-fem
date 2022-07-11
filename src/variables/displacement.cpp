#include "displacement.h"

#include "mesh/mesh.h"
#include "energies/material_model.h"
#include "svd/newton_procrustes.h"
#include "pinning_matrix.h"
#include "config.h"
#include "factories/integrator_factory.h"
#include "time_integrators/implicit_integrator.h"

using namespace Eigen;
using namespace mfem;

template<int DIM>
Displacement<DIM>::Displacement(std::shared_ptr<Mesh> mesh,
    std::shared_ptr<SimConfig> config)
    : Variable<DIM>(mesh), config_(config), is_mixed_(true) {
}

template<int DIM>
double Displacement<DIM>::energy(const VectorXd& x) {

  double h = integrator_->dt();
  VectorXd xt = P_.transpose()*x + b_;
  VectorXd diff = xt - integrator_->x_tilde() - h*h*f_ext_;
  double e = 0.5*diff.transpose()*M_*diff;

  if (!is_mixed_) {
    VectorXd def_grad;
    mesh_->deformation_gradient(xt, def_grad);
    double e_psi = 0.0;
    #pragma omp parallel for reduction(+ : e_psi)
    for (int i = 0; i < nelem_; ++i) {
      double vol = mesh_->volumes()[i];
      const VecM& F = def_grad.segment<M()>(M()*i);
      e_psi += mesh_->material_->energy(F) * vol;
    }
    e += e_psi * h * h;
  }
  return e;
}

template<int DIM>
void Displacement<DIM>::post_solve() {
  // Update boundary positions
  BCs_.step_script(mesh_, config_->h);

  #pragma omp parallel for
  for (int i = 0; i < mesh_->V_.rows(); ++i) {
    if (mesh_->is_fixed_(i)) {
      b_.segment<DIM>(DIM*i) = mesh_->V_.row(i).transpose();
    }
  }

  VectorXd x = P_.transpose()*x_ + b_;
  integrator_->update(x);

  // Update mesh vertices
  MatrixXd V = Map<MatrixXd>(x.data(), mesh_->V_.cols(), mesh_->V_.rows());
  mesh_->V_ = V.transpose();
}

template<int DIM>
void Displacement<DIM>::update(const Eigen::VectorXd&, double) {

  if (!is_mixed_) {
    double h = integrator_->dt();
    double h2 = h*h;

    VectorXd def_grad;
    mesh_->deformation_gradient(P_.transpose()*x_+b_, def_grad);

    const std::vector<MatrixXd>& Jloc = mesh_->local_jacobians();

    // Computing per-element hessian and gradients
    #pragma omp parallel for
    for (int i = 0; i < nelem_; ++i) {
      const VecM& F = def_grad.segment<M()>(M()*i);
      double vol = mesh_->volumes()[i];
      
      H_[i] = (Jloc[i].transpose() * mesh_->material_->hessian(F)
          * Jloc[i]) * vol * h2;
      g_[i] = Jloc[i].transpose() * mesh_->material_->gradient(F) * vol * h2;
    }
    assembler_->update_matrix(H_);
    lhs_ = PMP_ + assembler_->A;
  }
}

template<int DIM>
VectorXd Displacement<DIM>::rhs() {
  data_.timer.start("RHS - x");
  rhs_ = -gradient();
  data_.timer.stop("RHS - x");
  return rhs_;
}

template<int DIM>
VectorXd Displacement<DIM>::gradient() {
  double h = integrator_->dt();
  double h2 = h*h;
  VectorXd xt = P_.transpose()*x_ + b_;
  VectorXd diff = xt - integrator_->x_tilde() - h*h*f_ext_;
  grad_ = PM_ * diff;

  if (!is_mixed_) {
    // Assuming update() was called to update per-element gradients, g_
    VectorXd g;
    vec_assembler_->assemble(g_, g);
    grad_ += g;
  }

  return grad_;
}

template<int DIM>
void Displacement<DIM>::reset() {
  nelem_ = mesh_->T_.rows();

  H_.resize(nelem_);
  g_.resize(nelem_);
  Aloc_.resize(nelem_);
  assembler_ = std::make_shared<Assembler<double,DIM,-1>>(
      mesh_->T_, mesh_->free_map_);
  vec_assembler_ = std::make_shared<VecAssembler<double,DIM,-1>>(mesh_->T_,
      mesh_->free_map_);

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    H_[i].setIdentity();
    g_[i].setZero();
  }

  mesh_->V_ = mesh_->V0_;
  mesh_->clear_fixed_vertices();

  BoundaryConditions<3>::init_boundary_groups(mesh_->V0_,
      mesh_->bc_groups_, 0.01); // .01, hang for astronaut

  BCs_.set_script(config_->bc_type);
  BCs_.init_script(mesh_);
  P_ = pinning_matrix(mesh_->V_, mesh_->T_, mesh_->is_fixed_, false);

  mesh_->mass_matrix(M_, mesh_->volumes());

  MatrixXd tmp = mesh_->V_.transpose();
  x_ = Map<VectorXd>(tmp.data(), mesh_->V_.size());

  IntegratorFactory factory;
  integrator_ = factory.create(config_->ti_type, x_, 0*x_, config_->h);  
  
  b_ = x_ - P_.transpose()*P_*x_;
  x_ = P_ * x_;
  dx_ = 0*x_;

  // Project out mass matrix pinned point
  PMP_ = P_ * M_ * P_.transpose();
  PM_ = P_ * M_;

  // If mixed, lhs_ is not modified, otherwise in unmixed systems
  // the LHS is changed each step.
  lhs_ = PMP_;

  // External gravity force
  Vector3d ext = Map<Vector3f>(config_->ext).cast<double>();
  f_ext_ = P_.transpose()*P_*ext.replicate(mesh_->V_.rows(),1);
}

template class mfem::Displacement<3>; // 3D
//template class mfem::Displacement<2>; // 2D

