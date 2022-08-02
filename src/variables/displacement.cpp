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
  const auto& P = mesh_->projection_matrix();
  VectorXd xt = P.transpose()*x + b_;
  VectorXd diff = xt - integrator_->x_tilde() - h*h*f_ext_;
  const auto& MM = mesh_->template mass_matrix<FULL>();
  double e = 0.5*diff.transpose()*MM*diff;

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
void Displacement<DIM>::update(const Eigen::VectorXd&, double) {

  if (!is_mixed_) {
    double h = integrator_->dt();
    double h2 = h*h;

    VectorXd def_grad;
    const auto& P = mesh_->projection_matrix();
    mesh_->deformation_gradient(P.transpose()*x_+b_, def_grad);

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

    const auto& PMP = mesh_->template mass_matrix<PROJECTED>();
    lhs_ = PMP + assembler_->A;
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
  const auto& P = mesh_->projection_matrix();
  VectorXd xt = P.transpose()*x_ + b_;
  VectorXd diff = xt - integrator_->x_tilde() - h*h*f_ext_;

  const auto& PM = mesh_->template mass_matrix<PROJECT_ROWS>();
  grad_ = PM * diff;

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

  MatrixXd tmp = mesh_->V_.transpose();
  x_ = Map<VectorXd>(tmp.data(), mesh_->V_.size());

  IntegratorFactory factory;
  integrator_ = factory.create(config_->ti_type, x_, 0*x_, config_->h);  
  
  // Project out Dirichlet boundary conditions
  const auto& P = mesh_->projection_matrix();
  b_ = x_ - P.transpose()*P*x_;
  x_ = P * x_;
  dx_ = 0*x_;

  // If mixed, lhs_ is not modified, otherwise in unmixed systems
  // the LHS is changed each step.
  lhs_ = mesh_->template mass_matrix<mfem::MatrixType::PROJECTED>();

  // External gravity force
  VecD ext = Map<Matrix<float,DIM,1>>(config_->ext).template cast<double>();
  //f_ext_ = P_.transpose()*P_*ext.replicate(mesh_->V_.rows(),1);
  f_ext_ = ext.replicate(mesh_->V_.rows(),1);
}

template class mfem::Displacement<3>; // 3D
template class mfem::Displacement<2>; // 2D

