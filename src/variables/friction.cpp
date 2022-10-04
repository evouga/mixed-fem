#include "friction.h"
#include "mesh/mesh.h"
#include "config.h"
#include <ipc/barrier/barrier.hpp>

using namespace Eigen;
using namespace mfem;

template<int DIM>
double Friction<DIM>::energy(const VectorXd& x) {

  // Convert configuration vector to matrix form
  MatrixXd V = Map<const MatrixXd>(x.data(), DIM, mesh_->V_.rows());
  V.transposeInPlace();

  // Get IPC mesh and face/edge/vertex data
  const auto& ipc_mesh = mesh_->collision_mesh();
  const Eigen::MatrixXi& E = ipc_mesh.edges();
  const Eigen::MatrixXi& F = ipc_mesh.faces();
  V = ipc_mesh.vertices(V);

  double e = ipc::compute_friction_potential(
      ipc_mesh, V0_, V, constraints_, config_->espv * dt_);
      std::cout << "FRICTION E: " << e << std::endl;
  return e / dt_ / dt_;
}

template<int DIM>
void Friction<DIM>::update(const Eigen::VectorXd& x, double dt) {
  // Get collision frames
  dt_ = dt;

  // Convert configuration vector to matrix form
  MatrixXd V = Map<const MatrixXd>(x.data(), DIM, mesh_->V_.rows());
  V.transposeInPlace();

  // Get IPC mesh and face/edge/vertex data
  const auto& ipc_mesh = mesh_->collision_mesh();
  const Eigen::MatrixXi& E = ipc_mesh.edges();
  const Eigen::MatrixXi& F = ipc_mesh.faces();
  V = ipc_mesh.vertices(V);

  // Compute gradient and hessian
  update_derivatives(V - V0_, dt);
}

template<int DIM>
void Friction<DIM>::update_derivatives(const MatrixXd& U, double dt) {

  if (nframes_ == 0) {
    return;
  }

  // vector of local hessians and gradients
  std::vector<Eigen::MatrixXd> Aloc(nframes_);
  std::vector<Eigen::VectorXd> gloc(nframes_); 

  // Get IPC mesh and face/edge/vertex data
  const auto& ipc_mesh = mesh_->collision_mesh();
  const Eigen::MatrixXi& E = ipc_mesh.edges();
  const Eigen::MatrixXi& F = ipc_mesh.faces();

  double epsv_h = config_->espv * dt_;

  // Hessian and gradient with respect to x
  data_.timer.start("g-H");
  #pragma omp parallel for
  for (int i = 0; i < nframes_; ++i) {
    Aloc[i] = constraints_[i].compute_potential_hessian(
        U, E, F, epsv_h, true);
    gloc[i] = constraints_[i].compute_potential_gradient(
        U, E, F, epsv_h);
  }
  data_.timer.stop("g-H");

  // Assemble hessian
  data_.timer.start("Update LHS");
  assembler_->update_matrix(Aloc);
  A_ = assembler_->A;
  data_.timer.stop("Update LHS");

  // Assemble gradient
  vec_assembler_->assemble(gloc, grad_);
}

template<int DIM>
VectorXd Friction<DIM>::rhs() {
  data_.timer.start("RHS - s");
  return -gradient();
}

template<int DIM>
VectorXd Friction<DIM>::gradient() {
  if (nframes_ == 0) {
    grad_.resize(mesh_->jacobian().rows());
    grad_.setZero();
  }
  return grad_;
}

template<int DIM>
void Friction<DIM>::reset() {
  grad_.resize(0);
}

template<int DIM>
void Friction<DIM>::pre_solve() {
  // Convert configuration vector to matrix form
  V0_ = mesh_->V_;

  // Get IPC mesh and face/edge/vertex data
  const auto& ipc_mesh = mesh_->collision_mesh();
  const Eigen::MatrixXi& E = ipc_mesh.edges();
  const Eigen::MatrixXi& F = ipc_mesh.faces();
  V0_ = ipc_mesh.vertices(V0_);

  // Computing collision constraints
  ipc::Candidates candidates;
  ipc::Constraints constraints;
  ipc::construct_collision_candidates(
      ipc_mesh, V0_, candidates, config_->dhat * 1.1);
  ipc::construct_constraint_set(candidates, ipc_mesh, V0_,
      config_->dhat, constraints);

  ipc::construct_friction_constraint_set(
      ipc_mesh, V0_, constraints, config_->dhat, config_->kappa,
      config_->mu, constraints_);

  nframes_ = constraints_.size();

  // Create element matrix for assemblers
  MatrixXi T(nframes_, 4);
  for (size_t i = 0; i < constraints_.size(); ++i) {
    std::array<long, 4> ids = constraints_[i].vertex_indices(E, F);
    for (int j = 0; j < 4; ++j) {
      T(i,j) = ids[j];
    }
  }

  // Remaking assemblers since collision frames change.
  assembler_ = std::make_shared<Assembler<double,DIM,-1>>(T, mesh_->free_map_);
  vec_assembler_ = std::make_shared<VecAssembler<double,DIM,-1>>(T,
      mesh_->free_map_);
}

template class mfem::Friction<3>; // 3D
template class mfem::Friction<2>; // 2D
