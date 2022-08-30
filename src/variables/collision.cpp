#include "collision.h"
#include "mesh/mesh.h"
#include "config.h"
#include <ipc/barrier/barrier.hpp>

using namespace Eigen;
using namespace mfem;

template<int DIM>
double Collision<DIM>::energy(const VectorXd& x) {

  // Convert configuration vector to matrix form
  MatrixXd V = Map<const MatrixXd>(x.data(), DIM, mesh_->V_.rows());
  V.transposeInPlace();

  // Get IPC mesh and face/edge/vertex data
  const auto& ipc_mesh = mesh_->collision_mesh();
  const Eigen::MatrixXi& E = ipc_mesh.edges();
  const Eigen::MatrixXi& F = ipc_mesh.faces();
  V = ipc_mesh.vertices(V);

  // Computing collision constraints
  ipc::Constraints constraints;
  ipc::Candidates candidates;
  ipc::construct_collision_candidates(
      ipc_mesh, V, candidates, config_->dhat * 1.1);
  ipc::construct_constraint_set(candidates, ipc_mesh, V,
      config_->dhat, constraints);


  double dhat_sqr = config_->dhat * config_->dhat;
  double h2 = dt_ * dt_;
  double e = 0;

  // Computing barrier potential for all collision frames
  #pragma omp parallel for reduction( + : e )
  for (size_t i = 0; i < constraints.size(); ++i) {
    double d = constraints[i].compute_distance(V, E, F);
    e += config_->kappa * ipc::barrier(d, dhat_sqr) / h2;
  }
  return e;
}

template<int DIM>
void Collision<DIM>::update(const Eigen::VectorXd& x, double dt) {
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

  // Computing collision constraints
  // ipc::construct_constraint_set(ipc_mesh, V, config_->dhat, constraints_);
  ipc::Candidates candidates;
  ipc::construct_collision_candidates(
      ipc_mesh, V, candidates, config_->dhat * 1.1);
  ipc::construct_constraint_set(candidates, ipc_mesh, V,
      config_->dhat, constraints_);

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

  // Compute gradient and hessian
  update_derivatives(V, dt);
}

template<int DIM>
void Collision<DIM>::update_derivatives(const MatrixXd& V, double dt) {

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

  // Hessian and gradient with respect to x
  data_.timer.start("g-H");
  #pragma omp parallel for
  for (int i = 0; i < nframes_; ++i) {
    Aloc[i] = config_->kappa * constraints_[i].compute_potential_hessian(
        V, E, F, config_->dhat, true);
    gloc[i] = config_->kappa * constraints_[i].compute_potential_gradient(
        V, E, F, config_->dhat);
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
VectorXd Collision<DIM>::rhs() {
  data_.timer.start("RHS - s");
  return -gradient();
}

template<int DIM>
VectorXd Collision<DIM>::gradient() {
  if (nframes_ == 0) {
    grad_.resize(mesh_->jacobian().rows());
    grad_.setZero();
  }
  return grad_;
}

template<int DIM>
void Collision<DIM>::reset() {
  grad_.resize(0);
}

template<int DIM>
void Collision<DIM>::post_solve() {}

template class mfem::Collision<3>; // 3D
template class mfem::Collision<2>; // 2D
