#include "collision.h"
#include "mesh/mesh.h"
#include "igl/unique.h"
#include "igl/boundary_facets.h"
// #include "simple_psd_fix.h"
#include "config.h"
#include <ipc/barrier/barrier.hpp>
#include "igl/edges.h"

using namespace Eigen;
using namespace mfem;

template<int DIM>
double Collision<DIM>::energy(const VectorXd& x) {

  MatrixXd V = Map<const MatrixXd>(x.data(), mesh_->V_.cols(),
      mesh_->V_.rows());
  V.transposeInPlace();
  
  MatrixXi tmp;  
  ipc::Constraints constraint_set;
  ipc::construct_constraint_set(ipc_mesh_, V, config_->dhat, constraint_set);

  double h2 = dt_ * dt_;
  double e = 0;

  #pragma omp parallel for reduction( + : e )
  for (size_t i = 0; i < constraint_set.size(); ++i) {
    std::array<long, 4> ids = constraint_set[i].vertex_indices(E_, tmp);
    double d = constraint_set[i].compute_distance(V, E_, tmp);
    double dhat_sqr = config_->dhat * config_->dhat;
    e += config_->kappa * ipc::barrier(d, dhat_sqr) / h2;
  }
  return e;
}

template<int DIM>
void Collision<DIM>::update(const Eigen::VectorXd& x, double dt) {
  // Get collision frames
  dt_ = dt;

  MatrixXd V = Map<const MatrixXd>(x.data(), mesh_->V_.cols(),
      mesh_->V_.rows());
  V.transposeInPlace();
  
  MatrixXi tmp;  
  ipc::construct_constraint_set(ipc_mesh_, V, config_->dhat, constraint_set_);

  nframes_ = constraint_set_.size();
  MatrixXi T(nframes_, 4);

  for (size_t i = 0; i < constraint_set_.size(); ++i) {
    std::array<long, 4> ids = constraint_set_[i].vertex_indices(E_, tmp);
    for (int j = 0; j < 4; ++j) {
      T(i,j) = ids[j];
    }
  }

  // Structure potentially changes each step, so just rebuild assembler :/
  // NOTE assuming each local jacobian has same size!
  assembler_ = std::make_shared<Assembler<double,DIM,-1>>(T, mesh_->free_map_);
  vec_assembler_ = std::make_shared<VecAssembler<double,DIM,-1>>(T,
      mesh_->free_map_);
  update_derivatives(V, dt);
}

template<int DIM>
void Collision<DIM>::update_collision_frames(const Eigen::VectorXd& x) {}

template<int DIM>
void Collision<DIM>::update_derivatives(const MatrixXd& V, double dt) {

  double h2 = dt * dt;

  if (nframes_ == 0) {
    return;
  }
  MatrixXi tmp;
  
  // Hessian with respect to x variable
  data_.timer.start("Local H");
  Aloc_.resize(nframes_);
  #pragma omp parallel for
  for (int i = 0; i < nframes_; ++i) {
    Aloc_[i] = config_->kappa * constraint_set_[i].compute_potential_hessian(
        V, E_, tmp, config_->dhat, true);
  }
  data_.timer.stop("Local H");

  data_.timer.start("Update LHS");
  assembler_->update_matrix(Aloc_);
  data_.timer.stop("Update LHS");

  // saveMarket(assembler_->A, "lhs_c1.mkt");
  A_ = assembler_->A;

  // Gradient with respect to x variable
  gloc_.resize(nframes_);
  for (int i = 0; i < nframes_; ++i) {
    gloc_[i] = config_->kappa * constraint_set_[i].compute_potential_gradient(
        V, E_, tmp, config_->dhat);
  }
  vec_assembler_->assemble(gloc_, grad_);
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

  igl::boundary_facets(mesh_->T_, F_);
  assert(F_.cols() == 2); // Only supports 2D right now
  igl::unique(F_,C_);
  igl::edges(mesh_->T_, E_);

  if constexpr (DIM ==2) {
    // TODO use "include_vertex"
    // TODO use the boundary facets
    MatrixXi tmp;
    ipc_mesh_ = ipc::CollisionMesh::build_from_full_mesh(mesh_->V_, E_, tmp);
  } else {
    std::cerr << "SHIT BRUH" << std::endl;
  } 
}

template<int DIM>
void Collision<DIM>::post_solve() {}

template class mfem::Collision<3>; // 3D
template class mfem::Collision<2>; // 2D
