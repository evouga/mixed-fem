#include "mixed_friction.h"
#include "mesh/mesh.h"
#include "config.h"
#include <ipc/barrier/barrier.hpp>

using namespace Eigen;
using namespace mfem;

template<int DIM>
double MixedFriction<DIM>::energy(VectorXd& x, VectorXd& z) {
  // double e = ipc::compute_friction_potential(
  //     ipc_mesh, V0_, V, constraints_, config_->espv * dt_ * dt_);
  //     // std::cout << "FRICTION E: " << e << std::endl;
  double e = 0;
  #pragma omp parallel for reduction(+:e)
  for (size_t i = 0; i < constraints_.size(); ++i) {
    e += constraints_[i].potential(z(i), config_->espv * dt_ * dt_);
  }
  // std::cout << "MF: " << e << std::endl;
  return e / dt_ / dt_;
}

template <int DIM>
double MixedFriction<DIM>::constraint_value(const VectorXd& x,
    const VectorXd& z) {

  if (constraints_.size() == 0) {
    return 0;
  }
  // Convert configuration vector to matrix form
  MatrixXd V = Map<const MatrixXd>(x.data(), DIM, mesh_->V_.rows());
  V.transposeInPlace();

  // Get IPC mesh and face/edge/vertex data
  const auto& ipc_mesh = mesh_->collision_mesh();
  const Eigen::MatrixXi& E = ipc_mesh.edges();
  const Eigen::MatrixXi& F = ipc_mesh.faces();
  V = ipc_mesh.vertices(V);

  V -= V0_; // Compute U

  double e = 0;
  #pragma omp parallel for reduction(+:e)
  for (size_t i = 0; i < constraints_.size(); ++i) {
    double u_norm = constraints_[i].u_norm(V,E,F);
    e += la_(i) * (u_norm - z(i));
  }
  // std::cout << "MF c: " << e << std::endl;

  return e;
}

template<int DIM>
void MixedFriction<DIM>::update(Eigen::VectorXd& x, double dt) {
  // Get collision frames
  dt_ = dt;

  if (constraints_.size() == 0) {
    return;
  }

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
void MixedFriction<DIM>::update_derivatives(const MatrixXd& U, double dt) {

  size_t num_frames = constraints_.size();

  // vector of local hessians and gradients
  std::vector<Eigen::MatrixXd> Aloc(num_frames);
  std::vector<Eigen::VectorXd> gloc(num_frames); 

  // Get IPC mesh and face/edge/vertex data
  const auto& ipc_mesh = mesh_->collision_mesh();
  const Eigen::MatrixXi& E = ipc_mesh.edges();
  const Eigen::MatrixXi& F = ipc_mesh.faces();

  double epsv_h = config_->espv * dt_ * dt_;

  // Hessian and gradient with respect to x
  // #pragma omp parallel for
  for (size_t i = 0; i < num_frames; ++i) {
    Z_(i) = constraints_[i].u_norm(U, E, F);
    Gx_[i] = constraints_[i].u_norm_gradient(U, E, F);
    g_(i) = constraints_[i].potential_gradient(z_(i), epsv_h);
    H_(i) = constraints_[i].potential_hessian(z_(i), epsv_h);
    Aloc[i] = Gx_[i] * H_(i) * Gx_[i].transpose();

    // Compute right-hand side
    double Gz_inv_sqr = 1.0 / (Gz_(i) * Gz_(i));
    double gl = Gz_inv_sqr * H_(i) * (Z_(i) - z_(i)) - g_(i) / Gz_(i);
    gloc[i] = -Gx_[i] * gl;
  }

  // Assemble hessian
  assembler_->update_matrix(Aloc);
  A_ = assembler_->A;

  // Assemble right hand side
  vec_assembler_->assemble(gloc, rhs_);
  // std::cout << "Z: " << Z_.transpose() << std::endl;
  // std::cout << "z: " << z_.transpose() << std::endl;
  // std::cout << "g: " << g_.transpose() << std::endl;
  // std::cout << "H: " << H_.transpose() << std::endl;
  // std::cout << "Gz: " << Gz_.transpose() << std::endl;
  // std::cout << "Gx: " << Gx_[0] << std::endl;
  // std::cout << "rhs: " << rhs_.transpose() << std::endl;
}

template<int DIM>
void MixedFriction<DIM>::solve(VectorXd& dx) {
  if (constraints_.empty()) {
    return;
  }
  
  OptimizerData::get().timer.start("solve", "MixedFriction");

  VectorXd q = mesh_->projection_matrix().transpose() * dx;
  Gdx_.resize(z_.size());

  #pragma omp parallel for
  for (size_t i = 0; i < constraints_.size(); ++i) {
    // Get frame configuration vector
    VectorXd qi(Gx_[i].size());
    for (int j = 0; j < 4; ++j) {
      if (T_(i,j) == -1) break;
      qi.segment<DIM>(DIM*j) = q.segment<DIM>(DIM*T_(i,j));
    }

    double Gz_inv_sqr = 1.0 / (Gz_(i) * Gz_(i));
    Gdx_(i) = Gz_inv_sqr * qi.dot(Gx_[i]);
  }
  dz_ = (Z_ - z_ + Gdx_);
  la_ = H_.array() * dz_.array() + g_.array();
  // std::cout << "dz: " << dz_.transpose() << std::endl;
  // std::cout << "la: " << la_.transpose() << std::endl;
  OptimizerData::get().timer.stop("solve", "MixedFriction");
}


template<int DIM>
VectorXd& MixedFriction<DIM>::rhs() {
  if (constraints_.empty()) {
    rhs_.resize(mesh_->jacobian().rows());
    rhs_.setZero();
  }
  return rhs_;
}

template<int DIM>
void MixedFriction<DIM>::reset() {
  grad_.resize(0);
}

template<int DIM>
void MixedFriction<DIM>::pre_solve() {
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

  size_t num_frames = constraints_.size();

  // Create element matrix for assemblers
  T_.resize(num_frames, 4);
  for (size_t i = 0; i < constraints_.size(); ++i) {
    std::array<long, 4> ids = constraints_[i].vertex_indices(E, F);
    for (int j = 0; j < 4; ++j) {
      T_(i,j) = -1;
      if (ids[j] != -1) {
        T_(i,j) = ipc_mesh.to_full_vertex_id(ids[j]);
      }
    }
  }

  // Initialize mixed variables
  z_.resize(num_frames);
  z_.setZero();
  Z_.resize(num_frames);
  g_.resize(num_frames);
  H_.resize(num_frames);
  Gx_.resize(num_frames);
  Gz_.resize(num_frames);
  Gz_.setConstant(-1);
  rhs_.resize(num_frames);

  // Remaking assemblers since collision frames change.
  assembler_ = std::make_shared<Assembler<double,DIM,-1>>(T_,
      mesh_->free_map_);
  vec_assembler_ = std::make_shared<VecAssembler<double,DIM,-1>>(T_,
      mesh_->free_map_);
  std::cout << "pre solve: " << num_frames << std::endl;
}

template<int DIM>
VectorXd MixedFriction<DIM>::gradient() {
  Eigen::VectorXd tmp;
  std::cout <<"gradient unimplemented for MixedFriction" << std::endl;
  return tmp;
}
template<int DIM>
VectorXd MixedFriction<DIM>::gradient_mixed() {
  Eigen::VectorXd tmp;
  std::cout <<"gradient_mixed unimplemented for MixedFriction" << std::endl;
  return tmp;
}
template<int DIM>
VectorXd MixedFriction<DIM>::gradient_dual() {
  Eigen::VectorXd tmp;
  std::cout <<"gradient_dual unimplemented for MixedFriction" << std::endl;
  return tmp;
}

template class mfem::MixedFriction<3>; // 3D
template class mfem::MixedFriction<2>; // 2D
