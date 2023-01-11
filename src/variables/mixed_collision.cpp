#include "mixed_collision.h"
#include "mesh/mesh.h"
#include "simple_psd_fix.h"
#include "config.h"
#include <unsupported/Eigen/SparseExtra>
#include <ipc/barrier/barrier.hpp>

using namespace Eigen;
using namespace mfem;

template<int DIM>
double MixedCollision<DIM>::energy(const VectorXd& x, const VectorXd& d) {

  // Convert configuration vector to matrix form
  MatrixXd V = Map<const MatrixXd>(x.data(), DIM, mesh_->V_.rows());
  V.transposeInPlace();
  
  // Get IPC mesh and face/edge/vertex data
  const auto& ipc_mesh = mesh_->collision_mesh();
  MatrixXd V_srf = ipc_mesh.vertices(V); // surface vertex set

  // Computing collision constraints
  ipc::MixedConstraints constraints = constraints_;
  constraints.update_distances(d);
  ipc::Candidates candidates;
  ipc::construct_collision_candidates(
      ipc_mesh, V_srf, candidates, config_->dhat * 1.1, ipc::BroadPhaseMethod::SPATIAL_HASH);
  ipc::construct_constraint_set(candidates, ipc_mesh, V_srf, config_->dhat,
      constraints);

  double h2 = dt_*dt_;
  double e = 0;

  #pragma omp parallel for reduction( + : e )
  for (size_t i = 0; i < constraints.size(); ++i) {
    double di = constraints.distance(i);
    //e += config_->kappa * ipc::barrier(di, config_->dhat) / h2;
    e += config_->kappa * ipc::barrier(di*di, config_->dhat*config_->dhat) / h2;
  }
  return e;
}

template <int DIM>
double MixedCollision<DIM>::constraint_value(const VectorXd& x,
    const VectorXd& d) {

  double e = 0;

  // Convert configuration vector to matrix form
  MatrixXd V = Map<const MatrixXd>(x.data(), DIM, mesh_->V_.rows());
  V.transposeInPlace();

  // Getting IPC mesh data
  const auto& ipc_mesh = mesh_->collision_mesh();
  const Eigen::MatrixXi& E = ipc_mesh.edges();
  const Eigen::MatrixXi& F = ipc_mesh.faces();

  // Convert to reduced (surface only) vertex set
  MatrixXd V_srf = ipc_mesh.vertices(V);
  double dhat = config_->dhat;

  // Only need to evaluate constraint value for existing frames.
  // New frames are initialized such that the mixed distance
  // equals the nodal distance, so the lagrange multipliers are zero.
  #pragma omp parallel for reduction( + : e )
  for (size_t i = 0; i < constraints_.size(); ++i) {
    // Make sure constraint is still valid
    double D = constraints_[i].compute_distance(V_srf, E, F,
        ipc::DistanceMode::SQRT);
    if (D <= dhat || d(i) <= dhat) {
      e += la_(i) * (D - d(i));
    }
  }
  return e;
}

template<int DIM>
void MixedCollision<DIM>::update(const Eigen::VectorXd& x, double dt) {
  // Get collision frames
  dt_ = dt;

  // Convert configuration vector to matrix form
  MatrixXd V = Map<const MatrixXd>(x.data(), DIM, mesh_->V_.rows());
  V.transposeInPlace();

  // Getting IPC mesh data
  const auto& ipc_mesh = mesh_->collision_mesh();
  const Eigen::MatrixXi& E = ipc_mesh.edges();
  const Eigen::MatrixXi& F = ipc_mesh.faces();

  // Convert to reduced (surface only) vertex set
  MatrixXd V_srf = ipc_mesh.vertices(V);

  // Computing collision constraints
  // ipc::construct_constraint_set(ipc_mesh, V_srf, config_->dhat, constraints_)
  constraints_.update_distances(d_);
  constraints_.update_lambdas(la_);
  ipc::Candidates candidates;
  ipc::construct_collision_candidates(
      ipc_mesh, V_srf, candidates, config_->dhat * 1.1);
  ipc::construct_constraint_set(candidates, ipc_mesh, V_srf,
      config_->dhat, constraints_);

  std::map<std::array<long, 4>, int> new_frame_map;

  int num_frames = constraints_.size();
  T_.resize(num_frames, 4);
  D_.resize(num_frames);
  dd_dx_.resize(num_frames);
  VectorXd d_new(num_frames);
  VectorXd la_new(num_frames);

  // Rebuilding mixed variables for the new set of collision frames.
  for (int i = 0; i < num_frames; ++i) {
    // Getting collision frame, and computing squared distance.
    std::array<long, 4> ids = constraints_[i].vertex_indices(E, F);

    // Add entry in element matrix. Used for assembly.
    for (int j = 0; j < 4; ++j) {
      T_(i,j) = -1;
      if (ids[j] != -1) {
        T_(i,j) = ipc_mesh.to_full_vertex_id(ids[j]);
      }
    }
  // TODO compute_distance_gradient
  // SHOULD be doing dtype
    D_(i) = constraints_[i].compute_distance(V_srf, E, F,
        ipc::DistanceMode::SQRT);
    dd_dx_[i] = constraints_[i].compute_distance_gradient(V_srf, E, F,
        ipc::DistanceMode::SQRT);
  }
  d_ = constraints_.get_distances();
  la_ = constraints_.get_lambdas();

  if (num_frames > 0) {
    double ratio = d_.minCoeff() / D_.minCoeff();
    if (ratio > 1e3) {
      std::cout << "WARNING: distance discrepancy is high" << std::endl;
      std::cout << "la max: " << la_.maxCoeff() 
                << " min: " << la_.minCoeff() << std::endl;
      std::cout << "d min: " << d_.minCoeff() << std::endl;
      std::cout << "D min: " << D_.minCoeff() << std::endl;
      std::cout << "Ratio: " << d_.minCoeff() / D_.minCoeff() << std::endl;
    }
  }

  // std::cout << "num constraints: "<< constraints_.num_constraints() << std::endl;
  // std::cout << "nframes_ : " << nframes_ << std::endl;

  // Create new assembler, since collision frame set changes.
  // TODO - use the IPC assemlber instead. it's better
  data_.timer.start("Create assemblers");
  assembler_ = std::make_shared<Assembler<double,DIM,-1>>(T_, mesh_->free_map_);
  vec_assembler_ = std::make_shared<VecAssembler<double,DIM,-1>>(T_,
      mesh_->free_map_);
  data_.timer.stop("Create assemblers");

  // Build gradients and hessian
  update_derivatives(V_srf, dt);
}

template<int DIM>
void MixedCollision<DIM>::update_derivatives(const MatrixXd& V, double dt) {

  if (constraints_.empty()) {
    return;
  }

  data_.timer.start("g-H");
  H_.resize(constraints_.size());
  g_.resize(constraints_.size());

  // vector of local hessians and gradients
  std::vector<Eigen::MatrixXd> Aloc(constraints_.size());
  std::vector<Eigen::VectorXd> gloc(constraints_.size()); 

  // Getting IPC mesh data
  const auto& ipc_mesh = mesh_->collision_mesh();
  const Eigen::MatrixXi& E = ipc_mesh.edges();
  const Eigen::MatrixXi& F = ipc_mesh.faces();

  double dhat = config_->dhat;
  double dhat_sqr = config_->dhat * config_->dhat;
  // double h2 = dt_*dt_;
 
  #pragma omp parallel for
  for (size_t i = 0; i < constraints_.size(); ++i) {
    // Mixed barrier energy gradients and hessians
    //g_[i] = config_->kappa * ipc::barrier_gradient(d_(i), dhat);
    //H_[i] = config_->kappa * ipc::barrier_hessian(d_(i), dhat);

    double d2 = d_(i) * d_(i);
    double g = ipc::barrier_gradient(d2, dhat_sqr);
    // dphi(d^2)/d = 2d g(d^2)
    // dphi2(d^2)/d^2 = 2g(d^2) + 4d^2 H(d^2)
    g_[i] = config_->kappa * (g * d_(i) * 2);
    H_[i] = config_->kappa * (ipc::barrier_hessian(d2, dhat_sqr)*4*d2 + 2*g);
    H_(i) = std::max(H_(i), 1e-8);
    Aloc[i] = dd_dx_[i] * H_(i) * dd_dx_[i].transpose();

    // Gradient with respect to x variable
    gloc[i] = dd_dx_[i] * la_(i);
  }
  data_.timer.stop("g-H");

  data_.timer.start("Update LHS");
  assembler_->update_matrix(Aloc);
  A_ = assembler_->A;
  //saveMarket(assembler_->A, "lhs_c1.mkt");
  data_.timer.stop("Update LHS");

  // Assemble gradient with respect to x variable
  data_.timer.start("Update RHS");
  vec_assembler_->assemble(gloc, grad_x_);
  data_.timer.stop("Update RHS");

  // Gradient with respect to mixed variable
  grad_ = g_ + la_;
}

template<int DIM>
VectorXd MixedCollision<DIM>::rhs() {
  data_.timer.start("RHS - s");

  assert(D_.size() == d_.size());

  rhs_.resize(mesh_->jacobian().rows());
  rhs_.setZero();
  gl_.resize(constraints_.size());

  // Computing schur-complement system right-hand-side
  std::vector<VectorXd> g(constraints_.size());
  #pragma omp parallel for
  for (size_t i = 0; i < constraints_.size(); ++i) {
    gl_(i) = H_(i) * (D_(i) - d_(i)) + g_(i);
    g[i] = -dd_dx_[i] * gl_(i);
  }
  vec_assembler_->assemble(g, rhs_);
  data_.timer.stop("RHS - s");
  return rhs_;
}

template<int DIM>
VectorXd MixedCollision<DIM>::gradient() {
  if (constraints_.empty()) {
    grad_x_.resize(mesh_->jacobian().rows());
    grad_x_.setZero();
  }
  return grad_x_;
}

template<int DIM>
VectorXd MixedCollision<DIM>::gradient_mixed() {
  if (constraints_.empty()) {
    grad_.resize(0);
  }
  return grad_;
}

template<int DIM>
void MixedCollision<DIM>::solve(const VectorXd& dx) {
  if (constraints_.empty()) {
    return;
  }
  
  VectorXd q = mesh_->projection_matrix().transpose() * dx;

  data_.timer.start("local");
  Gdx_.resize(d_.size());

  #pragma omp parallel for
  for (size_t i = 0; i < constraints_.size(); ++i) {
    // Get frame configuration vector
    VectorXd qi(dd_dx_[i].size());
    for (int j = 0; j < 4; ++j) {
      if (T_(i,j) == -1) break;
      qi.segment<DIM>(DIM*j) = q.segment<DIM>(DIM*T_(i,j));
    }
    Gdx_(i) = qi.dot(dd_dx_[i]);
  }
  // Update lagrange multipliers
  la_ = gl_.array() + (H_.array() * Gdx_.array());

  // Compute mixed variable descent direction
  delta_ = -(g_ - la_).array() / H_.array();
  if (delta_.hasNaN()) {
    std::cout << "la_: " << la_ << std::endl;
    std::cout << "g_: " << g_ << std::endl;
    std::cout << "H_: " << H_ << std::endl;
    exit(1);
  }
  data_.timer.stop("local");
}

template<int DIM>
void MixedCollision<DIM>::reset() {
  d_.resize(0);
  g_.resize(0);
  H_.resize(0);
  la_.resize(0);
  gl_.resize(0);
  rhs_.resize(0);
  grad_.resize(0);
  delta_.resize(0);
  dd_dx_.resize(0);
  grad_x_.resize(0);
  frame_map_.clear();
}

template<int DIM>
void MixedCollision<DIM>::post_solve() {
  d_.resize(0);
  la_.resize(0);
  la_.setZero();
  dd_dx_.clear();
  frame_map_.clear();
  constraints_.clear();
}

template class mfem::MixedCollision<3>; // 3D
template class mfem::MixedCollision<2>; // 2D
