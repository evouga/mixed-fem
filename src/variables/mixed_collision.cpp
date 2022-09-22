#include "mixed_collision.h"
#include "mesh/mesh.h"
#include "simple_psd_fix.h"
#include "config.h"
#include <unsupported/Eigen/SparseExtra>
#include <ipc/barrier/barrier.hpp>
#include "utils/mixed_ipc.h"

using namespace Eigen;
using namespace mfem;

template<int DIM>
double MixedCollision<DIM>::energy(const VectorXd& x, const VectorXd& d) {

  // Convert configuration vector to matrix form
  MatrixXd V = Map<const MatrixXd>(x.data(), DIM, mesh_->V_.rows());
  V.transposeInPlace();
  
  // Get IPC mesh and face/edge/vertex data
  const auto& ipc_mesh = mesh_->collision_mesh();
  const Eigen::MatrixXi& E = ipc_mesh.edges();
  const Eigen::MatrixXi& F = ipc_mesh.faces();
  MatrixXd V_srf = ipc_mesh.vertices(V); // surface vertex set

  // Computing collision constraints
  ipc::Constraints constraints;
  ipc::Candidates candidates;
  ipc::construct_collision_candidates(
      ipc_mesh, V_srf, candidates, config_->dhat * 1.1);
  ipc::construct_constraint_set(candidates, ipc_mesh, V_srf, config_->dhat,
      constraints);


  double dhat = config_->dhat;
  double h2 = dt_*dt_;
  double e = 0;

  #pragma omp parallel for reduction( + : e )
  for (size_t i = 0; i < constraints.size(); ++i) {
    double di = 0;
    // Find if this frame already exists, and use the mixed distance
    // variable if found.
    std::array<long, 4> ids = constraints[i].vertex_indices(E, F);
    if (auto it = frame_map_.find(ids); it != frame_map_.end()) {
      int idx = it->second;
      di = d(idx);
    } else {
      // If mixed variable for this frame does not exist,
      // compute the distance based on nodal positions.
      di = constraints[i].compute_distance(V_srf, E, F,
          ipc::DistanceMode::SQRT);
    }
    e += config_->kappa * ipc::barrier(di, dhat) / h2;
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
  for (int i = 0; i < nframes_; ++i) {
    // Make sure constraint is still valid
    double D = constraints_[i].compute_distance(V_srf, E, F,
        ipc::DistanceMode::SQRT);
    if (D <= dhat) {
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
  ipc::Candidates candidates;
  ipc::construct_collision_candidates(
      ipc_mesh, V_srf, candidates, config_->dhat * 1.1);
  ipc::construct_constraint_set(candidates, ipc_mesh, V_srf,
      config_->dhat, constraints_);

  std::map<std::array<long, 4>, int> new_frame_map;
  nframes_ = constraints_.size();

  T_.resize(nframes_, 4);
  D_.resize(nframes_);
  dd_dx_.resize(nframes_);
  VectorXd d_new(nframes_);
  VectorXd la_new(nframes_);

  // Mixed Constraints - each constraint has local distance mixed variable
  //  - same compute potential, potential gradient interface but plugs in mixed variable
  // When building constraint set first construct all mixed constraints in typical way
  //  - maintain ev_candidates -> constraint map w



  // Rebuilding mixed variables for the new set of collision frames.
  for (int i = 0; i < nframes_; ++i) {
    // Getting collision frame, and computing squared distance.
    std::array<long, 4> ids = constraints_[i].vertex_indices(E, F);
    double D = constraints_[i].compute_distance(V_srf, E, F,
        ipc::DistanceMode::SQRT);
    double la = 0;
    double d = D;

    // Add entry in element matrix. Used for assembly.
    for (int j = 0; j < 4; ++j) {
      T_(i,j) = ids[j];
    }

    // Find if this frame already exists. If so, set the mixed variable
    // and lagrange multiplier value to their values from the previous
    // step.
    if (auto it = frame_map_.find(ids); it != frame_map_.end()) {
      int idx = it->second;
      la = la_(idx);
      d = d_(idx);
    }
    D_(i) = D;
    d_new(i) = d;
    la_new(i) = la;
    dd_dx_[i] = constraints_[i].compute_distance_gradient(V_srf, E, F,
        ipc::DistanceMode::SQRT);
    new_frame_map[ids] = i;
  }
  d_ = d_new;
  la_ = la_new;
  std::swap(new_frame_map, frame_map_);

  // std::cout << "T\n: " << T_ << std::endl;
  if (nframes_ > 0) {
    // std::cout << "la max: " << la_.maxCoeff() 
    //           << " min: " << la_.minCoeff() << std::endl;
    // std::cout << "d min: " << d_.minCoeff() << std::endl;
    // std::cout << "D min: " << D_.minCoeff() << std::endl;
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

  if (nframes_ == 0) {
    return;
  }

  data_.timer.start("g-H");
  H_.resize(nframes_);
  g_.resize(nframes_);

  // vector of local hessians and gradients
  std::vector<Eigen::MatrixXd> Aloc(nframes_);
  std::vector<Eigen::VectorXd> gloc(nframes_); 

  // Getting IPC mesh data
  const auto& ipc_mesh = mesh_->collision_mesh();
  const Eigen::MatrixXi& E = ipc_mesh.edges();
  const Eigen::MatrixXi& F = ipc_mesh.faces();

  double dhat = config_->dhat;// * config_->dhat;
  // double h2 = dt_*dt_;

  #pragma omp parallel for
  for (int i = 0; i < nframes_; ++i) {
    // Mixed barrier energy gradients and hessians
    g_[i] = config_->kappa * ipc::barrier_gradient(d_(i), dhat);
    H_[i] = config_->kappa * ipc::barrier_hessian(d_(i), dhat);
    H_(i) = std::max(H_(i), 1e-8);
    Aloc[i] = dd_dx_[i] * H_(i) * dd_dx_[i].transpose();
    // sim::simple_psd_fix(Aloc[i]);

    // Gradient with respect to x variable
    gloc[i] = -dd_dx_[i] * la_(i);
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
  gl_.resize(nframes_);

  // Computing schur-complement system right-hand-side
  std::vector<VectorXd> g(nframes_);
  #pragma omp parallel for
  for (int i = 0; i < nframes_; ++i) {
    gl_(i) = H_(i) * (D_(i) - d_(i)) + g_(i);
    g[i] = -dd_dx_[i] * gl_(i);
  }
  vec_assembler_->assemble(g, rhs_);
  data_.timer.stop("RHS - s");
  return rhs_;
}

template<int DIM>
VectorXd MixedCollision<DIM>::gradient() {
  if (nframes_ == 0) {
    grad_x_.resize(mesh_->jacobian().rows());
    grad_x_.setZero();
  }
  return grad_x_;
}

template<int DIM>
VectorXd MixedCollision<DIM>::gradient_mixed() {
  if (nframes_ == 0) {
    grad_.resize(0);
  }
  return grad_;
}

template<int DIM>
void MixedCollision<DIM>::solve(const VectorXd& dx) {
  if (nframes_ == 0) {
    return;
  }
  
  VectorXd q = mesh_->projection_matrix().transpose() * dx;

  data_.timer.start("local");
  Gdx_.resize(d_.size());

  #pragma omp parallel for
  for (int i = 0; i < nframes_; ++i) {
    // Get frame configuration vector
    VectorXd qi(dd_dx_[i].size());
    for (int j = 0; j < 4; ++j) {
      if (T_(i,j) == -1) break;
      qi.segment<DIM>(DIM*j) = q.segment<DIM>(DIM*T_(i,j));
    }
    Gdx_(i) = -qi.dot(dd_dx_[i]);
  }
  // Update lagrange multipliers
  la_ = -gl_.array() + (H_.array() * Gdx_.array());

  // Compute mixed variable descent direction
  delta_ = -(la_ + g_).array() / H_.array();
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
  la_.setZero();
  dd_dx_.clear();
  frame_map_.clear();
}

template class mfem::MixedCollision<3>; // 3D
template class mfem::MixedCollision<2>; // 2D
