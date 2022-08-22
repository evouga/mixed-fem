#include "mixed_collision.h"
#include "mesh/mesh.h"
#include "igl/unique.h"
#include "igl/boundary_facets.h"
#include "simple_psd_fix.h"
#include "config.h"
#include <unsupported/Eigen/SparseExtra>
#include "igl/edges.h"
#include "igl/oriented_facets.h"
#include <ipc/barrier/barrier.hpp>

using namespace Eigen;
using namespace mfem;

template<int DIM>
double MixedCollision<DIM>::energy(const VectorXd& x, const VectorXd& d) {


  MatrixXd V = Map<const MatrixXd>(x.data(), mesh_->V_.cols(),
      mesh_->V_.rows());
  V.transposeInPlace();
  MatrixXd V_srf = ipc_mesh_.vertices(V);

  // MatrixXi F; 
  // if constexpr (DIM == 3) {
  //   F = F_;
  // }
  ipc::Constraints constraints;
  ipc::construct_constraint_set(ipc_mesh_, V_srf, config_->dhat, constraints);

  double h2 = dt_*dt_;
  double e = 0;
  // std::cout << "ENERGY \n V.rows: " << V.rows() << std::endl;
  // std::cout << "mesh.num_vertices(): " << ipc_mesh_.num_vertices() << std::endl;
  // std::cout << "vertices(): " << ipc_mesh_.vertices(V).rows() << std::endl;

  const Eigen::MatrixXi& E = ipc_mesh_.edges();
  const Eigen::MatrixXi& F = ipc_mesh_.faces();
    // std::cout << "E1: \n" << E << "\n E2: \n" << E_ << std::endl;

  #pragma omp parallel for reduction( + : e )
  for (size_t i = 0; i < constraints.size(); ++i) {
    std::array<long, 4> ids = constraints[i].vertex_indices(E, F);
    double D = constraints[i].compute_distance(V_srf, E, F);
    double la = 0;
    double d = D;
    // Find if this frame already exists
    if (auto it = frame_map_.find(ids); it != frame_map_.end()) {
      int idx = it->second;
      d = d_(idx);
    }
    double dhat_sqr = config_->dhat * config_->dhat;
    e += config_->kappa * ipc::barrier(d, dhat_sqr) / h2;
  }

  // std::cout << "energy new_d: " << e << std::endl;
  //std::cout << "x size & norm: " << x.size() << " norm: " << x.norm() << std::endl;
  return e;
}


template <int DIM>
double MixedCollision<DIM>::constraint_value(const VectorXd& x,
    const VectorXd& d) {

  double e = 0;

  MatrixXi F; 
  if constexpr (DIM == 3) {
    F = F_;
  }
  MatrixXd V = Map<const MatrixXd>(x.data(), mesh_->V_.cols(),
      mesh_->V_.rows());
  V.transposeInPlace();
  MatrixXd V_srf = ipc_mesh_.vertices(V);

  #pragma omp parallel for reduction( + : e )
  for (int i = 0; i < nframes_; ++i) {
    double D = constraints_[i].compute_distance(V_srf, E_, F);
    e += la_(i) * (D - d(i));
  }
  return e;
}

template<int DIM>
void MixedCollision<DIM>::update(const Eigen::VectorXd& x, double dt) {
  // Get collision frames
  dt_ = dt;

  MatrixXd V = Map<const MatrixXd>(x.data(), mesh_->V_.cols(),
      mesh_->V_.rows());
  V.transposeInPlace();
  
  MatrixXi F; 
  if constexpr (DIM == 3) {
    F = F_;
  } 
  std::cout << "V.rows: " << V.rows() << std::endl;
  std::cout << "mesh.num_vertices(): " << ipc_mesh_.num_vertices() << std::endl;
  std::cout << "vertices(): " << ipc_mesh_.vertices(V).rows() << std::endl;
  MatrixXd V_srf = ipc_mesh_.vertices(V);
  ipc::construct_constraint_set(ipc_mesh_, V_srf, config_->dhat, constraints_);

  std::vector<double> new_D;
  std::vector<double> new_d;
  std::vector<double> new_lambda;
  std::map<std::array<long, 4>, int> new_frame_map;
  dd_dx_.clear();

  nframes_ = constraints_.size();
  T_.resize(nframes_,4);
  for (size_t i = 0; i < constraints_.size(); ++i) {
    std::array<long, 4> ids = constraints_[i].vertex_indices(E_, F);
    double D = constraints_[i].compute_distance(V_srf, E_, F);
    double la = 0;
    double d = D;
    for (int j = 0; j < 4; ++j) {
      T_(i,j) = ids[j];
    }

    // Find if this frame already exists
    if (auto it = frame_map_.find(ids); it != frame_map_.end()) {
      int idx = it->second;
      la = la_(idx);
      d = d_(idx);
    }
    new_D.push_back(D);
    new_d.push_back(d);
    new_lambda.push_back(la);
    dd_dx_.push_back(constraints_[i].compute_distance_gradient(V_srf,E_,F));
    new_frame_map[ids] = i;
  }
  D_ = Map<VectorXd>(new_D.data(), new_D.size());
  d_ = Map<VectorXd>(new_d.data(), new_d.size());
  la_ = Map<VectorXd>(new_lambda.data(), new_lambda.size());
  std::swap(frame_map_, new_frame_map);
  data_.timer.start("Update Coll frames");
  // update_collision_frames(x);
  data_.timer.stop("Update Coll frames");

  std::cout << "la_: " << la_.norm() << std::endl;
  // std::cout << "d: " << d_.transpose() << std::endl;
  // std::cout << "D: " << D_.transpose() << std::endl;
  // std::cout << "num constraints: "<< constraints_.num_constraints() << std::endl;
  // std::cout << "nframes_ : " << nframes_ << std::endl;

  // Structure potentially changes each step, so just rebuild assembler :/
  // NOTE assuming each local jacobian has same size!
  data_.timer.start("Create assemblers");
  assembler_ = std::make_shared<Assembler<double,DIM,-1>>(T_, mesh_->free_map_);
  vec_assembler_ = std::make_shared<VecAssembler<double,DIM,-1>>(T_,
      mesh_->free_map_);
  data_.timer.stop("Create assemblers");
  update_derivatives(V_srf, dt);
}

template<int DIM>
void MixedCollision<DIM>::update_collision_frames(const Eigen::VectorXd& x) {
}

template<int DIM>
void MixedCollision<DIM>::update_derivatives(const MatrixXd& V, double dt) {

  if (nframes_ == 0) {
    return;
  }

  data_.timer.start("Hinv");
  H_.resize(nframes_);
  g_.resize(nframes_);
  MatrixXi F; 
  if constexpr (DIM == 3) {
    F = F_;
  } 

  double dhat_sqr = config_->dhat * config_->dhat;

  #pragma omp parallel for
  for (int i = 0; i < nframes_; ++i) {
    H_[i] = config_->kappa * ipc::barrier_hessian(d_(i), dhat_sqr);
    g_[i] = config_->kappa * ipc::barrier_gradient(d_(i), dhat_sqr);
  }
  data_.timer.stop("Hinv");
  
  data_.timer.start("Local H");
  Aloc_.resize(nframes_);
  #pragma omp parallel for
  for (int i = 0; i < nframes_; ++i) {
    const ipc::MatrixMax12d distance_hess = 
        constraints_[i].compute_distance_hessian(V, E_, F);

    Aloc_[i] = dd_dx_[i] * H_(i) * dd_dx_[i].transpose() 
        - la_(i) * distance_hess; // TODO + sign??????
    sim::simple_psd_fix(Aloc_[i]);

  }
  data_.timer.stop("Local H");

  data_.timer.start("Update LHS");
  assembler_->update_matrix(Aloc_);
  data_.timer.stop("Update LHS");

  //std::cout << "update_derivs: \n" << d_ << std::endl;
  //saveMarket(assembler_->A, "lhs_c1.mkt");

  A_ = assembler_->A;

  data_.timer.start("Update RHS");
  // Gradient with respect to x variable
  std::vector<VectorXd> g(nframes_);
  for (int i = 0; i < nframes_; ++i) {
    g[i] = -dd_dx_[i] * g_(i);
  }
  vec_assembler_->assemble(g, grad_x_);

  // Gradient with respect to mixed variable
  grad_ = g_ + la_;
  data_.timer.stop("Update RHS");
}

template<int DIM>
VectorXd MixedCollision<DIM>::rhs() {
  data_.timer.start("RHS - s");

  assert(D_.size() == d_.size());

  rhs_.resize(mesh_->jacobian().rows());
  rhs_.setZero();
  gl_.resize(nframes_);

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
  std::vector<VectorXd> g(nframes_);
  Gdx_.resize(d_.size());

  //std::cout << "T: \n" << T_ << std::endl;
  #pragma omp parallel for
  for (int i = 0; i < nframes_; ++i) {
    //Matrix<double,DIM*3,1> qi;
    //const Vector3i& E = collision_frames2_[i].E_;
    //qi << q.segment<DIM>(DIM*E(0)),
    //      q.segment<DIM>(DIM*E(1)),
    //      q.segment<DIM>(DIM*E(2));
    VectorXd qi(dd_dx_[i].size());
    for (int j = 0; j < 4; ++j) {
      if (T_(i,j) == -1) break;
      qi.segment<DIM>(DIM*j) = q.segment<DIM>(DIM*T_(i,j));
    }
    Gdx_(i) = -qi.dot(dd_dx_[i]);
  }
  la_ = -gl_.array() + (H_.array() * Gdx_.array());
  delta_ = -(la_ + g_).array() / H_.array();
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


  if constexpr (DIM ==2) {
    // TODO use "include_vertex"
    // TODO use the boundary facets
    // igl::boundary_facets(mesh_->T_, E_);
    igl::oriented_facets(mesh_->T_, E_);
    // igl::edges(mesh_->T_, E_);
    ipc_mesh_ = ipc::CollisionMesh::build_from_full_mesh(mesh_->V_, E_, F_);

    const Eigen::MatrixXi& E = ipc_mesh_.edges();
    const Eigen::MatrixXi& F = ipc_mesh_.faces();
    E_ = E;
    F_ = F;
  } else {
    igl::boundary_facets(mesh_->T_, F_);
    // igl::edges(mesh_->T_, E_);
    igl::edges(F_, E_);


    ipc_mesh_ = ipc::CollisionMesh::build_from_full_mesh(mesh_->V_, E_, F_);
    const Eigen::MatrixXi& E = ipc_mesh_.edges();
    const Eigen::MatrixXi& F = ipc_mesh_.faces();
    E_ = E;
    F_ = F;
  } 
}

template<int DIM>
void MixedCollision<DIM>::post_solve() {
  la_.setZero();
  dd_dx_.clear();
  frame_map_.clear();
}

template class mfem::MixedCollision<3>; // 3D
template class mfem::MixedCollision<2>; // 2D