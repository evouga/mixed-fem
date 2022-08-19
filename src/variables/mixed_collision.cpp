#include "mixed_collision.h"
#include "mesh/mesh.h"
#include "igl/unique.h"
#include "igl/boundary_facets.h"
#include "simple_psd_fix.h"
#include "config.h"
#include <unsupported/Eigen/SparseExtra>
#include "igl/edges.h"

using namespace Eigen;
using namespace mfem;

namespace {

  //TODO psi(s) can still be negative!
  // Log barrier energy
  double psi(double d, double h, double k) {
    if (d <= 0)
      return std::numeric_limits<double>::max();
    else if (d >= h)
      return 0;
    else
      return -k*log(d/h)*pow(d-h,2.0);
    //return k*pow(d-h,2);
  }

  double dpsi(double d, double h, double k) {
    return -(k*pow(d-h,2.0))/d-k*log(d/h)*(d*2.0-h*2.0);
    //return k*2*(d-h);
  }

  double d2psi(double d, double h, double k) {
    return k*log(d/h)*-2.0-(k*(d*2.0-h*2.0)*2.0)/d+1.0/(d*d)*k*pow(d-h,2.0);
    //return 2*k;
  }
}

template<int DIM>
double MixedCollision<DIM>::energy(const VectorXd& x, const VectorXd& d) {

  std::set<std::unique_ptr<CollisionFrame<2>>,FrameLess<2>> frames;
  std::vector<double> new_d;
  // For each boundary vertex find primitives within distance threshold
  for (int i = 0; i < C_.size(); ++i) {

    // Currently brute force check all primitives
    for (int j = 0; j < F_.rows(); ++j) {
      if (C_(i) == F_(j,0) || C_(i) == F_(j,1)) {
        continue;
      }

      auto frame = CollisionFrame<2>::make_collision_frame<
          Vector3i, POINT_EDGE>(x, Vector3i(F_(j,0), F_(j,1), C_(i)));

      // Avoid duplicate entries
      if (frame != nullptr && frames.find(frame) == frames.end()) {
        double D = frame->distance(x);
        double d = D; 
        if (auto it = frames_.find(frame); it != frames_.end()) {
          int idx = it->second;
          d = d_(idx);
        }
        // If valid and within distance thresholds add new frame
        if (D >= 0 && D < config_->dhat) {
          new_d.push_back(d);
          frames.insert(std::move(frame));
        }
      }
    }
  }

  //std::cout << "energy new_d: " << Map<VectorXd>(new_d.data(), new_d.size())
  //    << "\n d_ : " << d_ << std::endl;
  //std::cout << "x size & norm: " << x.size() << " norm: " << x.norm() << std::endl;
  double h2 = dt_*dt_;
  double e = 0;
  #pragma omp parallel for reduction( + : e )
  for (size_t i = 0; i < new_d.size(); ++i) {
    e += psi(new_d[i], config_->dhat, config_->kappa) / h2;
  }
  return e;
}


template <int DIM>
double MixedCollision<DIM>::constraint_value(const VectorXd& x,
    const VectorXd& d) {

  double e = 0;

  for (const auto& it : frames_) {
    int i = it.second;
    e += la_(i) * (it.first->distance(x) - d(i));
  }
  //#pragma omp parallel for reduction( + : e )
  //for (int i = 0; i < nframes_; ++i) {
  //  e += la_(i) * (collision_frames2_[i].distance(x) - d(i));
  //}
  return e;
}

template<int DIM>
void MixedCollision<DIM>::update(const Eigen::VectorXd& x, double dt) {
  // Get collision frames
  dt_ = dt;

  Eigen::MatrixXi E;
  igl::edges(mesh_->T_, E);
  MatrixXd V = Map<const MatrixXd>(x.data(), mesh_->V_.cols(),
      mesh_->V_.rows());
  V.transposeInPlace();
  
  // assert(E.size() == 0 || E.cols() == 2);
  // assert(F.size() == 0 || F.cols() == 3);
  MatrixXi tmp;
  ipc::CollisionMesh ipc_mesh;
  
  if constexpr (DIM ==2) {
    // TODO use "include_vertex"
    // TODO use the boundary facets
    ipc_mesh = ipc::CollisionMesh::build_from_full_mesh(V, E, tmp);
  } else {
    std::cerr << "SHIT BRUH WE NEED BOUNDARY EDGES TOO" << std::endl;
  } 
  // ipc::Constraints constraint_set_;
  ipc::construct_constraint_set(ipc_mesh, V, config_->dhat, constraint_set_);

  std::vector<double> new_D;
  std::vector<double> new_d;
  std::vector<double> new_lambda;
  std::map<std::array<long, 4>, int> new_frame_map;
  dd_dx_.clear();
  for (size_t i = 0; i < constraint_set_.size(); ++i) {
    std::array<long, 4> ids = constraint_set_[i].vertex_indices(E, tmp);
    double D = constraint_set_[i].compute_distance(V, E, tmp);
    double la = 0;
    double d = D;
    if (auto it = frame_map_.find(ids); it != frame_map_.end()) {
      int idx = it->second;
      la = la_(idx);
      d = d_(idx);
    }
    // If valid and within distance thresholds add new frame
    if (D > 0 && D < config_->dhat*config_->dhat) {
      new_D.push_back(D);
      new_d.push_back(d);
      new_lambda.push_back(la);
      dd_dx_.push_back(constraint_set_[i].compute_distance_gradient(V,E,tmp));
      new_frame_map[ids] = new_D.size()-1;
    }
  }
  D_ = Map<VectorXd>(new_D.data(), new_D.size());
  // d_ = Map<VectorXd>(new_d.data(), new_d.size());
  la_ = Map<VectorXd>(new_lambda.data(), new_lambda.size());
  std::swap(frame_map_, new_frame_map);
    // std::cout << "d: " << d_.transpose() << std::endl;
  std::cout << "D: " << D_.transpose().array().sqrt() << std::endl;
  data_.timer.start("Update Coll frames");
  update_collision_frames(x);
  data_.timer.stop("Update Coll frames");

  nframes_ = frames_.size();
  T_.resize(nframes_,3);
  T_.setConstant(-1);
  for (const auto& it : frames_) {
    const VectorXi& E = it.first->E();
    int idx = it.second;
    for (int j = 0; j < E.size(); ++j) {
      T_(idx,j) = E(j);
    }
  }
    std::cout << "d: " << d_.transpose() << std::endl;
  std::cout << "D: " << D_.transpose() << std::endl;
  std::cout << "num constraints: "<< constraint_set_.num_constraints() << std::endl;
  std::cout << "nframes_ : " << nframes_ << std::endl;

  // Structure potentially changes each step, so just rebuild assembler :/
  // NOTE assuming each local jacobian has same size!
  data_.timer.start("Create assemblers");
  assembler_ = std::make_shared<Assembler<double,DIM,-1>>(T_, mesh_->free_map_);
  vec_assembler_ = std::make_shared<VecAssembler<double,DIM,-1>>(T_,
      mesh_->free_map_);
  data_.timer.stop("Create assemblers");
  update_derivatives(dt);
}

template<int DIM>
void MixedCollision<DIM>::update_collision_frames(const Eigen::VectorXd& x) {
  // Detect Collision Frames
  // Initialize distance variables 
  std::vector<double> new_D;
  std::vector<double> new_d;
  std::vector<double> new_lambda;
  std::vector<CollisionFrame2> new_frames;
  std::map<std::tuple<int,int,int>, int> new_ids;
  dd_dx_.clear();

  std::map<std::unique_ptr<CollisionFrame<2>>,int,FrameLess<2>> frames;

  for (int i = 0; i < C_.size(); ++i) {
    // Currently brute force check all primitives
    for (int j = 0; j < F_.rows(); ++j) {
      if (C_(i) == F_(j,0) || C_(i) == F_(j,1)) {
        continue;
      }

      // Build a frame and compute distance for the primitive - point pair
      // NOTE won't work for DIM=3 right now
      auto frame = CollisionFrame<2>::make_collision_frame<
          Vector3i, POINT_EDGE>(x, Vector3i(F_(j,0), F_(j,1), C_(i)));

      // Avoid duplicate entries
      if (frame != nullptr && frames.find(frame) == frames.end()) {
        double D = frame->distance(x);
        double la = 0;
        double d = D; 
        if (auto it = frames_.find(frame); it != frames_.end()) {
          int idx = it->second;
          la = la_(idx);
          d = d_(idx);
        }
        // If valid and within distance thresholds add new frame
        if (D > 0 && D < config_->dhat) {

          new_D.push_back(D);
          new_d.push_back(d);
          new_lambda.push_back(la);
          dd_dx_.push_back(frame->gradient(x));
          frames.insert(std::make_pair(std::move(frame), new_D.size()-1));
        }
      }
    }
  }
  D_ = Map<VectorXd>(new_D.data(), new_D.size());
  d_ = Map<VectorXd>(new_d.data(), new_d.size());
  la_ = Map<VectorXd>(new_lambda.data(), new_lambda.size());
  std::swap(frames, frames_);
  std::swap(new_ids, frame_ids_);
  std::swap(new_frames, collision_frames2_);
}

template<int DIM>
void MixedCollision<DIM>::update_derivatives(double dt) {

  if (nframes_ == 0) {
    return;
  }

  data_.timer.start("Hinv");
  H_.resize(nframes_);
  g_.resize(nframes_);

  #pragma omp parallel for
  for (int i = 0; i < nframes_; ++i) {
    H_[i] = d2psi(d_(i), config_->dhat, config_->kappa);
    g_[i] = dpsi(d_(i), config_->dhat, config_->kappa);
  }
  data_.timer.stop("Hinv");
  
  data_.timer.start("Local H");
  Aloc_.resize(nframes_);
  #pragma omp parallel for
  for (int i = 0; i < nframes_; ++i) {
    Aloc_[i] = dd_dx_[i] * H_(i) * dd_dx_[i].transpose();
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
    for (int j = 0; j < 3; ++j) {
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
  collision_frames2_.clear();
  frames_.clear();

  igl::boundary_facets(mesh_->T_, F_);
  assert(F_.cols() == 2); // Only supports 2D right now
  igl::unique(F_,C_); 
}

template<int DIM>
void MixedCollision<DIM>::post_solve() {
  la_.setZero();
  dd_dx_.clear();
  collision_frames2_.clear();
  frame_ids_.clear();
  frames_.clear();
}

template class mfem::MixedCollision<3>; // 3D
template class mfem::MixedCollision<2>; // 2D
