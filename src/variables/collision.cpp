#include "mixed_collision.h" // TODO REMOVE ONLY USIN FOR COLLISION FRAME SHIT
#include "collision.h"
#include "mesh/mesh.h"
#include "igl/unique.h"
#include "igl/boundary_facets.h"
#include "simple_psd_fix.h"
#include "config.h"

using namespace Eigen;
using namespace mfem;

namespace {

  // TODO move this shit to the thing
  double psi(double d, double h, double k) {
    if (d <= 0)
      return std::numeric_limits<double>::max();
    else if (d >= h)
      return 0;
    else
      return -k*log(d/h)*pow(d-h,2.0);
  }

  double dpsi(double d, double h, double k) {
    return -(k*pow(d-h,2.0))/d-k*log(d/h)*(d*2.0-h*2.0);
  }

  double d2psi(double d, double h, double k) {
    return k*log(d/h)*-2.0-(k*(d*2.0-h*2.0)*2.0)/d+1.0/(d*d)*k*pow(d-h,2.0);
  }
}

template<int DIM>
double Collision<DIM>::energy(const VectorXd& x) {
  double e = 0;

  // Build collision Frames
  std::vector<double> new_D;
  // For each boundary vertex find primitives within distance threshold
  for (int i = 0; i < C_.size(); ++i) {

    // Currently brute force check all primitives
    for (int j = 0; j < F_.rows(); ++j) {
      if (C_(i) == F_(j,0) || C_(i) == F_(j,1)) {
        continue;
      }

      // Build a frame and compute distance for the primitive - point pair
      CollisionFrame frame(F_(j,0), F_(j,1), C_(i));
      double D = frame.distance(x);

      // If valid and within distance thresholds add new frame
      if (frame.is_valid(x) && D > 0 && D < h_) {
        new_D.push_back(D);
      }
    }
  }

  double h2 = dt_*dt_;
  #pragma omp parallel for reduction( + : e )
  for (size_t i = 0; i < new_D.size(); ++i) {
    e += psi(new_D[i], h_, config_->kappa) / h2;
  }
  return e;
}

template<int DIM>
void Collision<DIM>::update(const Eigen::VectorXd& x, double dt) {
  // Get collision frames
  dt_ = dt;

  //std::cout << "d: " << d_ << std::endl;
  //std::cout << "D: " << D_ << std::endl;
  update_collision_frames(x);
  
  nframes_ = collision_frames_.size();
  MatrixXi T(nframes_, 3);
  for (int i = 0; i < nframes_; ++i) {
    T(i,0) = collision_frames_[i].E_(0);
    T(i,1) = collision_frames_[i].E_(1);
    T(i,2) = collision_frames_[i].E_(2);
  }
  // Structure potentially changes each step, so just rebuild assembler :/
  // NOTE assuming each local jacobian has same size!
  assembler_ = std::make_shared<Assembler<double,DIM,-1>>(T, mesh_->free_map_);
  vec_assembler_ = std::make_shared<VecAssembler<double,DIM,-1>>(T,
      mesh_->free_map_);
  //update_derivatives(dt);
  update_derivatives(1.0);
}

template<int DIM>
void Collision<DIM>::update_collision_frames(const Eigen::VectorXd& x) {

  // Build collision Frames
  std::vector<double> new_D;
  dd_dx_.clear();
  collision_frames_.clear();
  // For each boundary vertex find primitives within distance threshold
  for (int i = 0; i < C_.size(); ++i) {

    // Currently brute force check all primitives
    for (int j = 0; j < F_.rows(); ++j) {
      if (C_(i) == F_(j,0) || C_(i) == F_(j,1)) {
        continue;
      }

      // Build a frame and compute distance for the primitive - point pair
      CollisionFrame frame(F_(j,0), F_(j,1), C_(i));
      double D = frame.distance(x);

      // If valid and within distance thresholds add new frame
      if (frame.is_valid(x) && D > 0 && D < h_) {
        new_D.push_back(D);
        dd_dx_.push_back(frame.gradient(x));
        collision_frames_.push_back(frame);
      }
    }
  }
  D_ = Map<VectorXd>(new_D.data(), new_D.size());
}

template<int DIM>
void Collision<DIM>::update_derivatives(double dt) {

  double h2 = dt * dt;

  if (nframes_ == 0) {
    return;
  }

  data_.timer.start("Hinv");
  H_.resize(nframes_);
  g_.resize(nframes_);

  #pragma omp parallel for
  for (int i = 0; i < nframes_; ++i) {
    H_[i] = h2 * d2psi(D_(i), h_, config_->kappa);
    g_[i] = h2 * dpsi(D_(i), h_, config_->kappa);
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

  // saveMarket(assembler_->A, "lhs_c1.mkt");
  A_ = assembler_->A;

  // Gradient with respect to x variable
  std::vector<VectorXd> g(nframes_);
  for (int i = 0; i < nframes_; ++i) {
    g[i] = dd_dx_[i] * g_(i);
  }
  vec_assembler_->assemble(g, grad_);
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
  h_ = 2e-2; // 1e-3 in ipc
  g_.resize(0);
  H_.resize(0);
  grad_.resize(0);
  dd_dx_.resize(0);
  collision_frames_.clear();

  igl::boundary_facets(mesh_->T_, F_);
  assert(F_.cols() == 2); // Only supports 2D right now
  igl::unique(F_,C_); 
}

template<int DIM>
void Collision<DIM>::post_solve() {}

template class mfem::Collision<3>; // 3D
template class mfem::Collision<2>; // 2D
