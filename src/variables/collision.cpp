#include "collision.h"
#include "mesh/mesh.h"
#include "igl/unique.h"
#include "igl/boundary_facets.h"
#include "simple_psd_fix.h"
#include "config.h"
#include "CTCD.h"
#include "Distance.h"

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
double Collision<DIM>::energy(const VectorXd& d) {

  double e = 0;
  return e;
  return e;
}


template <int DIM>
double Collision<DIM>::constraint_value(const VectorXd& x,
    const VectorXd& d) {

  //std::cout << "Energy() d: " << d << std::endl;
  //if ( (d.array() <= 0.0).any())
  //  return 1e8;
  double e = 0;
  #pragma omp parallel for reduction( + : e )
  for (int i = 0; i < nframes_; ++i) {
    e += dt_*dt_*(psi(d(i), h_, config_->kappa)
      - la_(i) * (collision_frames_[i].distance(x) - d(i)));
  }
  //std::cout << "energy (e): " << e << std::endl;
  // d - (p-x0)^T R(x) * N 
  //std::cout << " e: " << e << std::endl;
  //std::cout << " la norm: " << la_.norm() << std::endl;
  return -e; // negating cause my dumb fuckin linesearch negates it.....
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
  update_derivatives(dt);
}

template<int DIM>
void Collision<DIM>::update_collision_frames(const Eigen::VectorXd& x) {
  // Compute D_, dd_dx_
  // Update:
  // * Get boundary_facets
  // * Check all point-edge pairs
  //  - If less than h, create collision frame
  //  - Collision frame just has vids
  //
  // Detect Collision Frames
  // Initialize distance variables 
  std::vector<double> new_D;
  std::vector<double> new_d;
  std::vector<double> new_lambda;
  std::vector<CollisionFrame> new_frames;
  std::map<std::tuple<int,int,int>, int> new_ids;
  // For each boundary vertex find primitives within distance threshold
  for (int i = 0; i < C_.size(); ++i) {

    // Currently brute force check all primitives
    for (int j = 0; j < F_.rows(); ++j) {
      if (C_(i) == F_(j,0) || C_(i) == F_(j,1)) {
        continue;
      }

      // Use tuple of vertex ids for hashing
      std::tuple<int,int,int> tup = std::make_tuple(F_(j,0),F_(j,1), C_(i));
      auto it = frame_ids_.find(tup);

      // Build a frame and compute distance for the primitive - point pair
      CollisionFrame frame(F_(j,0), F_(j,1), C_(i));
      double D = frame.distance(x);
      double la = 0;
      double d = D; 

      // Check if frame already exists and maintain its variable
      // and lagrange multiplier values 
      if (it != frame_ids_.end()) {
        la = la_(it->second);
        d = d_(it->second);
      }

      // If valid and within distance thresholds add new frame
      if (frame.is_valid(x) && D > 0 && D < h_) {
        new_D.push_back(D);
        new_d.push_back(d);
        new_lambda.push_back(la);
        dd_dx_.push_back(frame.gradient(x));
        new_frames.push_back(frame);
        new_ids[tup] = new_frames.size() - 1;
      }
    }
  }
  D_ = Map<VectorXd>(new_D.data(), new_D.size());
  d_ = Map<VectorXd>(new_d.data(), new_d.size());
  la_ = Map<VectorXd>(new_lambda.data(), new_lambda.size());
  std::swap(new_ids, frame_ids_);
  std::swap(new_frames, collision_frames_);
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
    H_[i] = h2 * d2psi(d_(i),h_, config_->kappa);
    g_[i] = h2 * dpsi(d_(i),h_, config_->kappa);
  }
  data_.timer.stop("Hinv");
  
  data_.timer.start("Local H");
  Aloc_.resize(nframes_);
  #pragma omp parallel for
  for (int i = 0; i < nframes_; ++i) {
    Aloc_[i] = dd_dx_[i] * H_(i) * dd_dx_[i].transpose();
    sim::simple_psd_fix(Aloc_[i]);
    //Eigen::VectorXx<Scalar> diag_eval = es.eigenvalues().real();
    //std::cout << " d_(i) : " << d_(i) << std::endl;
    //std::cout << " g_(i) : " << g_(i) << std::endl;
    //std::cout << " H_(i) : " << H_(i) << std::endl;
  }
  data_.timer.stop("Local H");

  data_.timer.start("Update LHS");
  assembler_->update_matrix(Aloc_);
  data_.timer.stop("Update LHS");

  // saveMarket(assembler_->A, "lhs_c1.mkt");

  A_ = assembler_->A;
// std::cout << "A1: \n " << MatrixXd(A_) << std::endl;
  // std::cout << "nframes: " << nframes_ << std::endl;
  // saveMarket(assembler_->A, "lhs_c2.mkt");
// std::cout << "A2: \n " << A_ << std::endl;
std::cout << "E_ : " << collision_frames_[0].E_ << std::endl;

  // Gradient with respect to x variable
  std::vector<VectorXd> g(nframes_);
  for (int i = 0; i < nframes_; ++i) {
    g[i] = -dd_dx_[i] * g_(i);
  }
  vec_assembler_->assemble(g, grad_x_);

  // Gradient with respect to mixed variable
  grad_ = g_ + la_;
}

template<int DIM>
VectorXd Collision<DIM>::rhs() {
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
VectorXd Collision<DIM>::gradient() {
  if (nframes_ == 0) {
    grad_x_.resize(mesh_->jacobian().rows());
    grad_x_.setZero();
  }
  return grad_x_;
}

template<int DIM>
VectorXd Collision<DIM>::gradient_mixed() {
  if (nframes_ == 0) {
    grad_.resize(0);
  }
  return grad_;
}

template<int DIM>
void Collision<DIM>::solve(const VectorXd& dx) {
  if (nframes_ == 0) {
    return;
  }

  data_.timer.start("local");
  std::vector<VectorXd> g(nframes_);
  Gdx_.resize(d_.size());
  #pragma omp parallel for
  for (int i = 0; i < nframes_; ++i) {
    Matrix<double,DIM*3,1> q;
    const Vector3i& E = collision_frames_[i].E_;
    // WRONG
    q << dx.segment<DIM>(DIM*E(0)), dx.segment<DIM>(DIM*E(1)),
         dx.segment<DIM>(DIM*E(2));
    Gdx_(i) = -q.dot(dd_dx_[i]);
  }
  la_ = -gl_.array() + (H_.array() * Gdx_.array());
  delta_ = -(la_ + g_).array() / H_.array();
  data_.timer.stop("local");
}

template<int DIM>
void Collision<DIM>::reset() {
  h_ = 1e-1; // 1e-3 in ipc
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
  collision_frames_.clear();

  igl::boundary_facets(mesh_->T_, F_);
  assert(F_.cols() == 2); // Only supports 2D right now
  igl::unique(F_,C_); 
}

template<int DIM>
void Collision<DIM>::post_solve() {
  la_.setZero();
  dd_dx_.clear();
  collision_frames_.clear();
  frame_ids_.clear();
}

template<int DIM>
double Collision<DIM>::max_possible_step(const VectorXd& x1,
    const VectorXd& x2) {

  double min_step = 1.0;
  double eta0 = 1e-8;

  #pragma omp parallel for reduction(min:min_step)
  for (int i = 0; i < F_.rows(); ++i) {
    for (int j = 0; j < F_.rows(); ++j) {

      if (F_(i,0) == F_(j,0) || F_(i,0) == F_(j,1) || F_(i,1) == F_(j,0)
          || F_(i,1) == F_(j,1)) {
        continue;
      }
      const Vector2d& p0_2d_start = x1.segment<2>(2*F_(i,0));
      const Vector2d& p1_2d_start = x1.segment<2>(2*F_(i,1));
      const Vector2d& q0_2d_start = x1.segment<2>(2*F_(j,0));
      const Vector2d& q1_2d_start = x1.segment<2>(2*F_(j,1));
      const Vector2d& p0_2d_end = x2.segment<2>(2*F_(i,0));
      const Vector2d& p1_2d_end = x2.segment<2>(2*F_(i,1));
      const Vector2d& q0_2d_end = x2.segment<2>(2*F_(j,0));
      const Vector2d& q1_2d_end = x2.segment<2>(2*F_(j,1));

      Vector3d p0start(p0_2d_start(0), p0_2d_start(1), 0);
      Vector3d p1start(p1_2d_start(0), p1_2d_start(1), 0);
      Vector3d q0start(q0_2d_start(0), q0_2d_start(1), 0);
      Vector3d q1start(q1_2d_start(0), q1_2d_start(1), 0);
      Vector3d p0end(p0_2d_end(0), p0_2d_end(1), 0);
      Vector3d p1end(p1_2d_end(0), p1_2d_end(1), 0);
      Vector3d q0end(q0_2d_end(0), q0_2d_end(1), 0);
      Vector3d q1end(q1_2d_end(0), q1_2d_end(1), 0);
      double t = 1;

      double d_sqrt;
      double tmp;
      d_sqrt = Distance::edgeEdgeDistance(p0start,p1start,q0start,
          q1start,tmp,tmp,tmp,tmp).norm();
      double eta = d_sqrt * eta0;
      if (CTCD::edgeEdgeCTCD(p0start,p1start,q0start,q1start,
            p0end,p1end,q0end,q1end,eta,t)) {
        t = (t<1e-12) ? 1 : t;
        min_step = std::min(min_step,t);
      }
    }
  }
  return min_step;
}

template class mfem::Collision<3>; // 3D
template class mfem::Collision<2>; // 2D
