#include "newton_optimizer.h"

#include <chrono>
#include "sparse_utils.h"
#include "linesearch.h"
#include "pinning_matrix.h"
#include "linear_tetmesh_dphi_dX.h"
#include "assemble.h"

#include <amgcl/backend/eigen.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/cg.hpp>

using namespace mfem;
using namespace Eigen;
using namespace std::chrono;

void NewtonOptimizer::step() {

  std::cout << "/////////////////////////////////////////////" << std::endl;
  std::cout << " SQP Simulation step " << std::endl;
  
  data_.clear();
  E_prev_ = 0;

  int i = 0;
  double grad_norm;
  do {
    // Update gradient and hessian
    build_lhs();
    build_rhs();

    // Compute search direction
    substep(i==0, grad_norm);

    // Linesearch over search direction
    auto value = [&](const VectorXd& x)->double {
      return energy(x);
    };
    VectorXd tmp;
    SolverExitStatus status = linesearch_backtracking_bisection(x_, dx_, value,
        tmp, config_->ls_iters, 1.0, 0.1, 0.5, E_prev_);
    bool done = status == MAX_ITERATIONS_REACHED;

    double E = energy(x_);
    double res = std::abs((E - E_prev_) / E);
    data_.egrad_.push_back(rhs_.norm());
    data_.energies_.push_back(E);
    data_.energy_residuals_.push_back(res);
    E_prev_ = E;

    ++i;
  } while (i < config_->outer_steps && grad_norm > config_->newton_tol);

  data_.print_data();
  update_configuration();
}

void NewtonOptimizer::build_lhs() {
  double h = config_->h;
  double h2 = h*h;

  auto assemble_func = [&](auto &H,  auto &e, 
      const auto &dphidX, const auto &volume) { 
    Matrix<double, 4,3> dX = sim::unflatten<4,3>(dphidX);

    VectorXd x = P_.transpose()*x_ + b_;
    // Local block
    // NOTE: assuming tetmesh
    Matrix<double,9,12> B;
    B  << dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0, 0, dX(3,0), 0, 0,
          0, dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0, 0, dX(3,0), 0,
          0, 0, dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0, 0, dX(3,0),
          dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0, 0, dX(3,1), 0, 0, 
          0, dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0, 0, dX(3,1), 0,
          0, 0, dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0, 0, dX(3,1),
          dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0, 0, dX(3,2), 0, 0, 
          0, dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0, 0, dX(3,2), 0,
          0, 0, dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0, 0, dX(3,2);
    Vector12d qe;
    qe << x.segment(3*e(0),3), x.segment(3*e(1),3),
       x.segment(3*e(2),3), x.segment(3*e(3),3);
    Vector9d F = B * qe;

    Matrix12d tmp = (B.transpose() * object_->material_->hessian(F) * B) *volume(0)*h2;
    H = tmp.eval();
  };

  Eigen::Matrix12d Htmp;
  SparseMatrixd K;
  sim::assemble(K, object_->V_.size(), object_->V_.size(), 
      object_->T_, object_->T_, assemble_func, Htmp, dphidX_, vols_);
  lhs_ = P_ * K * P_.transpose() ;//
  lhs_ += M_;
}

void NewtonOptimizer::build_rhs() {
  double h = config_->h;
  double h2 = h*h;

  VectorXd x = P_.transpose()*x_ + b_;

  auto assemble_func = [&](auto &H,  auto &e, 
      const auto &dphidX, const auto &volume) { 
    Matrix<double, 4,3> dX = sim::unflatten<4,3>(dphidX);

    // Local block
    // NOTE: assuming tetmesh
    Matrix<double,9,12> B;
    B  << dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0, 0, dX(3,0), 0, 0,
          0, dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0, 0, dX(3,0), 0,
          0, 0, dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0, 0, dX(3,0),
          dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0, 0, dX(3,1), 0, 0, 
          0, dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0, 0, dX(3,1), 0,
          0, 0, dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0, 0, dX(3,1),
          dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0, 0, dX(3,2), 0, 0, 
          0, dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0, 0, dX(3,2), 0,
          0, 0, dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0, 0, dX(3,2);
    Vector12d qe;
    qe << x.segment(3*e(0),3), x.segment(3*e(1),3),
       x.segment(3*e(2),3), x.segment(3*e(3),3);
    Vector9d F = B * qe;

    H = B.transpose() * object_->material_->gradient(F) * volume(0) * h2;
  };
  Vector12d Htmp;
  VectorXd g;
  sim::assemble(g, object_->V_.size(), object_->T_, object_->T_,
      assemble_func, Htmp, dphidX_, vols_);

  // inertial term
  rhs_ = -(P_*g + M_*(x_ - x0_ - h*vt_ - h2*f_ext_));

  double gradx = rhs_.norm();
  double grads = 0;
  double gradl = 0;
  data_.egrad_x_.push_back(gradx);
  data_.egrad_s_.push_back(std::sqrt(grads));
  data_.egrad_la_.push_back(std::sqrt(gradl));
}

void NewtonOptimizer::substep(bool init_guess, double& decrement) {
  // Factorize and solve system
  // solver_.compute(lhs_);
  // if(solver_.info()!=Success) {
  //  std::cerr << "prefactor failed! " << std::endl;
  //  exit(1);
  // }

  // // Solve for update
  // dx_ = solver_.solve(rhs_);

  SparseMatrix<double, RowMajor> A = lhs_;
  typedef amgcl::backend::eigen<double> Backend;
  typedef amgcl::make_solver<
          amgcl::amg<
              Backend,
              amgcl::coarsening::smoothed_aggregation,
              amgcl::relaxation::spai0
              >,
          amgcl::solver::cg<Backend>
          > Solver;
  Solver solve(A);
  std::cout << solve << std::endl;
  int iters;
  double error;
  std::tie(iters, error) = solve(rhs_, dx_);
  std::cout << "Iters: " << iters << std::endl
            << "Error: " << error << std::endl;

  double relative_error = (lhs_*dx_ - rhs_).norm() / rhs_.norm();
  decrement = dx_.norm(); // if doing "full newton use this"
}

void NewtonOptimizer::update_configuration() {

  vt_ = (x_ - x0_) / config_->h;
  x0_ = x_;

  // Update mesh vertices
  VectorXd x = P_.transpose()*x_ + b_;
  MatrixXd V = Map<MatrixXd>(x.data(), object_->V_.cols(), object_->V_.rows());
  object_->V_ = V.transpose();
}

double NewtonOptimizer::energy(const VectorXd& x) {
  double h = config_->h;
  double h2 = h*h;
  VectorXd xdiff = x - x0_ - h*vt_ - h*h*f_ext_;
  
  double Em = 0.5*xdiff.transpose()*M_*xdiff;

  VectorXd def_grad = J_*(P_.transpose()*x+b_);
  double Epsi = 0.0;
  #pragma omp parallel for reduction(+ : Epsi)
  for (int i = 0; i < nelem_; ++i) {
    Vector9d F = def_grad.segment<9>(9*i);
    Epsi += object_->material_->energy(F) * vols_[i];
  }
  double e = Em + Epsi*h2;
  return e;
}

void NewtonOptimizer::reset() {
  // Reset variables
  nelem_ = object_->T_.rows();

  object_->V_ = object_->V0_;

  E_prev_ = 0;
  object_->volumes(vols_);
  object_->mass_matrix(M_, vols_);
  object_->jacobian(J_, vols_, false);

  // Pinning matrices
  double min_x = object_->V_.col(0).minCoeff();
  double max_x = object_->V_.col(0).maxCoeff();
  double pin_x = min_x + (max_x-min_x)*0.2;
  double min_y = object_->V_.col(1).minCoeff();
  double max_y = object_->V_.col(1).maxCoeff();
  double pin_y = max_y - (max_y-min_y)*0.1;
  //double pin_y = min_y + (max_y-min_y)*0.1;
  //pinnedV_ = (V_.col(0).array() < pin_x).cast<int>(); 
  pinnedV_ = (object_->V_.col(1).array() > pin_y).cast<int>();
  //pinnedV_ = (V_.col(0).array() < pin_x 
  //    && V_.col(1).array() > pin_y).cast<int>();
  //pinnedV_.resize(V_.rows());
  pinnedV_.setZero();
  pinnedV_(0) = 1;

  P_ = pinning_matrix(object_->V_, object_->T_, pinnedV_, false);

  MatrixXd tmp = object_->V_.transpose();
  x_ = Map<VectorXd>(tmp.data(), object_->V_.size());

  b_ = x_ - P_.transpose()*P_*x_;
  x_ = P_ * x_;
  x0_ = x_;
  dx_ = 0*x_;
  vt_ = 0*x_;

  // Project out mass matrix pinned point
  M_ = P_ * M_ * P_.transpose();

  // External gravity force
  Vector3d ext = Map<Vector3f>(config_->ext).cast<double>();
  f_ext_ = P_ * ext.replicate(object_->V_.rows(),1);

  // J matrix (big jacobian guy)
  sim::linear_tetmesh_dphi_dX(dphidX_, object_->V_, object_->T_);
}
