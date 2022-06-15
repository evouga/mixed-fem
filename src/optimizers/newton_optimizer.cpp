#include "newton_optimizer.h"

#include <chrono>
#include "linesearch.h"
#include "pinning_matrix.h"
#include "energies/material_model.h"
#include "mesh/mesh.h"

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
    
    if (config_->save_substeps) {
      VectorXd x = P_.transpose()*x_ + b_;
      step_x.push_back(x);
      step_v = vt_;
      step_x0 = x0_;
    }
    
    // Update jacobians if necessary
    if (!mesh_->fixed_jacobian()) {
      VectorXd x = P_.transpose()*x_ + b_;
      mesh_->update_jacobian(x);
    }

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
    double alpha = 1.0;
    SolverExitStatus status = linesearch_backtracking_bisection(x_, dx_, value,
        tmp, alpha, config_->ls_iters, 0.1, 0.5, E_prev_);
    bool done = status == MAX_ITERATIONS_REACHED;

    double E = energy(x_);
    double res = std::abs((E - E_prev_) / E);
    data_.add(" Iteration", i+1);
    data_.add("Energy", E);
    data_.add("Energy res", res);
    data_.add("||H^-1 g||", grad_norm);
    data_.add("||g||", rhs_.norm());

    E_prev_ = E;

    ++i;
  } while (i < config_->outer_steps && grad_norm > config_->newton_tol);

  data_.print_data();
  update_configuration();
}

void NewtonOptimizer::build_lhs() {
  double h = config_->h;
  double h2 = h*h;

  VectorXd def_grad;
  mesh_->deformation_gradient(P_.transpose()*x_+b_, def_grad);

  const std::vector<MatrixXd>& Jloc = mesh_->local_jacobians();
  std::vector<MatrixXd> Hloc(nelem_); 
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector9d& F = def_grad.segment<9>(9*i);
    
    Hloc[i] = (Jloc[i].transpose() * mesh_->material_->hessian(F)
        * Jloc[i]) * vols_[i] * h2;
  }
  assembler_->update_matrix(Hloc);

  lhs_ = assembler_->A;
  lhs_ += PMP_;
}

void NewtonOptimizer::build_rhs() {
  double h = config_->h;
  double h2 = h*h;

  VectorXd x = P_.transpose()*x_ + b_;

  VectorXd g;
  VectorXd def_grad;
  mesh_->deformation_gradient(x, def_grad);

  const std::vector<MatrixXd>& Jloc = mesh_->local_jacobians();
  std::vector<VectorXd> gloc(nelem_); 
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector9d& F = def_grad.segment<9>(9*i);
    gloc[i] = Jloc[i].transpose()
        * mesh_->material_->gradient(F) * vols_[i] * h2;
  }
  vec_assembler_->assemble(gloc, g);

  rhs_ = -(g + PM_*(x - x0_ - h*vt_ - h2*f_ext_));
}

void NewtonOptimizer::substep(bool init_guess, double& decrement) {
  // Factorize and solve system
  solver_.compute(lhs_);
  if(solver_.info()!=Success) {
   std::cerr << "prefactor failed! " << std::endl;
   exit(1);
  }

  // Solve for update
  dx_ = solver_.solve(rhs_);

  double relative_error = (lhs_*dx_ - rhs_).norm() / rhs_.norm();
  decrement = dx_.norm(); // if doing "full newton use this"
  decrement = dx_.dot(rhs_);
}

void NewtonOptimizer::update_vertices(const Eigen::MatrixXd& V) {
  MatrixXd tmp = V.transpose();
  VectorXd x = Map<VectorXd>(tmp.data(), V.size());
  x0_ = x;
  vt_ = 0*x;
  b_ = x - P_.transpose()*P_*x;
  x_ = P_ * x;
}

void NewtonOptimizer::set_state(const Eigen::VectorXd& x,
    const Eigen::VectorXd& v) {
  MatrixXd V = Map<const MatrixXd>(x.data(), mesh_->V_.cols(), mesh_->V_.rows());
  mesh_->V_ = V.transpose();
  x0_ = x;
  vt_ = v;
  b_ = x - P_.transpose()*P_*x;
  x_ = P_ * x;
  std::cout << "set_state: " << vt_.norm() << std::endl;
}

void NewtonOptimizer::update_configuration() {
  // Update boundary positions
  BCs_.step_script(mesh_, config_->h);
  #pragma omp parallel for
  for (int i = 0; i < mesh_->V_.rows(); ++i) {
    if (mesh_->is_fixed_(i)) {
      b_.segment(3*i,3) = mesh_->V_.row(i).transpose();
    }
  }

  VectorXd x = P_.transpose()*x_ + b_;
  vt_ = (x - x0_) / config_->h;
  x0_ = x;

  // Update mesh vertices
  MatrixXd V = Map<MatrixXd>(x.data(), mesh_->V_.cols(), mesh_->V_.rows());
  mesh_->V_ = V.transpose();
}

double NewtonOptimizer::energy(const VectorXd& x) {
  double h = config_->h;
  double h2 = h*h;
  VectorXd xt = P_.transpose()*x + b_;
  VectorXd xdiff = xt - x0_ - h*vt_ - h*h*f_ext_;
  
  double Em = 0.5*xdiff.transpose()*M_*xdiff;

  VectorXd def_grad;
  mesh_->deformation_gradient(xt, def_grad);
  double Epsi = 0.0;
  #pragma omp parallel for reduction(+ : Epsi)
  for (int i = 0; i < nelem_; ++i) {
    const Vector9d& F = def_grad.segment<9>(9*i);
    Epsi += mesh_->material_->energy(F) * vols_[i];
  }
  double e = Em + Epsi*h2;
  return e;
}

void NewtonOptimizer::reset() {
  // Reset variables
  Optimizer::reset();

  E_prev_ = 0;
  mesh_->volumes(vols_);

  SparseMatrixdRowMajor tmpM;
  mesh_->mass_matrix(tmpM, vols_);
  M_ = tmpM;
  
  MatrixXd tmp = mesh_->V_.transpose();
  x_ = Map<VectorXd>(tmp.data(), mesh_->V_.size());

  x0_ = x_;
  vt_ = 0*x_;
  b_ = x_ - P_.transpose()*P_*x_;
  x_ = P_ * x_;
  dx_ = 0*x_;

  // Project out mass matrix pinned point
  PM_ = P_ * M_;
  PMP_ = PM_ * P_.transpose();

  // External gravity force
  Vector3d ext = Map<Vector3f>(config_->ext).cast<double>();
  f_ext_ = P_.transpose()*P_*ext.replicate(mesh_->V_.rows(),1);

  assembler_ = std::make_shared<Assembler<double,3>>(mesh_->T_, mesh_->free_map_);
  vec_assembler_ = std::make_shared<VecAssembler<double,3>>(mesh_->T_,
      mesh_->free_map_);
}
