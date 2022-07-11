#include "newton_optimizer.h"

#include <chrono>
#include "linesearch.h"
#include "pinning_matrix.h"
#include "energies/material_model.h"
#include "mesh/mesh.h"
#include "factories/solver_factory.h"


using namespace mfem;
using namespace Eigen;
using namespace std::chrono;

void NewtonOptimizer::step() {
  data_.clear();
  E_prev_ = 0;

  int i = 0;
  double grad_norm;
  do {
    
    // if (config_->save_substeps) {
    //   VectorXd x = P_.transpose()*x_ + b_;
    //   step_x.push_back(x);
    //   step_v = vt_;
    //   step_x0 = x0_;
    // }

    VectorXd x = xvar_->value();
    xvar_->unproject(x);

    if (!mesh_->fixed_jacobian()) {
      mesh_->update_jacobian(x);
    }

    xvar_->update(x,0.);

    // Assemble blocks for left and right hand side
    lhs_ = xvar_->lhs();
    rhs_ = xvar_->rhs();

    // Compute search direction
    substep(grad_norm);

    double alpha = 1.0;
    SolverExitStatus status = linesearch_backtracking_cubic(xvar_, {}, alpha,
        config_->ls_iters);
    bool done = status == MAX_ITERATIONS_REACHED;

    double E = xvar_->energy(xvar_->value());
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
  xvar_->post_solve();
}


void NewtonOptimizer::substep(double& decrement) {
  // Factorize and solve system
  solver_->compute(lhs_);

  // Solve for update
  xvar_->delta() = solver_->solve(rhs_);

  double relative_error = (lhs_*xvar_->delta() - rhs_).norm() / rhs_.norm();
  decrement = xvar_->delta().norm();
  //decrement = dx_.dot(rhs_);
}

void NewtonOptimizer::update_vertices(const Eigen::MatrixXd& V) {
  // MatrixXd tmp = V.transpose();
  // VectorXd x = Map<VectorXd>(tmp.data(), V.size());
  // x0_ = x;
  // vt_ = 0*x;
  // b_ = x - P_.transpose()*P_*x;
  // x_ = P_ * x;
}

void NewtonOptimizer::set_state(const Eigen::VectorXd& x,
    const Eigen::VectorXd& v) {
  // MatrixXd V = Map<const MatrixXd>(x.data(), mesh_->V_.cols(), mesh_->V_.rows());
  // mesh_->V_ = V.transpose();
  // x0_ = x;
  // vt_ = v;
  // b_ = x - P_.transpose()*P_*x;
  // x_ = P_ * x;
  // std::cout << "set_state: " << vt_.norm() << std::endl;
}

void NewtonOptimizer::reset() {
  // Reset variables
  Optimizer::reset();

  xvar_ = std::make_shared<Displacement<3>>(mesh_, config_);
  xvar_->set_mixed(false);
  xvar_->reset();

  SolverFactory solver_factory;
  solver_ = solver_factory.create(config_->solver_type, mesh_, config_);
}
