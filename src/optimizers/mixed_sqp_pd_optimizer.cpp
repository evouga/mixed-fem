#include "mixed_sqp_pd_optimizer.h"

#include "linesearch.h"
#include "mesh/mesh.h"
#include "factories/solver_factory.h"
#include "factories/integrator_factory.h"

using namespace mfem;
using namespace Eigen;

template <int DIM>
void MixedSQPPDOptimizer<DIM>::step() {
  data_.clear();

  int i = 0;
  double grad_norm;
  double E = 0, E_prev = 0;
  // step_x.clear();
  do {
    if (config_->save_substeps) {
      // VectorXd x = P_.transpose()*x_ + b_;
      // step_x.push_back(x);
      // step_v = vt_;
      // step_x0 = x0_;
    }

    // Update gradient and hessian
    update_system();

    // Record initial energies
    // E = energy(x_, s_, la_);
    double res = std::abs((E - E_prev) / (E+1));
    E_prev = E;

    // Solve system
    substep(grad_norm);

    // Linesearch on descent direction
    double alpha = 1.0;
    SolverExitStatus status = linesearch_backtracking_cubic(xvar_, {svar_}, alpha,
        config_->ls_iters);

    // Record some data
    data_.add(" Iteration", i+1);
    data_.add("mixed E", E);
    data_.add("mixed E res", res);
    // data_.add("mixed grad", grad_.norm());
    data_.add("Newton dec", grad_norm);
    ++i;

  } while (i < config_->outer_steps && grad_norm > config_->newton_tol
    /*`&& (res > 1e-12)*/);

  if (config_->show_data) {
    data_.print_data(config_->show_timing);
  }

  xvar_->post_solve();
  svar_->post_solve();
}

template <int DIM>
void MixedSQPPDOptimizer<DIM>::update_system() {

  VectorXd x = xvar_->value();
  xvar_->unproject(x);

  if (!mesh_->fixed_jacobian()) {
    mesh_->update_jacobian(x);
  }

  svar_->update(x, xvar_->integrator()->dt());

  // Assemble blocks for left and right hand side
  lhs_ = xvar_->lhs() + svar_->lhs();
  rhs_ = xvar_->rhs() + svar_->rhs();
}

template <int DIM>
void MixedSQPPDOptimizer<DIM>::substep(double& decrement) {
  data_.timer.start("global");
  solver_->compute(lhs_);
  xvar_->delta() = solver_->solve(rhs_);
  data_.timer.stop("global");

  data_.timer.start("local");
  svar_->solve(xvar_->delta());
  data_.timer.stop("local");

  decrement = std::max(xvar_->delta().template lpNorm<Infinity>(),
                       svar_->delta().template lpNorm<Infinity>());
}

template <int DIM>
void MixedSQPPDOptimizer<DIM>::reset() {
  Optimizer<DIM>::reset();

  svar_ = std::make_shared<Stretch<DIM>>(mesh_);
  svar_->reset();
  xvar_ = std::make_shared<Displacement<DIM>>(mesh_, config_);
  xvar_->reset();

  SolverFactory solver_factory;
  solver_ = solver_factory.create(config_->solver_type, mesh_, config_);
}

template class mfem::MixedSQPPDOptimizer<3>;
template class mfem::MixedSQPPDOptimizer<2>;
