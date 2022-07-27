#include "mixed_sqp_pd_optimizer.h"

#include "linesearch.h"
#include "mesh/mesh.h"
#include "factories/solver_factory.h"
#include "factories/integrator_factory.h"
#include <unsupported/Eigen/SparseExtra>


using namespace mfem;
using namespace Eigen;

template <int DIM>
void MixedSQPPDOptimizer<DIM>::step() {
  data_.clear();

  int i = 0;
  double grad_norm;
  double E = 0, E_prev = 0;
  // step_x.clear();
  //config_->kappa = 1e0;
  double kappa0 = config_->kappa;
  do {
    if (config_->save_substeps) {
      // VectorXd x = P_.transpose()*x_ + b_;
      // step_x.push_back(x);
      // step_v = vt_;
      // step_x0 = x0_;
    }

    // Update gradient and hessian
    update_system();
    
    if (i == 0) {
      //config_->kappa = -cvar_->rhs().dot(xvar_->rhs() + svar_->rhs()) /
      //  (1e-12 + cvar_->rhs().squaredNorm());
      //std::cout << " config_->kappa "<< config_->kappa << std::endl;
      //config_->kappa = std::max(1e-6, config_->kappa);
    }

    // Record initial energies
    // E = energy(x_, s_, la_);
    double res = std::abs((E - E_prev) / (E+1));
    E_prev = E;

    // Solve system
    substep(grad_norm);

    // Linesearch on descent direction
    double alpha = 1.0;
    //SolverExitStatus status = linesearch_backtracking_cubic(xvar_,
    //    {svar_,cvar_}, alpha, config_->ls_iters);
    Eigen::VectorXd x0 = xvar_->value();
    SolverExitStatus status = linesearch_backtracking(xvar_,
        {svar_,cvar_}, alpha, config_->ls_iters, 0.0, 0.9);

    // Record some data
    data_.add(" Iteration", i+1);
    data_.add("mixed E", E);
    data_.add("mixed E res", res);
    // data_.add("mixed grad", grad_.norm());
    data_.add("Newton dec", grad_norm);
    data_.add("alpha ", alpha);
    data_.add("kappa ", config_->kappa);
    //config_->kappa *= 2;
    ++i;

  } while (i < config_->outer_steps && grad_norm > config_->newton_tol
    /*`&& (res > 1e-12)*/);

  if (config_->show_data) {
    data_.print_data(config_->show_timing);
  }

  xvar_->post_solve();
  svar_->post_solve();
  cvar_->post_solve();
  config_->kappa = kappa0;
}

template <int DIM>
void MixedSQPPDOptimizer<DIM>::update_system() {

  VectorXd x = xvar_->value();
  xvar_->unproject(x);

  if (!mesh_->fixed_jacobian()) {
    mesh_->update_jacobian(x);
  }

  svar_->update(x, xvar_->integrator()->dt());
  //cvar_->update(x, xvar_->integrator()->dt());
  cvar_->update(x, 1.0);//xvar_->integrator()->dt());

  // Assemble blocks for left and right hand side
  std::cout << "NFRAMES: " << cvar_->num_collision_frames() << std::endl; 
  std::cout << "xvar_->rhs(): " << xvar_->rhs().norm() << std::endl;
  std::cout << "svar_->rhs(): " << svar_->rhs().norm() << std::endl;
  std::cout << "cvar_->rhs(): " << cvar_->rhs().norm() << std::endl;


  lhs_ = xvar_->lhs() + svar_->lhs() + cvar_->lhs();
  rhs_ = xvar_->rhs() + svar_->rhs() + cvar_->rhs();
  //saveMarket(lhs_, "lhs.mkt");
  //saveMarket(xvar_->lhs(), "lhs_x.mkt");
  //saveMarket(svar_->lhs(), "lhs_s.mkt");
  //saveMarket(cvar_->lhs(), "lhs_c.mkt");
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

  data_.timer.start("local-c");
  VectorXd fuck(xvar_->delta());
  xvar_->unproject(fuck);
  cvar_->solve(fuck);
  //cvar_->solve(xvar_->delta());
  data_.timer.stop("local-c");

  data_.add("||x delta||", xvar_->delta().norm());
  data_.add("||s delta||", svar_->delta().norm());
  data_.add("||c delta||", cvar_->delta().norm());
  //std::cout << "Cvar delta: " << cvar_->delta() << std::endl;
  if (cvar_->delta().hasNaN()) {
    std::cout << "xvar delta(): " << xvar_->delta().norm() << std::endl;
    exit(1);
  }

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
  cvar_ = std::make_shared<Collision<DIM>>(mesh_, config_);
  cvar_->reset();

  SolverFactory solver_factory;
  solver_ = solver_factory.create(config_->solver_type, mesh_, config_);
}

template class mfem::MixedSQPPDOptimizer<3>;
template class mfem::MixedSQPPDOptimizer<2>;
