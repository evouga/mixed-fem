#pragma once

#include "linear_solver.h"
#include "linear_solvers/preconditioners/laplacian_preconditioner.h"
#include "linear_solvers/preconditioners/dual_ascent_preconditioner.h"
#include "ccd/additive_ccd.h"

namespace mfem {

  template <typename Solver, typename SystemMatrix, typename Scalar, int DIM>
  class EigenIterativeSolver : public LinearSolver<Scalar, DIM> {

    typedef LinearSolver<Scalar, DIM> Base;
    // typedef Eigen::LaplacianPreconditioner<Scalar, DIM> InitialGuesser;
    typedef Eigen::DualAscentPreconditioner<Scalar, DIM> InitialGuesser;

  public:

    EigenIterativeSolver(SimState<DIM>* state) 
        : LinearSolver<Scalar,DIM>(state) {
      solver_.setMaxIterations(state->config_->max_iterative_solver_iters);
      solver_.setTolerance(state->config_->itr_tol);
      
      // guesser_.init(state);
    }

    void solve() override {
      system_matrix_.pre_solve(Base::state_);

      solver_.compute(system_matrix_.A());
      // tmp_ = solver_.solve(system_matrix_.b());
      // std::cout << "Solver will crash on non PD systems!" << std::endl;

      double h = Base::state_->x_->integrator()->dt();
      const auto& P = Base::state_->mesh_->projection_matrix();
      // tmp_ = -(Base::state_->x_->value() - P*Base::state_->x_->integrator()->x_tilde()) 
      // + h*h*P*Base::state_->mesh_->external_force();  

      // Get current x position as well as its explicit euler guess
      Eigen::VectorXd x1 = Base::state_->x_->value();
      Base::state_->x_->unproject(x1);
      Eigen::VectorXd p =  -(x1 - Base::state_->x_->integrator()->x_tilde())
          + h*h*Base::state_->mesh_->external_force();

      // Perform CCD on this explicit euler guess
      ipc::Candidates candidates;
      double alpha = ipc::additive_ccd<DIM>(x1, p,
          Base::state_->mesh_->collision_mesh(), candidates,
          Base::state_->config_->dhat);
      // std::cout << "ALPHA: " << alpha << std::endl;
      tmp_ = alpha * P * p; 

      // tmp_.setZero();
      // guesser_.update_gradients();
      // tmp_ = guesser_.solve(system_matrix_.b());
      tmp_ = solver_.solveWithGuess(system_matrix_.b(), tmp_);

      // TODO support integral types
      OptimizerData::get().add("Solver iters", double(solver_.iterations()));
      OptimizerData::get().add("Solver error", solver_.error());

      std::cout << "- CG iters: " << solver_.iterations() << std::endl;
      std::cout << "- CG error: " << solver_.error() << std::endl;
      system_matrix_.post_solve(Base::state_, tmp_);
    }

    Solver& eigen_solver() {
      return solver_;
    }

  private:

    InitialGuesser guesser_;
    SystemMatrix system_matrix_;
    Solver solver_;
    Eigen::VectorXx<Scalar> tmp_;
  };
}
