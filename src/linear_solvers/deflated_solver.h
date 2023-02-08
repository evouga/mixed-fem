#pragma once

#include "linear_solver.h"
#include "utils/additive_ccd.h"
#include "linear_solvers/preconditioners/deflated_block_jacobi.h"

namespace mfem {

  template <typename Solver, typename SystemMatrix, typename Scalar, int DIM>
  class DeflatedSolver : public LinearSolver<Scalar, DIM> {

    typedef LinearSolver<Scalar, DIM> Base;
    // typedef Eigen::LaplacianPreconditioner<Scalar, DIM> InitialGuesser;
    // typedef Eigen::DualAscentPreconditioner<Scalar, DIM> InitialGuesser;

    // SFINAE residual test
    template <typename T>
    class has_residual
    {
        typedef char one;
        struct two { char x[2]; };

        template <typename C> static one test( decltype(&C::getResiduals) ) ;
        template <typename C> static two test(...);    

    public:
        enum { value = sizeof(test<T>(0)) == sizeof(char) };
    };

  public:

    DeflatedSolver(SimState<DIM>* state) : LinearSolver<Scalar,DIM>(state) {
      solver_.setMaxIterations(state->config_->max_iterative_solver_iters);
      solver_.setTolerance(state->config_->itr_tol);
      
      if constexpr (has_residual<Solver>::value) {
        solver_.setSaveResiduals(state->config_->itr_save_residuals);
      }
      // guesser_.init(state);
    }

    void solve() override {
      system_matrix_.pre_solve(Base::state_);

      solver_.compute(system_matrix_.A());
      // tmp_ = solver_.solve(system_matrix_.b());
      // std::cout << "Solver will crash on non PD systems!" << std::endl;

      if (Base::state_->config_->itr_explicit_guess) {

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
        double alpha = 1.0;

        if (Base::state_->config_->itr_guess_ccd) {
          ipc::Candidates candidates;
          alpha = ipc::additive_ccd<DIM>(x1, p,
              Base::state_->mesh_->collision_mesh(), candidates,
              Base::state_->config_->dhat);
        }
        tmp_ = alpha * P * p; 
      } else {
        tmp_.resizeLike(system_matrix_.b());
        tmp_.setZero();
      }
      double res_pre = (system_matrix_.A()*tmp_ - system_matrix_.b()).norm();
      solver_.preconditioner().guess(system_matrix_.b(), tmp_);
      double res_post = (system_matrix_.A()*tmp_ - system_matrix_.b()).norm();
      std::cout << "Residual before: " << res_pre << std::endl;
      std::cout << "Residual after: " << res_post << std::endl;

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

    std::vector<double> residuals() override {
      if constexpr (has_residual<Solver>::value) {
        return solver_.getResiduals();
      } else {
        return std::vector<double>();
      }
    }

  private:
    SystemMatrix system_matrix_;
    Solver solver_;
    Eigen::VectorXx<Scalar> tmp_;
  };
}
