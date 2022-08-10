#pragma once

#include <EigenTypes.h>
#include "variables/mixed_variable.h"
#include "variables/displacement.h"
#include <iomanip>

namespace {
  // Taken from https://github.com/mattoverby/mcloptlib
  template <typename Scalar>
  static inline Scalar range( Scalar alpha, Scalar low, Scalar high ){
    if( alpha < low ){ return low; }
    else if( alpha > high ){ return high; }
    return alpha;
  }

  // Cubic interopolation to find minimum alpha along an interval
  //    Cubic has form g(alpha) = a alpha^3 + b alpha^2 + f'(x0) alpha + f(x0)
  //    Solve for [a b] then use quadratic equation to find the minimum.
  //
  // f(x) = f(x + alpha*d) where d is the descent direction
  //
  // Params
  //    fx0   - f(x)
  //    gTd   - f'(x0)^T d
  //    fx1   - f(x + a1*d)
  //    fx2   - f(x + a2*d)
  //    a1    - newest step size in interval (alpha_1)
  //    a2    - previous step size in interval (alpha_2)
  template <typename Scalar>
  static inline Scalar cubic2(Scalar fx0, Scalar gTd, Scalar fx1, Scalar fx2,
      Scalar a1, Scalar a2){

    // Determinant of coefficient fitting matrix
    Scalar det = 1.0 / (a1*a1*a2*a2*(a1-a2));
    
    // Adjugate matrix (for 2x2 inverse)
    Eigen::Matrix<Scalar,2,2> A;
    A <<     a2*a2,   -a1*a1,
         -a2*a2*a2, a1*a1*a1;

    // Right hand side of interpolation system 
    Eigen::Matrix<Scalar,2,1> B;
    B(0) = fx1 - a1*gTd - fx0;
    B(1) = fx2 - a2*gTd - fx1;
    
    // Solve cubic coefficients c = [a b]
    Eigen::Matrix<Scalar,2,1> c = det * A * B; 

    // If the cubic coefficient is 0, use solution for quadratic
    // i.e. g(alpha) = b * alpha^2 + f'(x0) * alpha + f(x0)
    //      g'(alpha) = (2b * alpha + f'(x0))
    //      Set equal to zero and solve for alpha.
    if (std::abs(c(0)) < 1e-12) {
      return -gTd / (2.0*c(1));
    } else {
      // Otherwise use quadratic formula to find the minimum of the cubic
      Scalar d = std::sqrt(c(1)*c(1) - 3*c(0)*gTd);
      return (-c(1) + d) / (3.0 * c(0));
    }
  }
  //
  // Cubic interpolation
  // fx0 = f(x0)
  // gtp = f'(x0)^T p
  // fxa = f(x0 + alpha*p)
  // alpha = step length
  // fxp = previous fxa
  // alphap = previous alpha
  template <typename Scalar>
  static inline Scalar cubic( Scalar fx0, Scalar gTd, Scalar fx, Scalar alpha, Scalar fxp, Scalar alphap ){
    typedef Eigen::Matrix<Scalar,2,1> Vec2;
    typedef Eigen::Matrix<Scalar,2,2> Mat2;

    Scalar mult = 1.0 / ( alpha*alpha * alphap*alphap * (alpha-alphap) );
    Mat2 A;
    A(0,0) = alphap*alphap;   A(0,1) = -alpha*alpha;
    A(1,0) = -alphap*alphap*alphap; A(1,1) = alpha*alpha*alpha; 
    Vec2 B;
    B[0] = fx - fx0 - alpha*gTd; B[1] = fxp - fx0 - alphap*gTd;
    Vec2 r = mult * A * B;
    if( std::abs(r[0]) <= 0.0 ){ return -gTd / (2.0*r[1]); } // if quadratic
    Scalar d = std::sqrt( r[1]*r[1] - 3.0*r[0]*gTd ); // discrim
    return (-r[1] + d) / (3.0*r[0]);
  }
}

//Implementation of simple backtracking linesearch using bisection 
//Input:
//  x - initial point
//  d -  search direction
//  f - the optimization objective function f(x)
//  g - gradient of function at initial point
//  max_iterations - max line search iterations
//  alpha - max proportion of line search
//  c- sufficient decrease parameter
//  p - bisection ratio
//  Callback - callback that executes each line search iteration
//Output:
//  x - contains point to which linesearch converged, or final accepted point before termination
//  SolverExitStatus - exit code for solver 
namespace mfem {
  enum SolverExitStatus {
      CONVERGED,
      MAX_ITERATIONS_REACHED
  };
    
  // x     - Displacement variable
  // vars  - Mixed variables
  // alpha - step size (modified by function)
  // c     - sufficient decrease factor for armijo rule
  // p     - factor by which alpha is decreased
  // func  - callback function
  template <int DIM, typename Scalar,
            class Callback = decltype(default_linesearch_callback)>
  SolverExitStatus linesearch_backtracking_cubic(
      std::shared_ptr<Displacement<DIM>> x,
      std::vector<std::shared_ptr<MixedVariable<DIM>>> vars,
      Scalar& alpha, unsigned int max_iterations, Scalar c=1e-4, Scalar p=0.5,
      const Callback func = default_linesearch_callback) {

    double h2 = std::pow(x->integrator()->dt(),2);

    auto f = [=](double a)->Scalar {
      Eigen::VectorXd x0 = x->value() + a * x->delta();
      Scalar val = x->energy(x0);
      x->unproject(x0);
      for (int i = 0; i < vars.size(); ++i) {
        const Eigen::VectorXd si = vars[i]->value() + a * vars[i]->delta();
        val += h2*vars[i]->energy(si) - vars[i]->constraint_value(x0, si);  
      }
      return val;
    };

    // Compute gradient dot descent direction
    Scalar gTd = x->gradient().dot(x->delta());
    for (int i = 0; i < vars.size(); ++i) {
      gTd += vars[i]->gradient().dot(x->delta())
        + vars[i]->gradient_mixed().dot(vars[i]->delta());
    }

    Scalar fx0 = f(0);
    Scalar fx_prev = fx0;
    Scalar alpha_prev = alpha;

    int iter = 0;
    while (iter < max_iterations) {

      //func(x + alpha*d); // callback

      Scalar fx = f(alpha);

      // Armijo sufficient decrease condition
      Scalar fxn = fx0 + (alpha * c) * gTd;
      if (fx < fxn) {
        break;
      }

      Scalar alpha_tmp = (iter == 0) ? (gTd / (2.0 * (fx0 + gTd - fx)))
          : cubic(fx0, gTd, fx, alpha, fx_prev, alpha_prev);
      fx_prev = fx;
      alpha_prev = alpha;
      alpha = range(alpha_tmp, 0.1*alpha, 0.5*alpha );
      ++iter;
    }

    if(iter < max_iterations) {
      x->value() += alpha * x->delta();
      for (int i = 0; i < vars.size(); ++i) {
        vars[i]->value() += alpha * vars[i]->delta();
      }
    }
    return (iter == max_iterations 
        ? SolverExitStatus::MAX_ITERATIONS_REACHED 
        : SolverExitStatus::CONVERGED);
  }

  template <int DIM, typename Scalar,
            class Callback = decltype(default_linesearch_callback)>
  SolverExitStatus linesearch_backtracking(const SimState<DIM>& state,
      Scalar& alpha, Scalar c=1e-4, Scalar p=0.5,
      const Callback func = default_linesearch_callback) {
    
    unsigned int max_iterations = state.config_->ls_iters;

    double h2 = std::pow(state.x_->integrator()->dt(), 2);

    auto f = [&](double a)->Scalar {
      Eigen::VectorXd x0 = state.x_->value() + a * state.x_->delta();
      Scalar val = state.x_->energy(x0);
      state.x_->unproject(x0);
      for (const auto& var : state.mixed_vars_) {
        const Eigen::VectorXd si = var->value() + a * var->delta();
        val += h2 * var->energy(x0, si) - var->constraint_value(x0, si);  
      }
      for (const auto& var : state.vars_) {
        val += h2 * var->energy(x0);  
      }
      return val;
    };

    // Compute gradient dot descent direction
    Scalar gTd = state.x_->gradient().dot(state.x_->delta());
    for (const auto& var : state.mixed_vars_) {
      gTd += var->gradient().dot(state.x_->delta())
           + var->gradient_mixed().dot(var->delta());
    }
    for (const auto& var : state.vars_) {
      gTd += var->gradient().dot(state.x_->delta());
    }

    Scalar fx0 = f(0);
    Scalar fx_prev = fx0;
    Scalar alpha_prev = alpha;

    //std::cout << "LS: " << fx0 << std::endl;
    alpha *= 0.9;

    int iter = 0;
    while (iter < max_iterations) {

      Scalar fx = f(alpha);

      // Armijo sufficient decrease condition
      Scalar fxn = fx0 + (alpha * c) * gTd;
      //std::cout << std::setprecision(10) << "LS: i: " << iter << " f: " << fx << std::endl;
      if (fx < fxn) {
        break;
      }

      alpha = alpha * p;
      ++iter;
    }

    if(iter < max_iterations) {
      state.x_->value() += alpha * state.x_->delta();
      for (auto& var : state.mixed_vars_) {
        var->value() += alpha * var->delta();
      }
    }
    return (iter == max_iterations 
        ? SolverExitStatus::MAX_ITERATIONS_REACHED 
        : SolverExitStatus::CONVERGED);
  }
}
