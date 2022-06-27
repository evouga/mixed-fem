#pragma once

#include <EigenTypes.h>
#include "mixed_variables/mixed_variable.h"

namespace {
  // Taken from https://github.com/mattoverby/mcloptlib
  template <typename Scalar>
  static inline Scalar range( Scalar alpha, Scalar low, Scalar high ){
    if( alpha < low ){ return low; }
    else if( alpha > high ){ return high; }
    return alpha;
  }

  // Cubic interpolation
  // fx0 = f(x0)
  // gtp = f'(x0)^T p
  // fxa = f(x0 + alpha*p)
  // alpha = step length
  // fxp = previous fxa
  // alphap = previous alpha
  template <typename Scalar>
  static inline Scalar cubic( Scalar fx0, Scalar gtp, Scalar fxa, Scalar alpha, Scalar fxp, Scalar alphap ){
    typedef Eigen::Matrix<Scalar,2,1> Vec2;
    typedef Eigen::Matrix<Scalar,2,2> Mat2;

    Scalar mult = 1.0 / ( alpha*alpha * alphap*alphap * (alpha-alphap) );
    Mat2 A;
    A(0,0) = alphap*alphap;		A(0,1) = -alpha*alpha;
    A(1,0) = -alphap*alphap*alphap;	A(1,1) = alpha*alpha*alpha;	
    Vec2 B;
    B[0] = fxa - fx0 - alpha*gtp; B[1] = fxp - fx0 - alphap*gtp;
    Vec2 r = mult * A * B;
    if( std::abs(r[0]) <= 0.0 ){ return -gtp / (2.0*r[1]); } // if quadratic
    Scalar d = std::sqrt( r[1]*r[1] - 3.0*r[0]*gtp ); // discrim
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
    
  template <typename DerivedX, typename Scalar, class Objective,
           class Callback = decltype(default_linesearch_callback)>
  SolverExitStatus linesearch_backtracking_bisection(
      Eigen::MatrixBase<DerivedX> &x, 
      const Eigen::MatrixBase<DerivedX> &d,
      const Objective &f,
      Eigen::MatrixBase<DerivedX> &g,
      Scalar& alpha,
      unsigned int max_iterations,
      Scalar c,
      Scalar p,
      Scalar fx0,
      const Callback func = default_linesearch_callback) {

    unsigned int iteration_count = 0;
    fx0 = f(x);
    
    bool done = false;
    while (iteration_count < max_iterations && !done) {
      // f(x+alpha*d) may expect a modified simulation state,
      // so need to execute callback first
      //std::cout << "Alpha: " << alpha << std::endl;
      func(x + alpha*d);
      if (f(x + alpha*d) > fx0) {
        alpha  *= p; 
        iteration_count++;
      } else {
        done = true;
      }
    }

    if(iteration_count < max_iterations) {
      x += alpha*d;
    }
    // printf("  - LS: f(x0): %.5g, f(x + a*d): %.5g, alpha: %.5g\n", fx0, f(x), alpha);
    return (iteration_count == max_iterations 
        ? SolverExitStatus::MAX_ITERATIONS_REACHED 
        : SolverExitStatus::CONVERGED);
    //std::cout << "f(x): " << f(x) << " alpha: " << alpha << std::endl;
    //std::cout << "f(x+alpha*d): " << f(x + alpha * d) << " f(x): " << f(x) << " cg'd: " << alpha * (c * g.transpose()*d) << std::endl;
    //while ((iteration_count < max_iterations) && (f(x + alpha * d) > (f(x) + alpha*c*g.transpose()*d))) {
  }

  // Modified from https://github.com/mattoverby/mcloptlib/blob/master/include/MCL/Backtracking.hpp
  template <typename DerivedX, typename Scalar, class Objective,
            class Callback = decltype(default_linesearch_callback)>
  SolverExitStatus linesearch_backtracking_cubic(
      Eigen::MatrixBase<DerivedX> &x,       // x_n
      const Eigen::MatrixBase<DerivedX> &d, // proposed descent direction
      const Objective &f,                   // energy function
      Eigen::MatrixBase<DerivedX> &g,       // gradient at f(x_n)
      Scalar& alpha,                        // initial step size
      unsigned int max_iterations,          // maximum linesearch iters
      Scalar c,                             // sufficient decrease factor 1e-4 
      Scalar p,                             // factor to reduce alpha by
      Scalar fx0,                           // dummy ignore this shit
      const Callback func = default_linesearch_callback) {
      
    fx0 = f(x);
    // grad = g;
    Scalar gTp = g.dot(d);
    Scalar fx_prev = fx0;
    Scalar alpha_prev = alpha;

    int iter = 0;
    while (iter < max_iterations) {

      func(x + alpha*d); // callback

      Scalar fx = f(x + alpha*d);

      // Armijo sufficient decrease condition
      Scalar fxn = fx0 + (alpha * c) * gTp;
      if (fx < fxn) {
        break;
      }

      Scalar alpha_tmp = (iter == 0) ? (gTp / (2.0 * (fx0 + gTp - fx)))
          : cubic(fx0, gTp, fx, alpha, fx_prev, alpha_prev);
      fx_prev = fx;
      alpha_prev = alpha;
      alpha = range(alpha_tmp, 0.1*alpha, 0.5*alpha );
      ++iter;
    }

    if(iter < max_iterations) {
      x += alpha*d;
    }
      // printf("  - LS: f(x0): %.5g, f(x + a*d): %.5g, alpha: %.5g\n", fx0, f(x), alpha);
    return (iter == max_iterations 
        ? SolverExitStatus::MAX_ITERATIONS_REACHED 
        : SolverExitStatus::CONVERGED);
  }

  // x     - Displacement variable
  // vars  - Mixed variables
  // alpha - step size (modified by function)
  // c     - sufficient decrease factor for armijo rule
  // p     - factor by which alpha is decreased
  // func  - callback function
  template <int DIM, typename DerivedX, typename Scalar, class Objective,
            class Callback = decltype(default_linesearch_callback)>
  SolverExitStatus linesearch_backtracking_cubic(
      std::shared_ptr<MixedVariable<DIM>> x,
      std::vector<std::shared_ptr<MixedVariable<DIM>>> vars,
      Scalar& alpha, unsigned int max_iterations, Scalar c, Scalar p,  
      const Callback func = default_linesearch_callback) {

    auto f = [=](double a)->Scalar {
      const Eigen::VectorXd x0 = x->value() + a * x->delta();
      Scalar val = x->energy(x0);
      for (int i = 0; i < vars.size(); ++i) {
        const Eigen::VectorXd si = vars[i]->value() + a * vars[i]->delta();
        val += vars->energy(si) + vars->constraint_value(x0, si);  
      }
      return val;
    };

    // Compute gradient dot descent direction
    Scalar gTd = x->gradient().dot(x->delta());
    for (int i = 0; i < vars.size(); ++i) {
      gTd += vars[i]->gradient().dot(vars[i]->delta());
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
}


