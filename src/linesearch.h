#pragma once

#include <EigenTypes.h>

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
    
  template <typename DerivedX, typename Scalar, class Objective,
           class Callback = decltype(default_linesearch_callback)>
  bool linesearch_backtracking_bisection(
      Eigen::MatrixBase<DerivedX> &x, 
      const Eigen::MatrixBase<DerivedX> &d,
      const Objective &f,
      Eigen::MatrixBase<DerivedX> &g,
      unsigned int max_iterations,
      Scalar alpha,
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

    x += alpha*d;
    printf("  - LS: f(x0): %.5g, f(x + a*d): %.5g, alpha: %.5g\n", fx0, f(x), alpha);
    return (iteration_count == max_iterations); 
    //std::cout << "f(x): " << f(x) << " alpha: " << alpha << std::endl;
    //std::cout << "f(x+alpha*d): " << f(x + alpha * d) << " f(x): " << f(x) << " cg'd: " << alpha * (c * g.transpose()*d) << std::endl;
    //while ((iteration_count < max_iterations) && (f(x + alpha * d) > (f(x) + alpha*c*g.transpose()*d))) {
  }
}