#pragma once

#include <EigenTypes.h>

// Utility for workin with corotational
//preconditioned conjugate gradient
template<typename PreconditionerSolver>
inline int pcg(Eigen::VectorXd& x, const Eigen::SparseMatrixd &A,
    const Eigen::VectorXd &b, Eigen::VectorXd &r, Eigen::VectorXd &z,
    Eigen::VectorXd &p, Eigen::VectorXd &Ap, PreconditionerSolver &pre,
    double tol = 1e-4) {

  unsigned int num_itr = 100;
  r = b - A * x;
  z = pre.solve(r);
  p = z;
  double rsold = r.dot(z);
  double rsnew = 0.;
  double alpha = 0.;
  double beta = 0.;

  for(unsigned int i=0; i<num_itr; ++i) {
    Ap = A * p;
    alpha = rsold / (p.dot(Ap));
    x = x + alpha * p;
    r = r - alpha * Ap;
    rsnew = r.dot(r);
    
    if (sqrt(rsnew) < tol) {
      return i; 
    }
    
    z = pre.solve(r);
    
    rsnew = r.dot(z);
    beta = rsnew/rsold;

    p = z + beta * p;
    rsold = rsnew;
  }
  return num_itr;
}

#include <pcg.cpp>
