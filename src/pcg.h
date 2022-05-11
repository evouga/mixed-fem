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

template<typename PreconditionerSolver>
inline int corot_pcg(Eigen::VectorXd& x, const Eigen::SparseMatrixd &A,
    const Eigen::VectorXd &b, Eigen::VectorXd &r, Eigen::VectorXd &z,
    Eigen::VectorXd &p, Eigen::VectorXd &Ap, std::vector<Eigen::Matrix3d>& R,
    PreconditionerSolver &pre,
    double tol = 1e-4) {

  unsigned int num_itr = 100;
  r = b - A * x;

  int n = x.size() - 6 * R.size();
  Eigen::VectorXd rtilde(n + 9 * R.size());
  rtilde.segment(0, n) = r.segment(0, n);
  #pragma omp parallel for
  for (int i = 0; i < R.size(); ++i) {
    Eigen::Vector6d rl = r.segment<6>(n + 6*i);
    Eigen::Matrix3d RL;
    RL << rl(0), rl(3), rl(4),
          rl(3), rl(1), rl(5),
          rl(4), rl(5), rl(2);
    RL = R[i] * RL;
    rtilde.segment<9>(n + 9*i) = Eigen::Vector9d(RL.data());
  }
  Eigen::VectorXd ztilde = pre.solve(rtilde);
  z.resize(x.size());
  z.segment(0, n) = ztilde.segment(0, n);
  #pragma omp parallel for
  for (int i = 0; i < R.size(); ++i) {
    Eigen::Vector6d rl = r.segment<6>(n + 6*i);
    Eigen::Matrix3d A = R[i].transpose() * Eigen::Map<Eigen::Matrix3d>(
        ztilde.segment<9>(n + 9*i).data());
    Eigen::Matrix3d S = 0.5*(A + A.transpose());
    Eigen::Vector6d s;
    s << S(0,0), S(1,1), S(2,2), S(1,0), S(2,0), S(2,1);
    z.segment<6>(n + 6*i) = s;
  }

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
    
    rtilde.segment(0, n) = r.segment(0, n);
    #pragma omp parallel for
    for (int i = 0; i < R.size(); ++i) {
      Eigen::Vector6d rl = r.segment<6>(n + 6*i);
      Eigen::Matrix3d RL;
      RL << rl(0), rl(3), rl(4),
            rl(3), rl(1), rl(5),
            rl(4), rl(5), rl(2);
      RL = R[i] * RL;
      rtilde.segment<9>(n + 9*i) = Eigen::Vector9d(RL.data());
    }
    Eigen::VectorXd ztilde = pre.solve(rtilde);
    z.resize(x.size());
    z.segment(0, n) = ztilde.segment(0, n);
    #pragma omp parallel for
    for (int i = 0; i < R.size(); ++i) {
      Eigen::Vector6d rl = r.segment<6>(n + 6*i);
      Eigen::Matrix3d A = R[i].transpose() * Eigen::Map<Eigen::Matrix3d>(
          ztilde.segment<9>(n + 9*i).data());
      Eigen::Matrix3d S = 0.5*(A + A.transpose());
      Eigen::Vector6d s;
      s << S(0,0), S(1,1), S(2,2), S(1,0), S(2,0), S(2,1);
      z.segment<6>(n + 6*i) = s;
    }
  //  z = pre.solve(r);
    
    rsnew = r.dot(z);
    beta = rsnew/rsold;

    p = z + beta * p;
    rsold = rsnew;
  }
  return num_itr;
}
#include <pcg.cpp>
