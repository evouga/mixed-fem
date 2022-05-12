#pragma once

#include <EigenTypes.h>
#include <Eigen/SVD>
#include <Eigen/QR>

// Utility for workin with corotational
//preconditioned conjugate gradient
template<typename PreconditionerSolver>
inline int pcg(Eigen::VectorXd& x, const Eigen::SparseMatrixd &A,
    const Eigen::VectorXd &b, Eigen::VectorXd &r, Eigen::VectorXd &z,
    Eigen::VectorXd &p, Eigen::VectorXd &Ap, PreconditionerSolver &pre,
    double tol = 1e-4) {

  unsigned int num_itr = 20;
  r = b - A * x;
  z = pre.solve(r);
  p = z;
  double rsold = r.dot(z);
  double rsnew = 0.;
  double alpha = 0.;
  double beta = 0.;

  Eigen::VectorXd zm1;

  for(unsigned int i=0; i<num_itr; ++i) {
    Ap = A * p;
    alpha = rsold / (p.dot(Ap));
    x = x + alpha * p;
    r = r - alpha * Ap;
    rsnew = r.dot(r);
    
    if (rsnew < tol) {
      return i; 
    }
    
    zm1 = z;
    z = pre.solve(r);
    
    rsnew = r.dot(z-zm1);
    beta = rsnew/rsold;

    p = z + beta * p;
    rsold = r.dot(z);
  }
  return num_itr;
}

template<typename PreconditionerSolver>
inline int corot_pcg(Eigen::VectorXd& x, const Eigen::SparseMatrixd &A,
    const Eigen::VectorXd &b, Eigen::VectorXd &r, Eigen::VectorXd &z,
    Eigen::VectorXd &p, Eigen::VectorXd &Ap, std::vector<Eigen::Matrix<double, 9,6> > & R,
    PreconditionerSolver &pre,
    double tol = 1e-4) {

  unsigned int num_itr = 200;
  r = b - A * x;
  z = pre.solve(r);
  /*int n = x.size() - 6 * R.size();
  Eigen::VectorXd rtilde(n + 9 * R.size());
  rtilde.segment(0, n) = r.segment(0, n);
  #pragma omp parallel for
  for (int i = 0; i < R.size(); ++i) {

    //Eigen::JacobiSVD<Eigen::Matrix<double, 6,9>> svd(R[i].transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Vector9d rl = R[i]*r.segment<6>(n + 6*i);
    /*Eigen::Matrix3d RL;
    RL << rl(0), rl(4), rl(7),
          rl(2), rl(5), rl(8),
          rl(3), rl(6), rl(9);
    //RL = RL;
    //RL.transposeInPlace();*/
    //rtilde.segment<9>(n + 9*i) = rl;
  /*}

  Eigen::VectorXd ztilde = pre.solve(rtilde);
  z.resize(x.size());
  z.segment(0, n) = ztilde.segment(0, n);
  #pragma omp parallel for
  for (int i = 0; i < R.size(); ++i) {
    
    //Eigen::JacobiSVD<Eigen::Matrix<double, 9,6>> svd(R[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
  //Eigen::CompleteOrthogonalDecomposition<Eigen::Matrix<double, 9,6>> qr(R[i]);
    // S(x^k)
    //Eigen::Matrix3d S = 0.5*(A.transpose() + A); //svd.matrixV() * svd.singularValues().asDiagonal() 
       // * svd.matrixV().transpose();
    z.segment<6>(n + 6*i) = R[i].transpose()*ztilde.segment<9>(n + 9*i);
  }*/

  //z = r;
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
    
    //std::cout<<"RSNEW: "<<rsnew<<"\n";
    if (sqrt(rsnew)/b.norm() < tol) {
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
inline int pcr(Eigen::VectorXd& x, unsigned int nd, unsigned int ne, const Eigen::SparseMatrixd &A,
    const Eigen::VectorXd &b, Eigen::VectorXd &r, Eigen::VectorXd &z,
    Eigen::VectorXd &p, Eigen::VectorXd &Ap,
    PreconditionerSolver &pre,
    double tol = 1e-4) {

      Eigen::VectorXd Ar;
      Eigen::VectorXd Api;

      r = b-A*x;
      r = pre.solve(r);
      p = r;
      Ap = A*p;
      Ar = A*r;
      unsigned int num_itr = 500;
      double rold;

      for(unsigned int i=0; i<num_itr; ++i) {

        rold = r.dot(Ar);

        Api = pre.solve(Ap);
        double alpha = rold/(Ap.dot(Api));

        x = x + alpha*p;

        
        r = r - alpha*Api;
        
        if ((b-A*x).squaredNorm() < (tol*b.squaredNorm() + tol)) {
          return i;
        }

        Ar = A*r;
        double beta = r.dot(Ar)/rold;

        p = r + beta*p;
        
        Ap = Ar + beta*Ap;

      }

}
#include <pcg.cpp>




