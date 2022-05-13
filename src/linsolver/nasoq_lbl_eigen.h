//
// Created by kazem on 2020-05-18.
//

#pragma once

#include <nasoq/QP/linear_solver_wrapper.h>

namespace nasoq {
 //  Solving Ax=b
 // Inputs:
 //   H  n by n sparse Hessian matrix **lower triangle only** (see
 //     .triangularView<Eigen::Lower>() )
 //   q  n by 1 vector
 // Outputs:
 //   x  n by 1 solution vector
 // Returns nasoq exit flag
 //
 int linear_solve(
   // Pass inputs by copy so we get non-const and casted data
   Eigen::SparseMatrix<double,Eigen::ColMajor,int> A,
   Eigen::Matrix<double,Eigen::Dynamic,1> b,
   Eigen::Matrix<double,Eigen::Dynamic,1> & x){
  assert(A.isApprox(A.triangularView<Eigen::Lower>(),0) &&
         "P should be lower triangular");
  assert(A.isCompressed());
  assert(A.rows()==A.cols());
  assert(A.rows()==b.rows());

  std::cout << " nas 0" << std::endl;
  CSC *H = new CSC;
  H->nzmax = A.nonZeros();
  H->ncol= H->nrow=A.rows();
  H->p = A.outerIndexPtr();
  H->i = A.innerIndexPtr();
  H->x = A.valuePtr();
  H->stype=-1;
  // TODO: fix this
  H->xtype=CHOLMOD_REAL;
  H->packed=1;
  //H->nz = NULL;
  H->sorted = 0;
  int reg_diag = -9;

  std::cout << " nas 1" << std::endl;
  SolverSettings *lbl = new SolverSettings(H,b.data());
  lbl->ldl_variant = 2;
  lbl->req_ref_iter = 2;
  lbl->solver_mode = 0;
  lbl->reg_diag = pow(10,reg_diag);
  std::cout << " nas 3" << std::endl;
  lbl->symbolic_analysis();
  std::cout << " nas 4" << std::endl;
  lbl->numerical_factorization();
  std::cout << " nas 5" << std::endl;
  double *sol = lbl->solve_only();
  lbl->compute_norms();
  std::cout<<"nasoq residual: "<<lbl->res_l1<<"\n";

  x = Eigen::Map< Eigen::Matrix<double,Eigen::Dynamic,1> >(
    sol,A.rows(),1);

  // Exitflag TODO
  int exitflag = 0;
  delete lbl;
  delete H;
  delete []sol;
  return exitflag;
 }
}

