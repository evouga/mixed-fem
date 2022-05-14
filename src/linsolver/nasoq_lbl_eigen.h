//
// Created by kazem on 2020-05-18.
//

#pragma once

#include <nasoq/QP/linear_solver_wrapper.h>

// 1. NH in WRAPD
// 2. norm(dx;ds) for termination criteria?


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
 template<typename Scalar>
 int linear_solve(
   // Pass inputs by copy so we get non-const and casted data
   Eigen::SparseMatrix<Scalar,Eigen::ColMajor,int> A,
   Eigen::Matrix<Scalar,Eigen::Dynamic,1> b,
   Eigen::Matrix<Scalar,Eigen::Dynamic,1> & x){
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
  H->nz = NULL;
  H->sorted = 0; // TODO switch that back eigen does sort it u dummy
  int reg_diag = -9;

  SolverSettings *lbl = new SolverSettings(H,b.data());
  lbl->ldl_variant = 4;
  lbl->req_ref_iter = 10;
  lbl->solver_mode = 0;
  lbl->reg_diag = pow(10,reg_diag);
  lbl->symbolic_analysis();
  lbl->numerical_factorization();
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

