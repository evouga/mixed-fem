#pragma once

#include "linear_solver.h"
// #include <unsupported/Eigen/SparseExtra>

namespace mfem {

  template <typename Solver, typename SystemMatrix, typename Scalar, int DIM,
      StorageType STORAGE = STORAGE_EIGEN>
  class EigenSolver : public LinearSolver<Scalar, DIM, STORAGE> {

    typedef LinearSolver<Scalar, DIM, STORAGE> Base;

  public:

    EigenSolver(SimState<DIM,STORAGE>* state)
        : LinearSolver<Scalar,DIM,STORAGE>(state),
        has_init_(false) {}

    void solve() override {
      system_matrix_.pre_solve(Base::state_);
        // saveMarket(system_matrix_.A(), "lhs.mkt");

      solver_.compute(system_matrix_.A());
      if (solver_.info() != Eigen::Success) {
       std::cerr << "prefactor failed! " << std::endl;
       exit(1);
      }
      tmp_ = solver_.solve(system_matrix_.b());
      std::cout << "rhs norm() : " << system_matrix_.b().norm() << " dx norm(): " << tmp_.norm()  << std::endl;
      system_matrix_.post_solve(Base::state_, tmp_);
    }

  private:
    // Type to represent the left-hand-side system matrix for the linear solve.
    // Different SystemMatrix types vary in their assembly process and matrix
    // structure, necessitating this abstraction.
    SystemMatrix system_matrix_;
    
    Solver solver_;
    Eigen::VectorXx<Scalar> tmp_;
    bool has_init_;

  };


}
