#pragma once

#include "linear_solver.h"
#include <amgcl/backend/eigen.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/value_type/static_matrix.hpp>

namespace mfem {

  template <typename System, int DIM>
  class AMGCLSolver : public LinearSolver<double, DIM> {

    static constexpr int N() {
      return (DIM==3)?6:3;
    }

    typedef LinearSolver<double, DIM> Base;
    //typedef amgcl::backend::eigen<double> Backend;
    typedef amgcl::static_matrix<double, N(), N()> dmat_type;
    typedef amgcl::static_matrix<float, N(), N()> smat_type;
    typedef amgcl::static_matrix<double, N(), 1> dvec_type;

    typedef amgcl::backend::builtin<dmat_type> SBackend;
    typedef amgcl::backend::builtin<smat_type> PBackend;

  public:

    AMGCLSolver(SimState<DIM>* state) : LinearSolver<double,DIM>(state)
    {}

    void solve() override {
      system_.pre_solve(Base::state_);

      typedef amgcl::make_solver<amgcl::amg<SBackend,
          amgcl::coarsening::smoothed_aggregation,
          amgcl::relaxation::spai0>,
          amgcl::solver::cg<SBackend>
          > Solver;

      typename Solver::params prm;
      prm.solver.tol = Base::state_->config_->itr_tol;
      prm.solver.maxiter = Base::state_->config_->max_iterative_solver_iters;
      prm.precond.coarsening.aggr.block_size = N();
      auto A = amgcl::adapter::block_matrix<dmat_type>(system_.A());
      Solver solve(A, prm);
      int rows = system_.A().rows();
      if (tmp_.size() != rows) {
        tmp_.resize(rows);
        tmp_.setZero();
      }
      tmp_.setZero();

      auto x_ptr = reinterpret_cast<dvec_type*>(tmp_.data());
      auto b_ptr = reinterpret_cast<const dvec_type*>(system_.b().data());
      auto x = amgcl::make_iterator_range(x_ptr, x_ptr + rows / N());
      auto b = amgcl::make_iterator_range(b_ptr, b_ptr + rows / N());

      int iters;
      double error;
      std::cout << solve << std::endl;
      //std::tie(iters, error) = solve(system_.b(), tmp_);
      std::tie(iters, error) = solve(A,b,x);
      std::cout << "Iters: " << iters << std::endl
                << "Error: " << error << std::endl;
      system_.post_solve(Base::state_, tmp_);
    }

  private:

    System system_;
    Eigen::VectorXd tmp_;
  };


}
