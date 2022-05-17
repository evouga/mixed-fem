#pragma once

#include "optimizers/mixed_sqp_optimizer.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

#include <amgcl/backend/eigen.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/idrs.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/spai1.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/relaxation/ilut.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/relaxation/chebyshev.hpp>
#include <amgcl/solver/gmres.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/fgmres.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/preconditioner/schur_pressure_correction.hpp>
#include <amgcl/solver/preonly.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/make_block_solver.hpp>
#include <amgcl/profiler.hpp>
#include <amgcl/coarsening/aggregation.hpp>

namespace mfem {

  // Mixed FEM Sequential Quadratic Program
  class MixedSQPROptimizer : public MixedSQPOptimizer {
  public:
    
    MixedSQPROptimizer(std::shared_ptr<SimObject> object,
        std::shared_ptr<SimConfig> config) : MixedSQPOptimizer(object, config) {}

  public:

    virtual void step() override;
    virtual void reset() override;

    // Build system left hand side
    virtual void build_lhs() override;

    // Build linear system right hand side
    virtual void build_rhs() override;

    // Update gradients, LHS, RHS for a new configuration
    virtual void update_system() override;

    virtual void substep(int step, double& decrement) override;

    virtual bool linesearch_x(Eigen::VectorXd& x,
        const Eigen::VectorXd& dx) override;
    virtual bool linesearch_s_local(Eigen::VectorXd& s,
        const Eigen::VectorXd& ds) override;

    virtual double energy(const Eigen::VectorXd& x,
        const Eigen::VectorXd& s, const Eigen::VectorXd& la) override;

    Eigen::VectorXd gl_;
    double constraint_l1_;
    double mu_;
    double step_;
    Eigen::VectorXd mu_vec_;


    // Eigen::SimplicialLDLT<Eigen::SparseMatrixdRowMajor> solver_;
    // Solve used for preconditioner
    #if defined(SIM_USE_CHOLMOD)
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrixdRowMajor> solver_;
    #else
    Eigen::SimplicialLLT<Eigen::SparseMatrixdRowMajor> solver_;
    #endif
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrixdRowMajor> solver_arap_;

    typedef amgcl::backend::eigen<double> Backend;

    // typedef amgcl::make_solver<
    //     amgcl::relaxation::as_preconditioner<Backend, amgcl::relaxation::ilu0>,
    //     amgcl::solver::bicgstab<Backend>
    // > Solver;
    typedef amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >,
        amgcl::solver::bicgstab<Backend>
        > Solver;

    std::shared_ptr<Solver> amg_solver_;
  };
}
