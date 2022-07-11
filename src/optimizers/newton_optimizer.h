#pragma once

#include "optimizers/optimizer.h"
#include "variables/displacement.h"
#include "sparse_utils.h"
#include "linear_solvers/linear_solver.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

namespace mfem {

  // Standard linear FEM 
  class NewtonOptimizer : public Optimizer {
  public:
    
    NewtonOptimizer(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<SimConfig> config) : Optimizer(mesh, config) {}

    static std::string name() {
      return "Newton";
    }

    void reset() override;
    void step() override;
    virtual void update_vertices(const Eigen::MatrixXd& V) override;
    virtual void set_state(const Eigen::VectorXd& x,
        const Eigen::VectorXd& v) override;

  public:

    // Simulation substep for this object
    // init_guess - whether to initialize guess with a prefactor solve
    // decrement  - newton decrement norm
    virtual void substep(double& decrement);

    Eigen::VectorXd rhs_;       // linear system right hand side
    Eigen::SparseMatrix<double, Eigen::RowMajor> lhs_;  // linear system left hand side

    double E_prev_; // energy from last result of linesearch
    std::shared_ptr<Displacement<3>> xvar_;

    std::shared_ptr<LinearSolver<double, Eigen::RowMajor>> solver_;

  };
}
