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
  template <int DIM>
  class NewtonOptimizer : public Optimizer<DIM> {

    typedef Optimizer<DIM> Base;

  public:
    
    NewtonOptimizer(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<SimConfig> config) : Optimizer<DIM>(mesh, config) {}

    static std::string name() {
      return "Newton";
    }

    void reset() override;
    void step() override;
    virtual void update_vertices(const Eigen::MatrixXd& V) override;
    virtual void set_state(const Eigen::VectorXd& x,
        const Eigen::VectorXd& v) override;

  private:

    using Base::mesh_;
    using Base::data_;
    using Base::config_;

    // Simulation substep for this object
    // init_guess - whether to initialize guess with a prefactor solve
    // decrement  - newton decrement norm
    virtual void substep(double& decrement);

    // linear system right hand side
    Eigen::VectorXd rhs_;       

    // linear system left hand side
    Eigen::SparseMatrix<double, Eigen::RowMajor> lhs_;

    double E_prev_; // energy from last result of linesearch
    std::shared_ptr<Displacement<DIM>> xvar_;

    std::shared_ptr<LinearSolver<double, Eigen::RowMajor>> solver_;

  };
}
