#pragma once

#include "optimizers/mixed_sqp_optimizer.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

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

    Eigen::VectorXd gl_;
    // Eigen::SimplicialLDLT<Eigen::SparseMatrixdRowMajor> solver_;
    // Solve used for preconditioner
    #if defined(SIM_USE_CHOLMOD)
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrixdRowMajor> solver_;
    #else
    Eigen::SimplicialLLT<Eigen::SparseMatrixdRowMajor> solver_;
    #endif
    Eigen::SimplicialLDLT<Eigen::SparseMatrixdRowMajor> solver_arap_;

    Eigen::Matrix<double, 12,12> pre_affine_;

    Eigen::MatrixXd T0_;
    Eigen::VectorXd Jdx_;
  };
}
