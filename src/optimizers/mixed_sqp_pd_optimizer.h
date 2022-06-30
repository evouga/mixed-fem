#pragma once

#include "optimizers/mixed_sqp_optimizer.h"
#include "mixed_variables/symmetric_deformation.h"
#include "mixed_variables/displacement.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

namespace mfem {

  // Mixed FEM Sequential Quadratic Program
  class MixedSQPPDOptimizer : public MixedSQPOptimizer {
  public:
    
    MixedSQPPDOptimizer(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<SimConfig> config) : MixedSQPOptimizer(mesh, config) {}

    static std::string name() {
      return "SQP-PD";
    }

  public:

    virtual void gradient(Eigen::VectorXd& g, const Eigen::VectorXd& x,
        const Eigen::VectorXd& s, const Eigen::VectorXd& la) override;

    virtual void step() override;
    virtual void reset() override;

    // Build system left hand side
    virtual void build_lhs() override;

    // Build linear system right hand side
    virtual void build_rhs() override;

    // Update gradients, LHS, RHS for a new configuration
    virtual void update_system() override;


    //TODO change this global solver step + "local_update"
    virtual void substep(int step, double& decrement) override;

    Eigen::VectorXd gl_;

    #if defined(SIM_USE_CHOLMOD)
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver_;
    #else
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver_;
    #endif
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver_arap_;

    Eigen::Matrix<double, 12,12> pre_affine_;

    Eigen::MatrixXd T0_;
    Eigen::VectorXd Jdx_;

    std::shared_ptr<SymmetricDeformation<3>> svar_;
    std::shared_ptr<Displacement<3>> xvar_;
  };
}
