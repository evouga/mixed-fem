#pragma once

#include "optimizers/mixed_optimizer.h"
#include "sparse_utils.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

#include <Eigen/SVD>
#include <Eigen/QR>

namespace mfem {

  // Mixed FEM Sequential Quadratic Program
  class MixedSQPOptimizer : public MixedOptimizer {
  public:
    
    MixedSQPOptimizer(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<SimConfig> config) : MixedOptimizer(mesh, config) {}

    static std::string name() {
      return "SQP";
    }

    void reset() override;
  
  public:

    virtual double energy(const Eigen::VectorXd& x, const Eigen::VectorXd& s,
        const Eigen::VectorXd& la) override;

    virtual void gradient(Eigen::VectorXd& g, const Eigen::VectorXd& x,
        const Eigen::VectorXd& s, const Eigen::VectorXd& la) override;

    virtual void build_lhs() override;

    virtual void build_rhs() override;

    virtual void update_system() override;

    virtual void substep(int step, double& decrement) override;

    Eigen::SparseMatrixd W_;
    Eigen::SparseMatrixd G_;
    Eigen::SparseMatrixd C_;
    Eigen::SparseMatrixd Gx_;
    Eigen::SparseMatrixd Gx0_;
    Eigen::SparseMatrixd Gs_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> PJ_; // integrated (weighted) jacobian
    Eigen::SparseMatrix<double, Eigen::RowMajor> PM_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> Jw_; // integrated (weighted) jacobian

    // CG temp variables
    Eigen::VectorXd tmp_r_;
    Eigen::VectorXd tmp_z_;
    Eigen::VectorXd tmp_zm1_;
    Eigen::VectorXd tmp_p_;
    Eigen::VectorXd tmp_Ap_;

    std::vector<Eigen::Matrix3f> U_;
    std::vector<Eigen::Matrix3f> V_;
    std::vector<Eigen::Vector3f> sigma_;
    std::vector<Eigen::MatrixXd> Jloc_;
    std::shared_ptr<Assembler<double,3,-1>> assembler_;
    std::shared_ptr<VecAssembler<double,3,4>> vec_assembler_;

    // // Solve used for preconditioner
    // #if defined(SIM_USE_CHOLMOD)
    // Eigen::CholmodSimplicialLDLT<Eigen::SparseMatrixd> solver_;
    // #else
    // Eigen::SimplicialLLT<Eigen::SparseMatrixd> solver_;
    // #endif
    Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver_;

    //Eigen::SimplicialLDLT<Eigen::SparseMatrixd> preconditioner_;
  };
}
