#pragma once

#include "optimizers/mixed_sqp_pd_optimizer.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

namespace mfem {

  // Mixed FEM Sequential Quadratic Program
  class MixedSQPBending : public MixedSQPPDOptimizer {
  public:
    
    MixedSQPBending(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<SimConfig> config) : MixedSQPPDOptimizer(mesh, config) {}

    static std::string name() {
      return "SQP-PD";
    }

  public:

    // Build system left hand side
    virtual void build_lhs() override;

    // Build linear system right hand side
    virtual void build_rhs() override;

    virtual void substep(int step, double& decrement) override;

    virtual double energy(const Eigen::VectorXd& x, const Eigen::VectorXd& s,
        const Eigen::VectorXd& a, const Eigen::VectorXd& la,
        const Eigen::VectorXd& ga);
    
    virtual void reset() override;

    virtual void update_system() override;

    void normals(const Eigen::VectorXd& x, Eigen::MatrixXd& n);
    void angles(const Eigen::VectorXd& x, const Eigen::MatrixXd& n,
        Eigen::VectorXd& a);
    void grad_angles(const Eigen::VectorXd& x, const Eigen::MatrixXd& n,
        std::vector<Eigen::VectorXd> da);
      
    Eigen::MatrixXi EV_;
    Eigen::MatrixXi FE_;
    Eigen::MatrixXi EF_;
    Eigen::VectorXd l_;
    Eigen::VectorXd a_;
    Eigen::VectorXd da_;
    Eigen::VectorXd a0_;
    Eigen::VectorXd ga_;
    // Eigen::SparseMatrixdRowMajor Dx_;
    Eigen::SparseMatrixd L_;
    Eigen::MatrixXd n_;
    int nedges_;

    std::vector<double> Ha_;
    std::vector<double> Ha_inv_;
    std::vector<double> grad_a_;
    Eigen::VectorXd gg_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> Dx_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> PDW_;



  };
}
