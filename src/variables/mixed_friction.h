#pragma once

#include "mixed_variable.h"
#include "optimizers/optimizer_data.h"
#include "utils/sparse_utils.h"
#include <iostream>
#include "ipc/ipc.hpp"
#include "ipc/friction/friction.hpp"

namespace mfem {

  class SimConfig;

  template<int DIM>
  class MixedFriction : public MixedVariable<DIM> {

    typedef MixedVariable<DIM> Base;

  public:

    MixedFriction(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<SimConfig> config)
        : MixedVariable<DIM>(mesh), config_(config)
    {}

    static std::string name() {
      return "mixed-friction";
    }

    double energy(Eigen::VectorXd& x, Eigen::VectorXd& z) override;
    double constraint_value(const Eigen::VectorXd& x,
        const Eigen::VectorXd& z) override;

    void update(Eigen::VectorXd& x, double dt) override;
    void reset() override;
    void pre_solve() override;
    void solve(Eigen::VectorXd& dx) override;

    Eigen::VectorXd& rhs() override;
    Eigen::VectorXd gradient() override;
    Eigen::VectorXd gradient_mixed() override;
    Eigen::VectorXd gradient_dual() override;
    
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& lhs() override {
      if (constraints_.empty()) {
        A_ = Eigen::SparseMatrix<double, Eigen::RowMajor>();
        A_.resize(mesh_->jacobian().rows(),mesh_->jacobian().rows());
      }
      return A_;
    }

    Eigen::VectorXd& delta() override {
      if (constraints_.empty()) {
        dz_.resize(0);
      }
      return dz_;    }

    Eigen::VectorXd& value() override {
      if (constraints_.empty()) {
        z_.resize(0);
      }
      return z_;
    }

    Eigen::VectorXd& lambda() override {
      if (constraints_.empty()) {
        la_.resize(0);
      }
      return la_;
    }

    int size() const override {
      return z_.size();
    }

    int size_dual() const override {
      return la_.size();
    }

    void product_hessian(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const override {}

  protected:

    void update_derivatives(const Eigen::MatrixXd& V, double dt);

  private:

    using Base::mesh_;

    double dt_;
    Eigen::VectorXd grad_;  // Gradient with respect to 'd' variables
    Eigen::VectorXd rhs_;   // negative Gradient with respect to 'd' variables
    Eigen::MatrixXd V0_;    // Vertex positions at beginning of lagged step
    Eigen::VectorXd z_;     // mixed variable
    Eigen::VectorXd Z_;     // per-frames non-mixed ||u|| values
    Eigen::VectorXd dz_;    // mixed variable delta
    Eigen::VectorXd la_;    // dual variable 

    Eigen::VectorXd g_;      // per-frame gradients
    Eigen::VectorXd H_;      // per-frame hessians
    std::vector<ipc::VectorMax12d> Gx_; // constraint jacobian w.r.t x
    Eigen::VectorXd Gz_;                // constraint jacobian w.r.t z
    Eigen::VectorXd Gdx_;     // tmp var: Jacobian multiplied by dx

    Eigen::MatrixXi T_;
    std::shared_ptr<SimConfig> config_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> A_;
    std::shared_ptr<Assembler<double,DIM,-1>> assembler_;
    std::shared_ptr<VecAssembler<double,DIM,-1>> vec_assembler_;
    ipc::FrictionConstraints constraints_;
  };
}
