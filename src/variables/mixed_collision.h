#pragma once

#include "mixed_variable.h"
#include "optimizers/optimizer_data.h"
#include "utils/sparse_utils.h"
#include <iostream>
#include <set>
#include "ipc/ipc.hpp"
#include "utils/mixed_ipc.h"

namespace mfem {

  class SimConfig;

  template<int DIM>
  class MixedCollision : public MixedVariable<DIM> {

    typedef MixedVariable<DIM> Base;

  public:

    MixedCollision(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<SimConfig> config)
        : MixedVariable<DIM>(mesh), config_(config)
    {}

    static std::string name() {
      return "mixed-collision";
    }

    double energy(const Eigen::VectorXd& x, const Eigen::VectorXd& d) override;
    double constraint_value(const Eigen::VectorXd& x,
        const Eigen::VectorXd& d) override;
    void update(const Eigen::VectorXd& x, double dt) override;
    void reset() override;
    void post_solve() override;

    Eigen::VectorXd rhs() override;
    Eigen::VectorXd gradient() override;
    Eigen::VectorXd gradient_mixed() override;
    Eigen::VectorXd gradient_dual() override {
      Eigen::VectorXd tmp;
      std::cout <<"gradient dual unimplemented for mixed_collision" << std::endl;
      return tmp;
    }

    const Eigen::SparseMatrix<double, Eigen::RowMajor>& lhs() override {
      if (constraints_.empty()) {
        A_ = Eigen::SparseMatrix<double, Eigen::RowMajor>();
        A_.resize(mesh_->jacobian().rows(),mesh_->jacobian().rows());
      }
      return A_;
    }
    void solve(const Eigen::VectorXd& dx) override;

    Eigen::VectorXd& delta() override {
      if (constraints_.empty()) {
        delta_.resize(0);
      }
      return delta_;
    }

    Eigen::VectorXd& value() override {
      if (constraints_.empty()) {
        d_.resize(0);
      }
      return d_;
    }

    Eigen::VectorXd& lambda() override {
      if (constraints_.empty()) {
        la_.resize(0);
      }
      return la_;
    }

    int num_collision_frames() const {
      return constraints_.size();
    }

    const ipc::MixedConstraints& frames() const {
      return constraints_;
    }

    int size() const override {
      return d_.size();
    }

    int size_dual() const override {
      return la_.size();
    }

    void product_hessian(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const override {}
    void product_jacobian_x(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out, bool transposed) const override {}
    void product_jacobian_mixed(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const override {}

  protected:

    void update_derivatives(const Eigen::MatrixXd& V, double dt);

  private:

    static constexpr int M() {
      return DIM * DIM;
    }

    using Base::mesh_;

    OptimizerData data_;     // Stores timing results
    double dt_;
    Eigen::VectorXd D_;      // per-frames distances
    Eigen::VectorXd d_;      // distance variables
    Eigen::VectorXd delta_;  // distance variables deltas
    Eigen::VectorXd la_;     // lagrange multipliers
    Eigen::VectorXd g_;      // per-frame gradients
    Eigen::VectorXd H_;      // per-frame hessians
    Eigen::VectorXd rhs_;    // RHS for schur complement system
    Eigen::VectorXd grad_;   // Gradient with respect to 'd' variables
    Eigen::VectorXd grad_x_; // Gradient with respect to 'x' variables
    Eigen::VectorXd gl_;     // tmp var: g_\Lambda in the notes
    Eigen::VectorXd Gdx_;     // tmp var: Jacobian multiplied by dx

    Eigen::MatrixXi T_;

    std::shared_ptr<SimConfig> config_;
    //std::vector<Eigen::VectorXd> dd_dx_;
    // std::vector<bool> is_mollified_;    // whether constraint is mollified
    std::vector<ipc::VectorMax12d> Gx_; // constraint jacobian w.r.t x
    Eigen::VectorXd Gd_;                // constraint jacobian w.r.t d
    Eigen::SparseMatrix<double, Eigen::RowMajor> A_;
    std::shared_ptr<Assembler<double,DIM,-1>> assembler_;
    std::shared_ptr<VecAssembler<double,DIM,-1>> vec_assembler_;

    ipc::MixedConstraints constraints_;
  };
}
