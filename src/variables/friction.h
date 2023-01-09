#pragma once

#include "variable.h"
#include "optimizers/optimizer_data.h"
#include "utils/sparse_utils.h"
#include <iostream>
#include "ipc/ipc.hpp"
#include "ipc/friction/friction.hpp"

namespace mfem {

  class SimConfig;

  template<int DIM>
  class Friction : public Variable<DIM> {

    typedef Variable<DIM> Base;

  public:

    Friction(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<SimConfig> config)
        : Variable<DIM>(mesh), config_(config)
    {}

    static std::string name() {
      return "friction";
    }

    double energy(const Eigen::VectorXd& x) override;
    void update(const Eigen::VectorXd& x, double dt) override;
    void reset() override;
    void pre_solve() override;

    Eigen::VectorXd rhs() override;
    Eigen::VectorXd gradient() override;

    const Eigen::SparseMatrix<double, Eigen::RowMajor>& lhs() override {
      if (constraints_.empty()) {
        A_ = Eigen::SparseMatrix<double, Eigen::RowMajor>();
        A_.resize(mesh_->jacobian().rows(),mesh_->jacobian().rows());
      }
      return A_;
    }

    Eigen::VectorXd& delta() override {
      std::cerr << "friction::delta() unused" << std::endl;
      return grad_;
    }

    Eigen::VectorXd& value() override {
      std::cerr << "friction::value() unused" << std::endl;
      return grad_;
    }

    int size() const override {
      return 0;
    }

    void product_hessian(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const override {}

  protected:

    void update_derivatives(const Eigen::MatrixXd& V, double dt);

  private:

    using Base::mesh_;

    OptimizerData data_;     // Stores timing results
    double dt_;
    Eigen::VectorXd grad_;   // Gradient with respect to 'd' variables
    Eigen::MatrixXd V0_;

    std::shared_ptr<SimConfig> config_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> A_;
    std::shared_ptr<Assembler<double,DIM,-1>> assembler_;
    std::shared_ptr<VecAssembler<double,DIM,-1>> vec_assembler_;
    ipc::FrictionConstraints constraints_;
  };
}
