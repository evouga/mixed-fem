#pragma once

#include "variable.h"
#include "optimizers/optimizer_data.h"
#include "utils/sparse_utils.h"
#include <iostream>
#include "ipc/ipc.hpp"

namespace mfem {

  class SimConfig;

  template<int DIM>
  class Collision : public Variable<DIM> {

    typedef Variable<DIM> Base;

  public:

    Collision(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<SimConfig> config)
        : Variable<DIM>(mesh), config_(config)
    {}

    static std::string name() {
      return "collision";
    }

    double energy(const Eigen::VectorXd& x) override;
    void update(const Eigen::VectorXd& x, double dt) override;
    void reset() override;

    Eigen::VectorXd rhs() override;
    Eigen::VectorXd gradient() override;

    const Eigen::SparseMatrix<double, Eigen::RowMajor>& lhs() override {
      if (nframes_ == 0) {
        A_ = Eigen::SparseMatrix<double, Eigen::RowMajor>();
        A_.resize(mesh_->jacobian().rows(),mesh_->jacobian().rows());
      }
      return A_;
    }

    Eigen::VectorXd& delta() override {
      std::cerr << "stretch::delta() unused" << std::endl;
      return grad_;
    }

    Eigen::VectorXd& value() override {
      std::cerr << "stretch::value() unused" << std::endl;
      return grad_;
    }

    int num_collision_frames() const {
      return nframes_;
    }

    const ipc::Constraints& frames() const {
      return constraints_;
    }

    int size() const override {
      return 0;
    }

    void product_hessian(const Eigen::VectorXd& x,
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
    int nframes_;            // number of elements
    Eigen::VectorXd grad_;   // Gradient with respect to 'd' variables

    std::shared_ptr<SimConfig> config_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> A_;
    std::shared_ptr<Assembler<double,DIM,-1>> assembler_;
    std::shared_ptr<VecAssembler<double,DIM,-1>> vec_assembler_;
    ipc::Constraints constraints_;

  };
}
