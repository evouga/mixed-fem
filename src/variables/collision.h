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
    void post_solve() override;

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
      return constraint_set_;
    }

  protected:

    void update_rotations(const Eigen::VectorXd& x);
    void update_derivatives(const Eigen::MatrixXd& V, double dt);
    void update_collision_frames(const Eigen::VectorXd& x);

  private:

    static constexpr int M() {
      return DIM * DIM;
    }

    // Matrix and vector data types
    //using MatD  = Eigen::Matrix<double, DIM, DIM>; // 3x3 or 2x2
    //using VecM  = Eigen::Vector<double, M()>;      // 9x1
    //using MatM  = Eigen::Matrix<double, M(), M()>; // 9x9

    using Base::mesh_;

    OptimizerData data_;     // Stores timing results
    double dt_;
    int nframes_;            // number of elements
    Eigen::VectorXd grad_;   // Gradient with respect to 'd' variables

    Eigen::MatrixXi F_;
    Eigen::VectorXi C_;
    Eigen::MatrixXi E_;

    std::shared_ptr<SimConfig> config_;
    std::vector<Eigen::MatrixXd> Aloc_;
    std::vector<Eigen::VectorXd> gloc_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> A_;
    std::shared_ptr<Assembler<double,DIM,-1>> assembler_;
    std::shared_ptr<VecAssembler<double,DIM,-1>> vec_assembler_;
    ipc::CollisionMesh ipc_mesh_;
    ipc::Constraints constraint_set_;

  };
}
