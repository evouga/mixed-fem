#pragma once

#include "mixed_variable.h"
#include "optimizers/optimizer_data.h"
#include "utils/sparse_utils.h"
#include <iostream>
#include <set>
#include "ipc/ipc.hpp"

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

    const Eigen::SparseMatrix<double, Eigen::RowMajor>& lhs() override {
      if (nframes_ == 0) {
        A_ = Eigen::SparseMatrix<double, Eigen::RowMajor>();
        A_.resize(mesh_->jacobian().rows(),mesh_->jacobian().rows());
      }
      return A_;
    }
    void solve(const Eigen::VectorXd& dx) override;

    Eigen::VectorXd& delta() override {
      if (nframes_ == 0) {
        delta_.resize(0);
      }
      return delta_;
    }

    Eigen::VectorXd& value() override {
      if (nframes_ == 0) {
        d_.resize(0);
      }
      return d_;
    }

    Eigen::VectorXd& lambda() override {
      if (nframes_ == 0) {
        la_.resize(0);
      }
      return la_;
    }

    int num_collision_frames() const {
      return nframes_;
    }

    const ipc::Constraints& frames() const {
      return constraints_;
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

    Eigen::MatrixXi F_;
    Eigen::MatrixXi E_;
    Eigen::MatrixXi T_;

    std::shared_ptr<SimConfig> config_;
    std::vector<Eigen::VectorXd> dd_dx_; 
    std::vector<Eigen::MatrixXd> Aloc_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> A_;
    std::shared_ptr<Assembler<double,DIM,-1>> assembler_;
    std::shared_ptr<VecAssembler<double,DIM,-1>> vec_assembler_;

    ipc::CollisionMesh ipc_mesh_;
    ipc::Constraints constraints_;
    std::map<std::array<long, 4>, int> frame_map_;
  };
}
