#pragma once

#include "mixed_variable.h"
#include "optimizers/optimizer_data.h"
#include "utils/sparse_utils.h"
#include <iostream>
#include <set>

namespace mfem {

  class SimConfig;

  // Make this an abstract class
  // and a folder with all the collision shit
  struct CollisionFrame {

    CollisionFrame(int a, int b, int p) : E_(a,b,p) {
    }

    double is_valid(const Eigen::VectorXd& x) {
      const Eigen::Vector2d& a = x.segment<2>(2*E_(0));
      const Eigen::Vector2d& b = x.segment<2>(2*E_(1));
      const Eigen::Vector2d& p = x.segment<2>(2*E_(2));
      Eigen::Vector2d e = b-a;
      double l = e.norm();
      double proj = (p-a).dot(e/l);
      return (proj > 0 && proj < l);
    }

    double distance(const Eigen::VectorXd& x) {
      const Eigen::Vector2d& a = x.segment<2>(2*E_(0));
      const Eigen::Vector2d& b = x.segment<2>(2*E_(1));
      const Eigen::Vector2d& p = x.segment<2>(2*E_(2));
      Eigen::Vector2d e = b-a;
      Eigen::Vector2d normal(e(1),-e(0));
      normal.normalize();
      return (p-a).dot(normal);
    }

    Eigen::Vector6d gradient(const Eigen::VectorXd& x) {
      // 0 -1  (b-a)  |  C (b-a)
      // 1  0
      //(p-a).dot( C*(b-a) / norm(C*(b-a)))
      // dn/db  =
      // C * (I/ ||l|| - (C*(b-a) * (C*b-a)T) 
      Eigen::Matrix2d C;
      C << 0, 1, -1, 0;
      const Eigen::Vector2d& a = x.segment<2>(2*E_(0));
      const Eigen::Vector2d& b = x.segment<2>(2*E_(1));
      const Eigen::Vector2d& p = x.segment<2>(2*E_(2));
      Eigen::Vector2d e = b-a;
      Eigen::Vector2d normal(e(1),-e(0));
      double l = normal.norm();
      normal /= l;
      Eigen::Vector6d g;
      Eigen::Vector2d tmp = C*(b-a);
      Eigen::Vector2d tmp2 = C*(Eigen::Matrix2d::Identity()/l
          - (tmp*tmp.transpose())/std::pow(l,3))*(p-a);
      g.segment<2>(0) = tmp2 - normal;
      g.segment<2>(2) = -tmp2;
      g.segment<2>(4) = normal;
      return g;
    }

    Eigen::Vector3i E_;
  };

  template<int DIM>
  class Collision : public MixedVariable<DIM> {

    typedef MixedVariable<DIM> Base;

  public:

    Collision(std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
        : MixedVariable<DIM>(mesh), config_(config)
    {}

    static std::string name() {
      return "collision";
    }

    double energy(const Eigen::VectorXd& d) override;
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

    const std::vector<CollisionFrame>& frames() const {
      return collision_frames_;
    }

    // Continuous
    double max_possible_step(const Eigen::VectorXd& x1,
        const Eigen::VectorXd& x2);

  protected:

    void update_rotations(const Eigen::VectorXd& x);
    void update_derivatives(double dt);
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
    double h_;
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
    Eigen::VectorXi C_;

    std::shared_ptr<SimConfig> config_;
    std::map<std::tuple<int,int,int>, int> frame_ids_;
    std::vector<Eigen::VectorXd> dd_dx_; 
    std::vector<Eigen::MatrixXd> Aloc_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> A_;
    std::vector<CollisionFrame> collision_frames_;
    std::shared_ptr<Assembler<double,DIM,-1>> assembler_;
    std::shared_ptr<VecAssembler<double,DIM,-1>> vec_assembler_;
  };


}
