#pragma once

#include "boundary_condition.h"

namespace mfem {

  // Pin one end along specified axis, and translate these points for an
  // amount of time and velocity
  class TranslateBC : public BoundaryCondition {
  public:
    TranslateBC(const Eigen::MatrixXd& V,
        const BoundaryConditionConfig& config)
        : BoundaryCondition(V, config) {}

    void init(Eigen::MatrixXd& V) override {
      elapsed_time_ = 0.;
      done_ = false;
      has_reversed_ = false;

      is_fixed_ = Eigen::VectorXi::Zero(V.rows());
      group_velocity_.resize(groups_.size());

      for (int j : groups_[group_id_]) {
        is_fixed_(j) = 1;
      }
      Eigen::RowVectorXd vel(V.cols());
      vel.setZero();
      vel(config_.axis) = std::pow(-1.0, group_id_) * config_.velocity;
      group_velocity_[group_id_] = vel;
      update_free_map();
    }

    void step(Eigen::MatrixXd& V, double dt) override {
      if (done_) return;


      Eigen::MatrixXd dV = Eigen::MatrixXd::Zero(V.rows(), V.cols());
      for (int j : groups_[group_id_]) {
        dV.row(j) = group_velocity_[group_id_] * dt; 
      }
      V += dV;

      elapsed_time_ += dt;
      if (elapsed_time_ > config_.duration) {
        if (config_.flip && !has_reversed_) {
          elapsed_time_ = 0.0;
          has_reversed_ = true;
          group_velocity_[group_id_] = -group_velocity_[group_id_];
        } else {
          done_ = true;
        }
      } 

    }

  private:
    std::vector<Eigen::RowVectorXd> group_velocity_;
    int group_id_ = 1;
    bool has_reversed_ = false;
    bool done_ = false;
    double elapsed_time_ = 0.;

  }; 

}