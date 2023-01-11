#pragma once

#include "external_force.h"

namespace mfem {

  class MechanicalPress : public ExternalForce {
  public:
    MechanicalPress(const Eigen::MatrixXd& V,
        const ExternalForceConfig& config) : ExternalForce(V, config), 
        target_velocity_(config.target_velocity) {
      
      int d = V.cols();
      f_ = target_velocity_;

      // Mark vertices that receive the area force
      is_forced_ = Eigen::VectorXi::Zero(V.rows());
      force_.resize(V.size());
      force_.setZero();

      if (config.is_body_force) {
        is_forced_.setOnes();
        for (int i = 0; i < V.rows(); ++i) {
          force_(d*i + config_.axis) = f_;
        }
        marker_idx_ = 0;
        marker_pos_initial_ = V.row(marker_idx_);
      } else {
        for (int i : groups_[group_id_]) {
          is_forced_(i) = 1;
          force_(d*i + config_.axis) = f_;

          // If we don't have a marker vertex index yet
          // assign it to the first force vertex we find
          if (marker_idx_ == -1) {
            marker_idx_ = i;
            marker_pos_initial_ = V.row(marker_idx_);
          }
        }
      }
      update_force_map();
    }

    bool is_constant() const override final {
      return false;
    }

    void init(Eigen::MatrixXd& V) override final {
      if (marker_idx_ != -1) {
        marker_pos_ = V.row(marker_idx_);
      }
    }

    void step(Eigen::MatrixXd& V, double dt) override final {
      if (marker_idx_ == -1) return;
      Eigen::RowVectorXd new_marker_pos = V.row(marker_idx_);

      const int axis = config_.axis;

      // Displacement rate
      double rate = (new_marker_pos(axis) - marker_pos_(axis)) / dt;

      // Get factor by which we modify the force magnitude
      double factor = std::clamp(target_velocity_/rate, 1.0, 1.1);

      // If we're moving in the wrong direction amp up the factor
      if (target_velocity_/rate < 0) {
        factor = 1.1;
      }

      // Only modify force if magnitude is increasing or if the
      // magnitude is higher than some small value.
      if (factor >= 1.0 || (std::abs(f_) > 0.01)) {
        f_ *= factor;
      }

      bool is_max_magnitude = false;
      // If magnitude of force is too high, clamp it to the maximum allowed
      // ammount.
      if (std::abs(f_) > config_.max_force) {
        is_max_magnitude = true;
        f_ = (f_ > 0) ? config_.max_force : -config_.max_force;
      } 

      std::cout << "Current rate: " << rate << " Target rate: " 
        << target_velocity_ << " Factor: " << factor 
        << " force magnitude: " << f_
        << std::endl;

      // We have reached the desired amount of displacement OR the
      // force magnitude has reached a maximum and the velocity is 0.
      if (std::abs(new_marker_pos(axis) - marker_pos_initial_(axis))
          > config_.max_displacement) {
        f_ = 0;
      }

      std::cout << "1" << std::endl;
      int d = V.cols();
      for (int i = 0; i < V.rows(); ++i) {
        if (is_forced_(i)) {
          force_(d*i + axis) = f_;
        }
      }
      marker_pos_ = new_marker_pos;
    //double max_force = 100;        // absolute maximum force (in Newtons)
    //double target_velocity = 0.1;  // in meters/second
    //double max_displacement = 0.3; // in meters
    }

  private:
    double target_velocity_;
    int group_id_ = 1;
    int marker_idx_ = -1;       // Vertex position we track to measure progress
    double f_ = 0; 
    Eigen::RowVectorXd marker_pos_;
    Eigen::RowVectorXd marker_pos_initial_;
  };
}
