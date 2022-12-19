#pragma once

#include "boundary_condition.h"

namespace mfem {

  // Pins both ends along specified axis, and twists along axis
  class TwistAndStretchBC : public BoundaryCondition {
  public:
    TwistAndStretchBC(const Eigen::MatrixXd& V,
        const BoundaryConditionConfig& config)
        : BoundaryCondition(V, config) {}

    void init(Eigen::MatrixXd& V) override {
      is_fixed_ = Eigen::VectorXi::Zero(V.rows());
      group_ang_velocity_.resize(groups_.size());
      group_velocity_.resize(groups_.size());

      Eigen::RowVectorXd bmin = V.colwise().minCoeff();
      Eigen::RowVectorXd bmax = V.colwise().maxCoeff();
      center_ = 0.5 * (bmin + bmax);

      for (size_t i = 0; i < groups_.size(); ++i) {

        for (int j : groups_[i]) {
          is_fixed_(j) = 1;
        }
        Eigen::RowVectorXd vel(V.cols());
        vel.setZero();
        vel(config_.axis) = std::pow(-1.0, i) * -.05;
        group_velocity_[i] = vel;

        group_ang_velocity_[i] = std::pow(-1.0, i) * config_.velocity * M_PI;
      }
      update_free_map();
    }

    void step(Eigen::MatrixXd& V, double dt) override {
      Eigen::MatrixXd dV = Eigen::MatrixXd::Zero(V.rows(), V.cols());
      int dim = V.cols();

      for (size_t i = 0; i < groups_.size(); ++i) {

        Eigen::MatrixXd R;
        double a = group_ang_velocity_[i];

        if (V.cols() == 3) {
          Eigen::Vector3d axis = Eigen::Vector3d::UnitX();
          R = Eigen::AngleAxis<double>(a * dt, axis).toRotationMatrix();
        } else {
          R = Eigen::Rotation2D<double>(a * dt).toRotationMatrix();
        }

        for (int j : groups_[i]) {
          dV.row(j) = (R * (V.row(j) - center_).transpose()).transpose() 
                    + center_ - V.row(j)
                    + group_velocity_[i] * dt;
        }
      }
      V += dV;
    }

  private:
    Eigen::RowVectorXd center_;
    std::vector<double> group_ang_velocity_; // Angular velocities
    std::vector<Eigen::RowVectorXd> group_velocity_;

  }; 

}