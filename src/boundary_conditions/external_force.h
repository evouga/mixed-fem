#pragma once

#include "config.h"
#include "boundary_condition.h"

namespace mfem {

  // Body or Neumann boundary conditions that are fixed within a timestep.
  // Useful for gravity or simple pushing forces, but not compatible
  // with something like friction.
  class ExternalForce {
  public:

    ExternalForce(const Eigen::MatrixXd& V, const ExternalForceConfig& config)
        : config_(config) {
      BoundaryCondition::init_boundary_groups(V, groups_, config.ratio,
          config.axis);
    }

    virtual ~ExternalForce() = default;

    virtual void init(Eigen::MatrixXd& V) {}
    virtual void step(Eigen::MatrixXd& V, double dt) {}

    const Eigen::VectorXd& force() const {
      return force_;
    }

    const Eigen::VectorXi& forced_vertices() const { 
      return is_forced_;
    }

    virtual bool is_constant() const {
      return true;
    }

  protected:

    void update_force_map()  {
      force_map_.resize(is_forced_.size());
      int curr = 0;
      for (int i = 0; i < is_forced_.size(); ++i) {
        force_map_(i) = is_forced_(i) ? curr++ : -1;
      }  
    }
  
    ExternalForceConfig config_;
    Eigen::VectorXi is_forced_;  // |V| x 1 - {0, 1}
    Eigen::VectorXi force_map_;  // |V| x 1 - Index into forced vertices list
    std::vector<std::vector<int>> groups_;
    Eigen::VectorXd force_; // d|V| x 1 - force vector, 'd' is the dimension
  };

  // Uniformly distributed force over an area of the mesh
  class AreaForce: public ExternalForce {
  public:
    AreaForce(const Eigen::MatrixXd& V,
        const ExternalForceConfig& config) : ExternalForce(V, config) {
      
      int d = V.cols();
      Eigen::VectorXd f = Eigen::Map<Eigen::VectorXd>(config_.force, V.cols());

      // Mark vertices that receive the area force
      is_forced_ = Eigen::VectorXi::Zero(V.rows());
      if (config.is_body_force) {
        is_forced_.setOnes();
        force_ = f.replicate(V.rows(),1);
      } else {
        force_.resize(V.size());
        force_.setZero();
        for (int i : groups_[group_id_]) {
          is_forced_(i) = 1;
          force_.segment(d*i,d) = f;
        }
      }
      update_force_map();
    }

  private:
    int group_id_ = 1;
  };


  class StretchForce : public ExternalForce {
  public:
    StretchForce(const Eigen::MatrixXd& V,
        const ExternalForceConfig& config) : ExternalForce(V, config) {
      
      int d = V.cols();
      Eigen::VectorXd f = Eigen::Map<Eigen::VectorXd>(config_.force, V.cols());

      // Mark vertices that receive the area force
      is_forced_ = Eigen::VectorXi::Zero(V.rows());
      force_.resize(V.size());
      force_.setZero();

      for (size_t i = 0; i < groups_.size(); ++i) {
        for (int j : groups_[i]) {
          is_forced_(j) = 1;
          force_.segment(d*j,d) = std::pow(-1.0,i) * f;
        }
      }
      update_force_map();
    }
  };
}
