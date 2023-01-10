#pragma once

#include "config.h"
#include "boundary_condition.h"

namespace mfem {

  // Body or Neumann boundary conditions that are fixed within a single timestep.
  // Useful for gravity or simple pushing forces, but not compatible with something like
  // friction.
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

    const Eigen::VectorXi& forced_vertices() const { 
      return is_forced_;
    }

    bool is_constant() const {
      return config_.is_fixed;
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
    Eigen::VectorXd force; // d|V| x 1 - force vector, 'd' is the dimension
  };

  // Constant force applied over the entire object
  class BodyForce : public ExternalForce {
  public:
    BodyForce(const Eigen::MatrixXd& V, const ExternalForceConfig& config)
        : ExternalForce(V, config) {
      int d = V.cols();
      Eigen::VectorXd ext = Eigen::Map<Eigen::VectorXd>(config_.force,
          V.cols());
      force = ext.replicate(V.rows(),1);
    }
  };

  // Uniformly distributed force over an area of the mesh
  class AreaForce: public ExternalForce {
  public:
    AreaForce(const Eigen::MatrixXd& V,
        const Eigen::SparseMatrix<double,Eigen::RowMajor>& M,
        const ExternalForceConfig& config) : ExternalForce(V, config) {
      
      int d = V.cols();
      Eigen::VectorXd f = Eigen::Map<Eigen::VectorXd>(config_.force, V.cols());

      // Mark vertices that receive the area force
      is_forced_ = Eigen::VectorXi::Zero(V.rows());
      if (config.is_body_force) {
        is_forced_.setOnes();
        force = f.replicate(V.rows(),1);
      } else {
        force.resize(V.size());
        for (int i : groups_[group_id_]) {
          is_forced_(i) = 1;
          force.segment(d*i,d) = f;
        }
      }
      update_force_map();



    }

  private:
    int group_id_ = 1;

  };



}