#pragma once

#include <EigenTypes.h>
#include <memory>
#include "mesh/mesh.h"
#include "config.h"
#include "optimizer_data.h"
#include "boundary_conditions.h"

namespace mfem {

  static Eigen::Vector6d I_vec = (Eigen::Vector6d() <<
      1, 1, 1, 0, 0, 0).finished();

  static Eigen::Matrix6d Sym = (Eigen::Vector6d() <<
      1, 1, 1, 2, 2, 2).finished().asDiagonal();

  static Eigen::Matrix6d Syminv = (Eigen::Vector6d() <<
    1, 1, 1, .5, .5, .5).finished().asDiagonal();
    
  class Optimizer {
  public:
    Optimizer(std::shared_ptr<Mesh> object,
          std::shared_ptr<SimConfig> config)
          : object_(object), config_(config) {}
          
    static std::string name() {
      return "base";
    }

    virtual void reset();
    virtual void step() = 0;

    virtual void update_vertices(const Eigen::MatrixXd& V) {
      std::cerr << "Update vertices not implemented!" << std::endl;
    }
    virtual void set_state(const Eigen::VectorXd& x,
        const Eigen::VectorXd& v) {
      std::cerr << "Update state not implemented!" << std::endl;
    }
    
    // Temporary. Should be a part of a callback function instead.
    // Used to save per substep vertices;
    std::vector<Eigen::MatrixXd> step_x;
    Eigen::VectorXd step_x0;
    Eigen::VectorXd step_v;
    Eigen::SparseMatrixd P_;          // pinning constraint (for vertices)

    double wx_, wx0_, wx1_, wx2_, wdt_;
    
  protected:

    OptimizerData data_;
    std::shared_ptr<Mesh> object_;
    std::shared_ptr<SimConfig> config_;

    // Debug timing variables (timings in milliseconds)
    std::map<std::string, double> timings;

    BoundaryConditions<3> BCs_;

    // Eigen::SparseMatrixd P_;          // pinning constraint (for vertices)
    int nelem_;             // number of elements

  };        
  
}
