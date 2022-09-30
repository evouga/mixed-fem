#pragma once

#include "config.h"
#include "mesh/mesh.h"

namespace mfem {

  class BoundaryCondition {
  public:
    BoundaryCondition(std::shared_ptr<Mesh> mesh,
        const BoundaryConditionConfig& config) : config_(config) {
      init_boundary_groups(mesh->Vref_, groups_, config.ratio, config.axis);
    }

    virtual ~BoundaryCondition() = default;

    virtual void step(Eigen::MatrixXd& V, double dt) {}

  protected:
    
    void update_free_map();

    static void init_boundary_groups(const Eigen::MatrixXd &V,
      std::vector<std::vector<int>> &bc_groups, double ratio, int axis);

    BoundaryConditionConfig config_;
    Eigen::VectorXi is_fixed_;  // |V| x 1 - {0, 1} to indicate if fixed
    Eigen::VectorXi free_map_;  // |V| x 1 - Index into free vertex list
    std::vector<std::vector<int>> groups_;
  };

}