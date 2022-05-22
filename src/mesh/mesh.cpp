#include "mesh.h"
#include "boundary_conditions.h"

using namespace mfem;
using namespace Eigen;


Mesh::Mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    std::shared_ptr<MaterialModel> material,
    std::shared_ptr<MaterialConfig> material_config)
    : V_(V), V0_(V), T_(T), material_(material),
      config_(material_config) {
  is_fixed_.resize(V_.rows());
  is_fixed_.setZero();

  bbox.block(0,0,1,3) = V0_.row(0);
  bbox.block(1,0,1,3) = V0_.row(0);
  for(int i = 1; i < V0_.rows(); i++) {
    const Eigen::RowVector3d& v = V0_.row(i);
    for(int d = 0; d < 3; d++) {
      if(v[d] < bbox(0, d)) {
          bbox(0, d) = v[d];
      }
      if(v[d] > bbox(1, d)) {
          bbox(1, d) = v[d];
      }
    }
  }
  BoundaryConditions<3>::init_boundary_groups(V0_, bc_groups_, 0.1);
}