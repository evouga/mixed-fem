#include "optimizer.h"
#include "pinning_matrix.h"
#include "mesh/mesh.h"

using namespace mfem;
using namespace Eigen;


void Optimizer::reset() {
  nelem_ = mesh_->T_.rows();
  mesh_->V_ = mesh_->V0_;
  mesh_->clear_fixed_vertices();

  BoundaryConditions<3>::init_boundary_groups(mesh_->V0_,
      mesh_->bc_groups_, 0.01); // .01, hang for astronaut

  BCs_.set_script(config_->bc_type);
  BCs_.init_script(mesh_);
  P_ = pinning_matrix(mesh_->V_, mesh_->T_, mesh_->is_fixed_, false);
  mesh_->update_free_map();
  mesh_->init();
}
