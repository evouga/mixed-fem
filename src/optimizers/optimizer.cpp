#include "optimizer.h"
#include "pinning_matrix.h"
#include "mesh/mesh.h"

using namespace mfem;
using namespace Eigen;


void Optimizer::reset() {
  nelem_ = object_->T_.rows();
  object_->V_ = object_->V0_;
  object_->clear_fixed_vertices();

  BoundaryConditions<3>::init_boundary_groups(object_->V0_,
      object_->bc_groups_, 0.01); // .01, hang for astronaut

  BCs_.set_script(config_->bc_type);
  BCs_.init_script(object_);
  P_ = pinning_matrix(object_->V_, object_->T_, object_->is_fixed_, false);
  object_->update_free_map();
  object_->init();
}
