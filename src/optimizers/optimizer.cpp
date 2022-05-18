#include "optimizer.h"
#include "pinning_matrix.h"

using namespace mfem;
using namespace Eigen;


void Optimizer::reset() {
  nelem_ = object_->T_.rows();
  object_->V_ = object_->V0_;



  BoundaryConditions<3>::init_boundary_groups(object_->V0_,
      object_->bc_groups_, 0.1);

  BCs_.set_script(config_->bc_type);

    // void init_script(std::shared_ptr<SimObject> &mesh);
    // int step_script(std::shared_ptr<SimObject> &mesh, double dt);
    // void set_script(BCScriptType script_type);
  object_->is_fixed_(0) = 1;
  P_ = pinning_matrix(object_->V_, object_->T_, object_->is_fixed_, false);


}
