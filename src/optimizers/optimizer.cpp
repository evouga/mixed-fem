#include "optimizer.h"
#include "mesh/mesh.h"
#include "time_integrators/BDF.h"

using namespace mfem;
using namespace Eigen;

template <int DIM>
void Optimizer<DIM>::reset() {
  state_.mesh_->V_ = state_.mesh_->Vref_;
  state_.mesh_->clear_fixed_vertices();
  
  BoundaryConditions<DIM>::init_boundary_groups(state_.mesh_->Vref_,
      state_.mesh_->bc_groups_, 0.01); // .01, hang for astronaut

  state_.BCs_.set_script(state_.config_->bc_type);
  state_.BCs_.init_script(state_.mesh_);

  state_.mesh_->update_free_map();
  state_.mesh_->init();
}

template class mfem::Optimizer<3>;
template class mfem::Optimizer<2>;
