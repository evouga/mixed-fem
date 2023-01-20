#include "optimizer.h"
#include "mesh/mesh.h"
#include "time_integrators/BDF.h"

using namespace mfem;
using namespace Eigen;

template <int DIM, StorageType STORAGE>
void Optimizer<DIM,STORAGE>::reset() {
  state_.mesh_->init();
}

template class mfem::Optimizer<3, STORAGE_THRUST>;
template class mfem::Optimizer<3, STORAGE_EIGEN>;
template class mfem::Optimizer<2, STORAGE_EIGEN>;
