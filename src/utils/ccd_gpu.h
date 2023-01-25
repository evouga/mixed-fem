#pragma once

#include "EigenTypes.h"
#include "ipc/ipc.hpp"
#include <thrust/device_vector.h>

namespace ipc {

  template <typename Scalar, int DIM>
  Scalar accd_gpu(const thrust::device_vector<Scalar>& x1,
      const thrust::device_vector<Scalar>& x2,
      const ipc::CollisionMesh& mesh,
      Scalar dhat);
}