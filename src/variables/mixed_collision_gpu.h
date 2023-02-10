#pragma once

#include "EigenTypes.h"
// #include <ccdgpu/helper.cuh>
#include <ipc/broad_phase/sweep_and_tiniest_queue.hpp>
#include <stq/cpu/aabb.hpp>
#include <stq/gpu/aabb.cuh>

namespace mfem {


  class SimConfig;

  template<int DIM>
  class MixedCollisionGpu : public MixedVariable<DIM> {

    typedef MixedVariable<DIM> Base;

  public:

    // MixedCollisionGpu(std::shared_ptr<Mesh> mesh,
    //     std::shared_ptr<SimConfig> config)
    //     : MixedVariable<DIM>(mesh), config_(config)
    // {}
  };
}