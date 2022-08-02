#pragma once

#include "factory.h"
#include "config.h"
#include "variables/mixed_variable.h"

namespace mfem {

	class Mesh;

  template<int DIM>
	class MixedVariableFactory : public Factory<VariableType,
			MixedVariable<DIM>, std::shared_ptr<Mesh>, std::shared_ptr<SimConfig>> {
	public:
    MixedVariableFactory();
	};
}
