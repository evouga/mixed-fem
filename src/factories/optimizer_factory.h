#pragma once

#include "factory.h"
#include "config.h"

namespace mfem {

	class Optimizer;
	class Mesh;

	class OptimizerFactory : public Factory<OptimizerType,
			Optimizer, std::shared_ptr<Mesh>, std::shared_ptr<SimConfig>> {
	public:
    OptimizerFactory();
	};
}
