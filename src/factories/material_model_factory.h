#pragma once

#include "factory.h"
#include "config.h"

namespace mfem {

	class MaterialModel;

	class MaterialModelFactory : public Factory<MaterialModelType,
			MaterialModel, std::shared_ptr<MaterialConfig>> {
	public:
    MaterialModelFactory();
	};
}
