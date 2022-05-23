#include "materials/neohookean.h"
#include "materials/corotational.h"
#include "materials/arap.h"
#include "materials/stable_neohookean.h"
#include "materials/fung.h"
#include "materials/material_model_factory.h"

using namespace mfem;

MaterialModelFactory::MaterialModelFactory() {
  
  // ARAP
  register_type(MaterialModelType::MATERIAL_ARAP, ArapModel::name(),
      [](const std::shared_ptr<MaterialConfig>& config)
      ->std::unique_ptr<MaterialModel>
      {return std::make_unique<ArapModel>(config);});

  // Fixed Corotated Elasticity
  register_type(MaterialModelType::MATERIAL_COROT, Corotational::name(),
      [](const std::shared_ptr<MaterialConfig>& config)
      ->std::unique_ptr<MaterialModel>
      {return std::make_unique<Corotational>(config);});

  // Fung
  register_type(MaterialModelType::MATERIAL_FUNG, Fung::name(),
      [](const std::shared_ptr<MaterialConfig>& config)
      ->std::unique_ptr<MaterialModel>
      {return std::make_unique<Fung>(config);});   


  // Neohookean
  register_type(MaterialModelType::MATERIAL_NH, Neohookean::name(),
      [](const std::shared_ptr<MaterialConfig>& config)
      ->std::unique_ptr<MaterialModel>
      {return std::make_unique<Neohookean>(config);});   

  // Stable Neohookean
  register_type(MaterialModelType::MATERIAL_SNH, StableNeohookean::name(),
      [](const std::shared_ptr<MaterialConfig>& config)
      ->std::unique_ptr<MaterialModel>
      {return std::make_unique<StableNeohookean>(config);});   

}

std::unique_ptr<MaterialModel> MaterialModelFactory::create(
    const std::shared_ptr<MaterialConfig>& config) {

  if (auto it = type_creators_.find(config->material_model);
      it !=  type_creators_.end()) {
    return it->second(config);
  }
  std::cout << "MaterialModelFactory create: type not found" << std::endl;
  return nullptr;
}

std::unique_ptr<MaterialModel> MaterialModelFactory::create(
    const std::string& type, const std::shared_ptr<MaterialConfig>& config) {

  if (auto it = str_type_creators_.find(type); it !=  str_type_creators_.end())
  {
    return it->second(config);
  }
  return nullptr;
}

void MaterialModelFactory::register_type(MaterialModelType type,
    const std::string& name, TypeCreator func) {
  type_creators_.insert(std::pair<MaterialModelType, TypeCreator>(type, func));
  str_type_creators_.insert(std::pair<std::string, TypeCreator>(name, func));
}