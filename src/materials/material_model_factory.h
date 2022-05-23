#include "materials/material_model.h"
#include "config.h"
#include <memory>
#include <map>

namespace mfem {
  // Factory object to create material models by typename or string
  class MaterialModelFactory {
  public:

    // Lambda type that takes as input a material config type and
    // returns a unique_ptr to a new material model type.
    using TypeCreator = std::unique_ptr<MaterialModel>(*)(
        const std::shared_ptr<MaterialConfig>& config);

    // Registers all the material models
    MaterialModelFactory();

    // Find and return material model by enumeration type
    std::unique_ptr<MaterialModel> create(MaterialModelType type,
        const std::shared_ptr<MaterialConfig>& config);

    // Find and return material model by string type
    std::unique_ptr<MaterialModel> create(const std::string& type,
        const std::shared_ptr<MaterialConfig>& config);

  private:

    void register_type(MaterialModelType type, const std::string& name,
        TypeCreator func);

    std::map<MaterialModelType, TypeCreator> type_creators_;
    std::map<std::string, TypeCreator> str_type_creators_;
    std::vector<std::string> types_;
  };
}