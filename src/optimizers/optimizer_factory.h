#include "config.h"
#include <memory>
#include <map>

namespace mfem {

  class Optimizer;

  // Factory to create optimizer by typename or string
  class OptimizerFactory {
  public:

    // Lambda type that takes as input a mesh and simulation config
    // returns a unique_ptr to a new optimizer
    using TypeCreator = std::unique_ptr<Optimizer>(*)(
        std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config);

    // Registers all the optimizers
    OptimizerFactory();

    // Find and return optimizer by enumeration type
    std::unique_ptr<Optimizer> create(
        std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config);

    // Find and return optimizer by string type
    std::unique_ptr<Optimizer> create(const std::string& type,
        std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config);

  private:

    void register_type(OptimizerType type, const std::string& name,
        TypeCreator func);

    std::map<OptimizerType, TypeCreator> type_creators_;
    std::map<std::string, TypeCreator> str_type_creators_;
    std::vector<std::string> types_;
  };
}