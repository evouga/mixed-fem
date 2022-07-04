#pragma once
#include "config.h"
#include <memory>
#include <map>
#include "linear_solver.h"

namespace mfem {

  class Mesh;

  // Factory to create linear solver by typename or string
  class SolverFactory {
  public:

    // Lambda type that takes as input a mesh and simulation config
    // returns a unique_ptr to a new optimizer
    using TypeCreator = std::unique_ptr<LinearSolver<double,Eigen::RowMajor>>(*)
        (std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config);

    // Registers all the optimizers
    SolverFactory();

    // Find and return optimizer by enumeration type
    std::unique_ptr<LinearSolver<double, Eigen::RowMajor>> create(
        std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config);

    // Find and return optimizer by string type
    std::unique_ptr<LinearSolver<double, Eigen::RowMajor>> create(
        const std::string& type, std::shared_ptr<Mesh> mesh,
        std::shared_ptr<SimConfig> config);

    const std::vector<std::string>& names() {
      return names_;
    }

    SolverType type_by_name(const std::string& name) {
      auto it = str_type_map_.find(name);
      if (it != str_type_map_.end()) {
        return str_type_map_[name];
      }
      return SolverType::SOLVER_EIGEN_LLT;
    }

    std::string name_by_type(SolverType type) {
      auto it = type_str_map_.find(type);
      if (it != type_str_map_.end()) {
        return type_str_map_[type];
      }
      return "";
    }

  private:

    void register_type(SolverType type, const std::string& name,
        TypeCreator func);

    std::map<SolverType, TypeCreator> type_creators_;
    std::map<std::string, TypeCreator> str_type_creators_;
    std::map<std::string, SolverType> str_type_map_;
    std::map<SolverType, std::string> type_str_map_;
    std::vector<std::string> names_;
  };
}
