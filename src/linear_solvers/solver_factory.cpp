#include "solver_factory.h"
#include "mesh/mesh.h"
#include "EigenTypes.h"
#include "eigen_solver.h"
#include "affine_pcg.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

using namespace mfem;
using namespace Eigen;

using Scalar = double;
//using RowMajor = Eigen::RowMajor;

SolverFactory::SolverFactory() {

  // Eigen LLT
  using LLT = SimplicialLLT<SparseMatrix<Scalar, RowMajor>>;
  register_type(SolverType::SOLVER_EIGEN_LLT, "eigen-llt",
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<LinearSolver<Scalar, RowMajor>>
      {return std::make_unique<EigenSolver<LLT, Scalar, RowMajor>>();});

  // Eigen LDLT
  using LDLT = SimplicialLDLT<SparseMatrix<Scalar, RowMajor>>;
  register_type(SolverType::SOLVER_EIGEN_LDLT, "eigen-ldlt",
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<LinearSolver<Scalar, RowMajor>>
      {return std::make_unique<EigenSolver<LDLT, Scalar, RowMajor>>();});

  // Eigen LU
  using LU = SparseLU<SparseMatrix<Scalar, RowMajor>>;
  register_type(SolverType::SOLVER_EIGEN_LU, "eigen-lu",
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<LinearSolver<Scalar, RowMajor>>
      {return std::make_unique<EigenSolver<LU, Scalar, RowMajor>>();});

  #if defined(SIM_USE_CHOLMOD)
  using CHOLMOD = CholmodSupernodalLLT<SparseMatrix<Scalar, RowMajor>>;
  register_type(SolverType::SOLVER_CHOLMOD, "cholmod",
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<LinearSolver<Scalar, RowMajor>>
      {return std::make_unique<EigenSolver<CHOLMOD, Scalar, RowMajor>>();});
  #endif

  // Affine Body Dynamics initialized PCG with ARAP preconditioner
  register_type(SolverType::SOLVER_AFFINE_PCG, "affine-pcg",
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<LinearSolver<Scalar, RowMajor>>
      {return std::make_unique<AffinePCG<Scalar, RowMajor>>(mesh, config);});
}

std::unique_ptr<LinearSolver<Scalar, RowMajor>>
SolverFactory::create(
    std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config) {

  if (auto it = type_creators_.find(config->solver_type);
      it !=  type_creators_.end()) {
    return it->second(mesh, config);
  }
  std::cout << "SolverFactory create: type not found" << std::endl;
  return nullptr;
}

std::unique_ptr<LinearSolver<Scalar, RowMajor>>
SolverFactory::create(
    const std::string& type, std::shared_ptr<Mesh> mesh,
    std::shared_ptr<SimConfig> config) {

  if (auto it = str_type_creators_.find(type); it !=  str_type_creators_.end())
  {
    return it->second(mesh, config);
  }
  return nullptr;
}

void SolverFactory::register_type(SolverType type,
    const std::string& name, TypeCreator func) {
  type_creators_.insert(std::pair<SolverType, TypeCreator>(type, func));
  str_type_creators_.insert(std::pair<std::string, TypeCreator>(name, func));
  str_type_map_.insert(std::pair<std::string, SolverType>(name, type));
  type_str_map_.insert(std::pair<SolverType, std::string>(type, name));
  names_.push_back(name);
}

