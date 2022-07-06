#include "solver_factory.h"
#include "mesh/mesh.h"
#include "EigenTypes.h"
#include "linear_solvers/eigen_solver.h"
#include "linear_solvers/affine_pcg.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

using namespace mfem;
using namespace Eigen;

using Scalar = double;

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
