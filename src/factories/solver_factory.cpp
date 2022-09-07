#include "solver_factory.h"
#include "mesh/mesh.h"
#include "EigenTypes.h"
#include "linear_solvers/eigen_solver.h"
#include "linear_solvers/eigen_iterative_solver.h"
#include "linear_solvers/affine_pcg.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

using namespace mfem;
using namespace Eigen;

using Scalar = double;

SolverFactory::SolverFactory() {

  // Sparse matrix type
  using SpMat = SparseMatrix<Scalar, RowMajor>; 
  
  // Eigen LLT
  using LLT = SimplicialLLT<SpMat>;
  register_type(SolverType::SOLVER_EIGEN_LLT, "eigen-llt",
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<LinearSolver<Scalar, RowMajor>>
      {return std::make_unique<EigenSolver<LLT, Scalar, RowMajor>>();});

  // Eigen LDLT
  using LDLT = SimplicialLDLT<SpMat>;
  register_type(SolverType::SOLVER_EIGEN_LDLT, "eigen-ldlt",
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<LinearSolver<Scalar, RowMajor>>
      {return std::make_unique<EigenSolver<LDLT, Scalar, RowMajor>>();});

  // Eigen LU
  using LU = SparseLU<SpMat>;
  register_type(SolverType::SOLVER_EIGEN_LU, "eigen-lu",
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<LinearSolver<Scalar, RowMajor>>
      {return std::make_unique<EigenSolver<LU, Scalar, RowMajor>>();});

  #if defined(SIM_USE_CHOLMOD)
  using CHOLMOD = CholmodSupernodalLLT<SpMat>;
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

  using EIGEN_CG_DIAG = ConjugateGradient<SpMat, Lower|Upper>;
  register_type(SolverType::SOLVER_EIGEN_CG_DIAG, "eigen-pcg-diag",
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<LinearSolver<Scalar, RowMajor>>
      { 
        return std::make_unique<EigenIterativeSolver<
          EIGEN_CG_DIAG, Scalar, RowMajor>>(config);
      }
  );

  using SOLVER_EIGEN_CG_IC = ConjugateGradient<SpMat, Lower|Upper,
      IncompleteCholesky<double>>;
  register_type(SolverType::SOLVER_EIGEN_CG_IC, "eigen-pcg-IC",
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<LinearSolver<Scalar, RowMajor>>
      { 
        return std::make_unique<EigenIterativeSolver<
          SOLVER_EIGEN_CG_IC, Scalar, RowMajor>>(config);
      }
  );
}
