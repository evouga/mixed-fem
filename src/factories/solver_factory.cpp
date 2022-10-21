#include "solver_factory.h"
#include "mesh/mesh.h"
#include "EigenTypes.h"
#include "linear_solvers/eigen_solver.h"
#include "linear_solvers/eigen_iterative_solver.h"
#include "linear_solvers/affine_pcg.h"
#include "linear_solvers/linear_system.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

using namespace mfem;
using namespace Eigen;

using Scalar = double;

template<int DIM>
SolverFactory<DIM>::SolverFactory() {

  // Sparse matrix type
  using SpMat = SystemMatrixPD<Scalar>::MatrixType; 
  
  // Eigen LLT
  using LLT = SimplicialLLT<SpMat>;
  this->register_type(SolverType::SOLVER_EIGEN_LLT, "eigen-llt",
      [](SimState<DIM>* state)
      ->std::unique_ptr<LinearSolver<Scalar, DIM>>
      {return std::make_unique<EigenSolver<LLT, SystemMatrixPD<Scalar>,
        Scalar, DIM>>(state);});

  // Eigen LDLT
  using LDLT = SimplicialLDLT<SpMat>;
  this->register_type(SolverType::SOLVER_EIGEN_LDLT, "eigen-ldlt",
      [](SimState<DIM>* state)
      ->std::unique_ptr<LinearSolver<Scalar, DIM>>
      {return std::make_unique<EigenSolver<LDLT, SystemMatrixPD<Scalar>,
        Scalar, DIM>>(state);});

  // Eigen LU
  using LU = SparseLU<SpMat>;
  this->register_type(SolverType::SOLVER_EIGEN_LU, "eigen-lu",
      [](SimState<DIM>* state)
      ->std::unique_ptr<LinearSolver<Scalar, DIM>>
      {return std::make_unique<EigenSolver<LU, SystemMatrixPD<Scalar>,
        Scalar, DIM>>(state);});

  #if defined(SIM_USE_CHOLMOD)
  using CHOLMOD = CholmodSupernodalLLT<SpMat>;
  this->register_type(SolverType::SOLVER_CHOLMOD, "cholmod",
      [](SimState<DIM>* state)
      ->std::unique_ptr<LinearSolver<Scalar, DIM>>
      {return std::make_unique<EigenSolver<CHOLMOD, SystemMatrixPD<Scalar>,
        Scalar, DIM>>(state);});
  #endif

  // Affine Body Dynamics initialized PCG with ARAP preconditioner
//   register_type(SolverType::SOLVER_AFFINE_PCG, "affine-pcg",
//       [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
//       ->std::unique_ptr<LinearSolver<Scalar, RowMajor>>
//       {return std::make_unique<AffinePCG<Scalar, RowMajor>>(mesh, config);});

  using EIGEN_CG_DIAG = ConjugateGradient<SpMat, Lower|Upper>;
  this->register_type(SolverType::SOLVER_EIGEN_CG_DIAG, "eigen-pcg-diag",
      [](SimState<DIM>* state)
      ->std::unique_ptr<LinearSolver<Scalar, DIM>>
      { 
        return std::make_unique<EigenIterativeSolver<
          EIGEN_CG_DIAG, SystemMatrixPD<Scalar>,
          Scalar, DIM>>(state);
      }
  );

  using SOLVER_EIGEN_CG_IC = ConjugateGradient<SpMat, Lower|Upper,
      IncompleteCholesky<Scalar>>;
  this->register_type(SolverType::SOLVER_EIGEN_CG_IC, "eigen-pcg-IC",
      [](SimState<DIM>* state)
      ->std::unique_ptr<LinearSolver<Scalar, DIM>>
      { 
        return std::make_unique<EigenIterativeSolver<
          SOLVER_EIGEN_CG_IC, SystemMatrixPD<Scalar>,
          Scalar, DIM>>(state);
      }
  );
}

template class mfem::SolverFactory<3>;
template class mfem::SolverFactory<2>;