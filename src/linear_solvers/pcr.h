#pragma once

#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>

namespace Eigen { 

namespace internal {

template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
void conjugate_residual(const MatrixType& mat, const Rhs& rhs, Dest& x,
                        const Preconditioner& precond, Index& iters,
                        typename Dest::RealScalar& tol_error,
                        bool save_residual_history,
                        std::vector<double>& residual_history) {
  using std::sqrt;
  using std::abs;
  typedef typename Dest::RealScalar RealScalar;
  typedef typename Dest::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,1> VectorType;
  
  RealScalar tol = tol_error;
  Index maxIters = iters;
  
  Index m = mat.rows(), n = mat.cols();

  VectorType residual = rhs - mat * x;

  RealScalar rhsNorm2 = rhs.squaredNorm();
  if(rhsNorm2 == 0) {
    x.setZero();
    iters = 0;
    tol_error = 0;
    return;
  }

  RealScalar threshold = tol*tol*rhsNorm2;
  RealScalar residualNorm2 = residual.squaredNorm();
  if (residualNorm2 < threshold) {
    iters = 0;
    tol_error = sqrt(residualNorm2 / rhsNorm2);
    return;
  }
  
  VectorType p = precond.solve(residual);
  residual = p;

  Index i = 0;
  // VectorType p = residual; // initial search direction
  VectorType Ap = mat * p;
  VectorType Ar = Ap;
  VectorType tmp(n);

  while(i < maxIters) {
    tmp = precond.solve(Ap);
    Scalar rAr = residual.dot(Ar);

    // the amount we travel on dir
    Scalar alpha = rAr / tmp.dot(Ap);

    // update solution
    x += alpha * p;

    // update residual
    residual -= alpha * tmp;

    residualNorm2 = residual.squaredNorm();

    if (save_residual_history) {
      double res_norm = (mat*x -rhs).norm();
      residual_history.push_back(res_norm/sqrt(rhsNorm2));
    }
    if(residualNorm2 < threshold)
      break;

    Ar = mat * residual;

    // Gram-Schmidt value used to create the new search direction
    RealScalar beta = residual.dot(Ar) / rAr;

    // update search direction
    p = residual + beta * p;
    Ap = Ar + beta * Ap;
    i++;
  }
  tol_error = sqrt(residualNorm2 / rhsNorm2);
  iters = i;
}

}

template<typename MatrixType_,
         typename Preconditioner_ = DiagonalPreconditioner<typename MatrixType_::Scalar> >
class ConjugateResidual;

namespace internal {

template< typename MatrixType_, typename Preconditioner_>
struct traits<ConjugateResidual<MatrixType_,Preconditioner_> >
{
  typedef MatrixType_ MatrixType;
  typedef Preconditioner_ Preconditioner;
};

}

template< typename MatrixType_, typename Preconditioner_>
class ConjugateResidual 
    : public IterativeSolverBase<ConjugateResidual<MatrixType_,Preconditioner_>>
{
  typedef IterativeSolverBase<ConjugateResidual> Base;
  using Base::matrix;
  using Base::m_error;
  using Base::m_iterations;
  using Base::m_info;
  using Base::m_isInitialized;
public:
  typedef MatrixType_ MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Preconditioner_ Preconditioner;

public:

  ConjugateResidual() : Base() {}

  template<typename MatrixDerived>
  explicit ConjugateResidual(const EigenBase<MatrixDerived>& A) : Base(A.derived()) {}

  ~ConjugateResidual() {}

  void setSaveResiduals(bool save_residual) { m_save_residual = save_residual; }
  const std::vector<double>& getResiduals() const { return m_residuals; }

  template<typename Rhs,typename Dest>
  void _solve_vector_with_guess_impl(const Rhs& b, Dest& x) const
  {
    m_iterations = Base::maxIterations();
    m_error = Base::m_tolerance;

    internal::conjugate_residual(matrix(), b, x, Base::m_preconditioner,
        m_iterations, m_error, m_save_residual, m_residuals);
    m_info = m_error <= Base::m_tolerance ? Success : NoConvergence;
  }
protected:
  bool m_save_residual;
  mutable std::vector<double> m_residuals;

};

}
