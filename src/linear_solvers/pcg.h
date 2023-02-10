#pragma once

// Simple modification of Eigen PCG with support for recording relative
// residuals during the solve.

#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>

namespace Eigen
{

  namespace internal
  {

    template <typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
    void conjugate_gradient(const MatrixType &mat, const Rhs &rhs, Dest &x,
                            const Preconditioner &precond, Index &iters,
                            typename Dest::RealScalar &tol_error,
                            bool save_residual_history,
                            std::vector<double>& residual_history) {
      using std::abs;
      using std::sqrt;
      typedef typename Dest::RealScalar RealScalar;
      typedef typename Dest::Scalar Scalar;
      typedef Matrix<Scalar, Dynamic, 1> VectorType;

      RealScalar tol = tol_error;
      Index maxIters = iters;

      Index n = mat.cols();

      VectorType residual = rhs - mat * x; // initial residual

      RealScalar rhsNorm2 = rhs.squaredNorm();
      if (rhsNorm2 == 0)
      {
        x.setZero();
        iters = 0;
        tol_error = 0;
        return;
      }
      const RealScalar considerAsZero = (std::numeric_limits<RealScalar>::min)();
      RealScalar threshold = numext::maxi(RealScalar(tol * tol * rhsNorm2), considerAsZero);
      RealScalar residualNorm2 = residual.squaredNorm();
      if (residualNorm2 < threshold)
      {
        iters = 0;
        tol_error = sqrt(residualNorm2 / rhsNorm2);
        return;
      }

      VectorType p(n);
      p = precond.solve(residual); // initial search direction

      VectorType z(n), tmp(n);
      RealScalar absNew = numext::real(residual.dot(p)); // the square of the absolute value of r scaled by invM
      Index i = 0;
      while (i < maxIters)
      {
        tmp.noalias() = mat * p; // the bottleneck of the algorithm

        Scalar alpha = absNew / p.dot(tmp); // the amount we travel on dir
        x += alpha * p;                     // update solution
        residual -= alpha * tmp;            // update residual

        residualNorm2 = residual.squaredNorm();

        if (save_residual_history)
        {
          double res_norm = (mat * x - rhs).norm();
          residual_history.push_back(res_norm / sqrt(rhsNorm2));
        }

        if (residualNorm2 < threshold)
          break;

        z = precond.solve(residual); // approximately solve for "A z = residual"

        RealScalar absOld = absNew;
        absNew = numext::real(residual.dot(z)); // update the absolute value of r
        RealScalar beta = absNew / absOld;      // calculate the Gram-Schmidt value used to create the new search direction
        p = z + beta * p;                       // update search direction
        i++;
      }
      tol_error = sqrt(residualNorm2 / rhsNorm2);
      iters = i;
    }

  }

  template <typename MatrixType_, int UpLo_ = Lower,
            typename Preconditioner_ = DiagonalPreconditioner<typename MatrixType_::Scalar>>
  class ConjugateGradient2;

  namespace internal
  {

    template <typename MatrixType_, int UpLo_, typename Preconditioner_>
    struct traits<ConjugateGradient2<MatrixType_, UpLo_, Preconditioner_>>
    {
      typedef MatrixType_ MatrixType;
      typedef Preconditioner_ Preconditioner;
    };

  }

  template <typename MatrixType_, int UpLo_, typename Preconditioner_>
  class ConjugateGradient2 : public IterativeSolverBase<ConjugateGradient2<MatrixType_, UpLo_, Preconditioner_>>
  {
    typedef IterativeSolverBase<ConjugateGradient2> Base;
    using Base::m_error;
    using Base::m_info;
    using Base::m_isInitialized;
    using Base::m_iterations;
    using Base::matrix;

  public:
    typedef MatrixType_ MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef Preconditioner_ Preconditioner;

    enum
    {
      UpLo = UpLo_
    };

  public:
    ConjugateGradient2() : Base() {}

    template <typename MatrixDerived>
    explicit ConjugateGradient2(const EigenBase<MatrixDerived> &A) : Base(A.derived()) {}

    ~ConjugateGradient2() {}


    void setSaveResiduals(bool save_residual) { m_save_residual = save_residual; }
    const std::vector<double>& getResiduals() const { return m_residuals; }
    void clearResiduals() { m_residuals.clear(); }

    template <typename Rhs, typename Dest>
    void _solve_vector_with_guess_impl(const Rhs &b, Dest &x) const
    {
      typedef typename Base::MatrixWrapper MatrixWrapper;
      typedef typename Base::ActualMatrixType ActualMatrixType;
      enum
      {
        TransposeInput = (!MatrixWrapper::MatrixFree) && (UpLo == (Lower | Upper)) && (!MatrixType::IsRowMajor) && (!NumTraits<Scalar>::IsComplex)
      };
      typedef std::conditional_t<TransposeInput, Transpose<const ActualMatrixType>, ActualMatrixType const &> RowMajorWrapper;
      //EIGEN_STATIC_ASSERT(internal::check_implication(MatrixWrapper::MatrixFree, UpLo == (Lower | Upper)), MATRIX_FREE_CONJUGATE_GRADIENT_IS_COMPATIBLE_WITH_UPPER_UNION_LOWER_MODE_ONLY);
      typedef std::conditional_t<UpLo == (Lower | Upper),
                                 RowMajorWrapper,
                                 typename MatrixWrapper::template ConstSelfAdjointViewReturnType<UpLo>::Type>
          SelfAdjointWrapper;

      m_iterations = Base::maxIterations();
      m_error = Base::m_tolerance;

      RowMajorWrapper row_mat(matrix());
      internal::conjugate_gradient(SelfAdjointWrapper(row_mat), b, x,
          Base::m_preconditioner, m_iterations, m_error,
          m_save_residual, m_residuals);
      m_info = m_error <= Base::m_tolerance ? Success : NoConvergence;
    }

  protected:
    bool m_save_residual;
    mutable std::vector<double> m_residuals;
  };

} // end namespace Eigen