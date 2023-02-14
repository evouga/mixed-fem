#pragma once

#include "variable.h"
#include "optimizers/optimizer_data.h"
#include "time_integrators/implicit_integrator.h"
#include "utils/sparse_utils.h"
#include "mesh/mesh.h"
#include <thrust/device_vector.h>
#include "EigenTypes.h"
#include "utils/sparse_matrix_gpu.h"
#include <thrust/host_vector.h>

namespace mfem {

  class SimConfig;

  // Nodal displacement variable
  template<int DIM>
  class DisplacementGpu : public Variable<DIM, STORAGE_THRUST> {

    typedef Variable<DIM, STORAGE_THRUST> Base;

    // Number of degrees of freedom per element
    // For DIM == 3 we have 6 DOFs per element, and
    // 3 DOFs for DIM == 2;
    __host__ __device__
    static constexpr int N() {
      return DIM == 3 ? 12 : 9;
    }

    __host__ __device__
    static constexpr int M() {
      return DIM * DIM;
    }

    // Matrix and vector data types
    using VecD  = Eigen::Matrix<double, DIM, 1>;   // 3x1 or 2x1
    using MatD  = Eigen::Matrix<double, DIM, DIM>; // 3x3 or 2x2
    using VecN  = Eigen::Vector<double, N()>;      // 6x1 or 3x1
    using VecM  = Eigen::Vector<double, M()>;      // 9x1
    using MatN  = Eigen::Matrix<double, N(), N()>; // 6x6 or 3x3
    using MatMN = Eigen::Matrix<double, M(), N()>; // 9x6 or 4x3

  public:

    using typename Base::VectorType;

    DisplacementGpu(std::shared_ptr<Mesh> mesh,
          std::shared_ptr<SimConfig> config);

    double energy(VectorType&) override;
    void update(VectorType& x, double dt) override;
    void reset() override;
    void post_solve() override;

    VectorType& rhs() override;
    VectorType gradient() override;

    enum class ProjectionType {
      WITH_DIRICHLET,
      WITHOUT_DIRICHLET
    };

    double* to_full(const VectorType& x,
        ProjectionType type = ProjectionType::WITH_DIRICHLET);

    void to_full(const VectorType& x, VectorType& out,
        ProjectionType type = ProjectionType::WITH_DIRICHLET);

    const Eigen::SparseMatrix<double, Eigen::RowMajor>& lhs() override {
      return lhs_;
    }

    VectorType& delta() override {
      return dx_;
    }
    VectorType& value() override {
      return x_;
    }

    const std::shared_ptr<ImplicitIntegrator<STORAGE_THRUST>> integrator()
        const {
      return integrator_;
    }

    std::shared_ptr<ImplicitIntegrator<STORAGE_THRUST>> integrator() {
      return integrator_;
    }

    int size() const override {
      return x_h_.size();
    }

    void product_hessian(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const override {
      assert(out.size() == x.size());
      out += lhs_ * x;
    }

    /// @brief Compute the explicit predictor equal to x - xtilde - h^2 fext
    /// @param out - output vector (size equals x.size())
    /// @param projected - if true, the output is projected to the free DOFs
    void explicit_predictor(double* out, bool projected = false);


    void apply(double* x, const double* b, int cols=1) override;

    void extract_diagonal(double* diag) override;

  private:

    using Base::mesh_;
    int full_size_;
    int reduced_size_;

    std::shared_ptr<SimConfig> config_;
    std::shared_ptr<ImplicitIntegrator<STORAGE_THRUST>> integrator_;

    Eigen::SparseMatrix<double, Eigen::RowMajor> lhs_;
    
    VectorType dx_;
    VectorType x_;
    VectorType grad_;
    VectorType rhs_;
    VectorType b_;

    VectorType x_tilde_;
    VectorType tmp_;
    VectorType f_ext_;

    Eigen::VectorXd x_h_;       // displacement variables
    Eigen::VectorXd b_h_;       // dirichlet values
    Eigen::VectorXd dx_h_;      // displacement deltas
    Eigen::VectorXd rhs_h_;     // right-hand-side vector
    Eigen::VectorXd grad_h_;    // Gradient with respect to 's' variables

    SparseMatrixGpu P_gpu_;   // projection matrix
    SparseMatrixGpu PT_gpu_;  // projection matrix, transposed
    SparseMatrixGpu PM_gpu_;  // projected mass matrix (from the left) P * M
    SparseMatrixGpu PMP_gpu_;  // projected mass matrix (from the left) P * M
    SparseMatrixGpu M_gpu_;   // mass matrix
    thrust::device_vector<double> diag_gpu_; // diagonal of the mass matrix

  };
}
