#pragma once

#include <thrust/device_vector.h>
#include "EigenTypes.h"
#include "mixed_variable.h"
#include <cusparse.h>
#include "utils/sparse_matrix_gpu.h"

namespace mfem {

  template<int DIM>
  class MixedStretchGpu : public MixedVariable<DIM> {

    typedef MixedVariable<DIM> Base;

    template <typename T>
    using vector = thrust::device_vector<T>;

    // Number of degrees of freedom per element
    // For DIM == 3 we have 6 DOFs per element, and
    // 3 DOFs for DIM == 2;
    __host__ __device__
    static constexpr int N() {
      return DIM == 3 ? 6 : 3;
    }

    __host__ __device__
    static constexpr int M() {
      return DIM * DIM;
    }

    // Matrix and vector data types
    using MatD  = Eigen::Matrix<double, DIM, DIM>; // 3x3 or 2x2
    using VecN  = Eigen::Vector<double, N()>;      // 6x1 or 3x1
    using VecM  = Eigen::Vector<double, M()>;      // 9x1
    using MatM  = Eigen::Matrix<double, M(), M()>; // 9x9
    using MatN  = Eigen::Matrix<double, N(), N()>; // 6x6 or 3x3
    using MatMN = Eigen::Matrix<double, M(), N()>; // 9x6 or 4x3

    __host__ __device__
    static constexpr MatN Syminv() {
      MatN m; 
      if constexpr (DIM == 3) {
        m = (VecN() << 1,1,1,0.5,0.5,0.5).finished().asDiagonal();
      } else {
        m = (VecN() << 1,1,0.5).finished().asDiagonal();
      }
      return m;
    }

  public:

    MixedStretchGpu(std::shared_ptr<Mesh> mesh);

    static std::string name() {
      return "mixed-stretch-gpu";
    }

    void reset();
    void init_variables(int i, double* si_data);
    void local_derivatives(int i, double* s, double* g,
        double* H, double* Hinv, double* dSdF, double* Jloc, double* Aloc,
        double* vols);

    double energy(const Eigen::VectorXd& x, const Eigen::VectorXd& s) override {return 0.0;}
    double constraint_value(const Eigen::VectorXd& x,
        const Eigen::VectorXd& s) override{return 0.0;}
    void update(const Eigen::VectorXd& x, double dt) override;
    void solve(const Eigen::VectorXd& dx) override {}

    const Eigen::SparseMatrix<double, Eigen::RowMajor>& lhs() override {
      return dummy_A_;
    }
    Eigen::VectorXd rhs() override { return dummy_; }
    Eigen::VectorXd gradient() override { return dummy_; }
    Eigen::VectorXd gradient_mixed() override { return dummy_; }
    Eigen::VectorXd gradient_dual() override { return dummy_; }

    Eigen::VectorXd& delta() override {
      return dummy_;
    }

    Eigen::VectorXd& value() override {
      return dummy_;
    }

    Eigen::VectorXd& lambda() override {
      return dummy_;
    }
    
    int size() const override {
      return 0;// s_.size() * N();
    }

    int size_dual() const override {
      return 0;//la_.size() * N();
    }

    void evaluate_constraint(const Eigen::VectorXd& x,
        Eigen::VectorXd&) override {}
    void hessian(Eigen::SparseMatrix<double>&) override {}
    void hessian_inv(Eigen::SparseMatrix<double>&) override {}
    void jacobian_x(Eigen::SparseMatrix<double>&) override {}
    void jacobian_mixed(Eigen::SparseMatrix<double>&) override {}

    void product_hessian(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const override{} 
    void product_hessian_inv(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const override {} 
    void product_jacobian_x(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out, bool transposed) const override {}
    void product_jacobian_mixed(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const override {}

  protected:

    double local_energy(const VecN& S, double mu);
    VecN local_gradient(const VecN& S, double mu);
    MatN local_hessian(const VecN& S, double mu);

    using Base::mesh_;

    int nelem_;          // number of elements, |E|

    Eigen::SparseMatrix<double, Eigen::RowMajor> dummy_A_;
    Eigen::VectorXd dummy_;
    vector<double> rhs_; // RHS for primal condensation system

    // Per element data
    vector<double> s_;     // deformation variable        N|E| x 1
    vector<double> ds_;    // deformation variable deltas N|E| x 1
    vector<double> la_;    // lagrange multipliers        N|E| x 1
    vector<double> R_;     // rotations                   M|E| x 1
    vector<double> S_;     // deformation (function of x) N|E| x 1
    vector<double> g_;     // gradients                   N|E| x 1
    vector<double> H_;     // hessians                  NxN|E| x 1
    vector<double> Hinv_;  // hessian inverse           NxN|E| x 1
    vector<double> dSdF_;  // gradient of S w.r.t. F    MxN|E| x 1
    vector<double> Aloc_;  // local stiffness matrices
    vector<double> Jloc_;  // local jacobians
    vector<double> vols_;  // element volumes |E| x 1

    SparseMatrixGpu J_gpu_;
    MatrixBatchInverseGpu<N()> Hinv_gpu_;

  };
}