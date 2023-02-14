#pragma once

#include <thrust/device_vector.h>
#include "EigenTypes.h"
#include "mixed_variable.h"
#include <cusparse.h>
#include "utils/sparse_matrix_gpu.h"
#include "utils/sparse_utils.h"
#include "utils/block_csr.h"
#include <ginkgo/ginkgo.hpp>

namespace mfem {

  template<int DIM, StorageType STORAGE>
  class MixedStretchGpu : public MixedVariable<DIM,STORAGE> {

    typedef MixedVariable<DIM,STORAGE> Base;

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

    __host__ __device__
    static constexpr int Aloc_N() {
      return DIM == 3 ? (DIM*4) : DIM*3;
    }

    // Matrix and vector data types
    using MatD  = Eigen::Matrix<double, DIM, DIM>; // 3x3 or 2x2
    using VecN  = Eigen::Vector<double, N()>;      // 6x1 or 3x1
    using VecM  = Eigen::Vector<double, M()>;      // 9x1
    using MatM  = Eigen::Matrix<double, M(), M()>; // 9x9
    using MatN  = Eigen::Matrix<double, N(), N()>; // 6x6 or 3x3
    using MatMN = Eigen::Matrix<double, M(), N()>; // 9x6 or 4x3

    __host__ __device__
    static constexpr MatN Sym() {
      VecN v;
      MatN m; m.setZero();
      // NOTE: asDiagonal returning NaNs in cuda
      if constexpr (DIM == 3) {
        v << 1,1,1,2,2,2;
      } else {
        v << 1,1,2;
      }
      m.diagonal() = v;
      return m;
    }

    __host__ __device__
    static constexpr MatN Syminv() {
      VecN v;
      MatN m; m.setZero();
      // NOTE: asDiagonal returning NaNs in cuda
      if constexpr (DIM == 3) {
        v << 1,1,1,0.5,0.5,0.5;
      } else {
        v << 1,1,0.5;
      }
      m.diagonal() = v;
      return m;
    }


  public:

    using typename Base::VectorType;

    MixedStretchGpu(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<SimConfig> config);

    static std::string name() {
      return "mixed-stretch-gpu";
    }

    void reset();
    double energy(VectorType& x, VectorType& s) override;
    double constraint_value(const VectorType& x,
        const VectorType& s) override{return 0.0;}
    void update(VectorType& x, double dt) override;
    void solve(VectorType& dx) override;
    void post_solve() override;

    const Eigen::SparseMatrix<double, Eigen::RowMajor>& lhs() override {
      return assembler2_->to_eigen_csr();
    }
    VectorType& rhs() override;
    VectorType gradient() override;
    
    Eigen::VectorXd gradient_mixed() override { return dummy_; }
    Eigen::VectorXd gradient_dual() override { return dummy_; }

    VectorType& delta();
    VectorType& value() override;
    VectorType& lambda() override;
    
    int size() const override {
      return s_.size();
    }

    int size_dual() const override {
      return 0;//la_.size() * N();
    }

    void apply(double* x, const double* b, int cols) override {}
    void apply_submatrix(double* x, const double* b, int cols, int start, int end);
    void free_matrix() {
      if (exec_ != nullptr)
        A_ = gko::matrix::Csr<double,int>::create(exec_);
    }

    void extract_diagonal(double* diag) override;

    void set_executor(std::shared_ptr<const gko::Executor> exec) {
      exec_ = exec;
    }
    
    std::shared_ptr<BlockMatrix<double,DIM,4>> assembler() {
      return assembler2_;
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

    struct energy_functor {

      energy_functor(double* _s, double* _F, double* _la, 
          double* _vols, double* _mu, double* _lambda)
        : s(_s), F(_F), la(_la), vols(_vols), mu(_mu), lambda(_lambda) {}
        
      double operator()(int i) const;

      double* s;
      double* F;
      double* la;
      double* vols;
      double* mu;
      double* lambda;
    };

    template <bool COMPUTE_GRADIENTS = false>
    struct rotation_functor {

      rotation_functor(double* _F, double* _R, double* _S,
          double* _dSdF)
        : F(_F), R(_R), S(_S), dSdF(_dSdF) {}
      
      void operator()(int i) const;

      double* F;
      double* R;
      double* S;
      double* dSdF;
    };

    struct derivative_functor {

      derivative_functor(double* _s, double* _g, double* _H,
          double* _dSdF, double* _Jloc, double* _Aloc, double* _vols,
          double* _mu, double* _lambda)
        : s(_s), g(_g), H(_H), dSdF(_dSdF), Jloc(_Jloc),
          Aloc(_Aloc), vols(_vols), mu(_mu), lambda(_lambda) {}
      
      void operator()(int i) const;

      double* s;
      double* g;
      double* H;
      double* dSdF;
      double* Jloc;
      double* Aloc;
      double* vols;
      double* mu;
      double* lambda;
    };

    struct rhs_functor {

      rhs_functor(double* _rhs, double* _s, double* _S, double* _g, double* _H,
          double* _dSdF, double* _vols)
        : rhs(_rhs), s(_s), S(_S), g(_g), H(_H), dSdF(_dSdF),
          vols(_vols) {}
      
      void operator()(int i) const;

      double* rhs;
      double* s;
      double* S;
      double* g;
      double* H;
      double* dSdF;
      double* vols;
    };

    struct solve_functor {

      solve_functor(double* _Jdx, double* _S, double* _s, double* _g, double* _H,
          double* _dSdF, double* _ds, double* _la, double* _vols)
        : Jdx(_Jdx), S(_S), s(_s), g(_g), H(_H), dSdF(_dSdF),
          ds(_ds), la(_la), vols(_vols) {}

      void operator()(int i) const;
  
      double* Jdx;
      double* S;
      double* s;
      double* g;
      double* H;
      double* dSdF;
      double* ds;
      double* la;
      double* vols;
    };

    struct extract_diagonal_functor {
      double* diag;
      const double* values;
      const int* row_offsets;
      const int* col_indices;

      extract_diagonal_functor(double* _diag, const double* _values,
          const int* _row_offsets, const int* _col_indices)
        : diag(_diag), values(_values), row_offsets(_row_offsets),
          col_indices(_col_indices) {}

      void operator()(int i) const;
    };

  protected:

    // static double local_energy(const VecN& S, double mu);
    // static VecN local_gradient(const VecN& S, double mu);
    // static MatN local_hessian(const VecN& S, double mu);
    
    using Base::mesh_;

    int nelem_;          // number of elements, |E|

    struct params {
      vector<double> mu;
      vector<double> lambda;
    } params_;

    Eigen::SparseMatrix<double, Eigen::RowMajor> dummy_A_;
    Eigen::VectorXd dummy_;
    Eigen::VectorXd rhs_h_;
    vector<double> rhs_; // RHS for primal condensation system
    vector<double> rhs_tmp_;
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
    Eigen::VectorXd ds_h_;
    Eigen::VectorXd la_h_;
    Eigen::VectorXd s_h_;
    vector<double> energy_tmp_;

    SparseMatrixGpu J_gpu_;
    SparseMatrixGpu JW_gpu_; // projected, weighted jacobian

    // TODO should just do transpose-multiply
    SparseMatrixGpu JWT_gpu_; // projected, weighted jacobian, transposed
    MatrixBatchInverseGpu<N()> psd_fixer_;
    std::shared_ptr<Assembler<double,DIM,-1>> assembler_;
    std::shared_ptr<BlockMatrix<double,DIM,4>> assembler2_;

    std::shared_ptr<const gko::Executor> exec_; // ginkgo executor
    std::shared_ptr<gko::matrix::Csr<double, int>> A_;  // ginkgo matrix
  };
}