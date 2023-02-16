#pragma once

#include "EigenTypes.h"
#include "mixed_variable.h"
#include <thrust/device_vector.h>
#include "utils/mixed_ipc.h"
#include "utils/block_csr.h"
#include "utils/sparse_utils.h"
#include <ginkgo/ginkgo.hpp>
#include "ipc/friction/friction.hpp"

namespace mfem {

  class SimConfig;

  template<int DIM, StorageType STORAGE>
  class MixedFrictionGpu : public MixedVariable<DIM,STORAGE> {

    typedef MixedVariable<DIM,STORAGE> Base;

    template <typename T>
    using vector = thrust::device_vector<T>;

    __host__ __device__
    static constexpr int Aloc_N() {
      return DIM == 3 ? (DIM*4) : DIM*3;
    }
    
  public:
    using typename Base::VectorType;

    MixedFrictionGpu(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<SimConfig> config);

    static std::string name() {
      return "mixed-friction-gpu";
    }

    int size() const override {
      return constraints_.size();
    }

    int size_dual() const override {
      return constraints_.size();
    }

    const Eigen::SparseMatrix<double, Eigen::RowMajor>& lhs() override {
      if (constraints_.size() == 0) {
        dummy_A_ = Eigen::SparseMatrix<double, Eigen::RowMajor>();
        dummy_A_.resize(mesh_->jacobian().rows(),mesh_->jacobian().rows());
        return dummy_A_;
      }
      std::cout << "LHS norm: " << assembler_->to_eigen_csr().norm() << std::endl;
      return assembler_->to_eigen_csr();
    }

    double energy(VectorType& x, VectorType& d) override;

    void update(VectorType& x, double dt) override;

    void pre_solve() override;

    void reset() override;
    VectorType& rhs() override;
    VectorType gradient() override;
    VectorType& delta() override;
    VectorType& value() override;
    VectorType& lambda() override;
    void solve(VectorType& dx) override;

    void apply_submatrix(double* x, const double* b, int cols, int start, int end);

    Eigen::VectorXd gradient_mixed() override { return dummy_; }
    Eigen::VectorXd gradient_dual() override { return dummy_; }
    
    void set_executor(std::shared_ptr<const gko::Executor> exec) {
      exec_ = exec;
    }

    std::shared_ptr<BlockMatrix<double,DIM,4>> assembler() {
      return assembler_;
    }

    struct derivative_functor {

      derivative_functor(double* _H, double* _Gx, double* _Aloc)
          : H(_H), Gx(_Gx), Aloc(_Aloc) {}
      
      void operator()(int i) const;

      double* H;
      double* Gx;
      double* Aloc;
    };

    struct rhs_functor {

      rhs_functor(int* _T, int* _free_map, double* _d, double* _D, double* _g,
          double* _H, double* _Gx, double* _rhs)
          : T(_T), free_map(_free_map), d(_d), D(_D), g(_g), H(_H),
            Gx(_Gx), rhs(_rhs) {}
      
      void operator()(int i) const;

      int* T;
      int* free_map;
      double* d;
      double* D;
      double* g;
      double* H;
      double* Gx;
      double* rhs;
    };

    struct solve_functor {

      solve_functor(int* _T, int* _free_map, double* _d, double* _D, double* _g,
          double* _H, double* _Gx, double* _la, double* _delta, double* _dx) 
          : T(_T), free_map(_free_map), d(_d), D(_D), g(_g), H(_H), Gx(_Gx),
            la(_la), delta(_delta), dx(_dx) {}

      void operator()(int i) const;

      int* T;
      int* free_map;
      double* d;
      double* D;
      double* g;
      double* H;
      double* Gx;
      double* la;
      double* delta;
      double* dx;
    };

  protected:

    using Base::mesh_;
    std::shared_ptr<SimConfig> config_;

    // Host data
    double dt_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> dummy_A_;
    Eigen::VectorXd dummy_;
    Eigen::VectorXd rhs_h_;
    Eigen::VectorXd delta_h_;
    Eigen::VectorXd z_h_;
    Eigen::VectorXd Z_h_;
    Eigen::VectorXd la_h_;
    Eigen::MatrixXi T_h_;  // element map |E| x 4
    Eigen::VectorXd Gx_h_; // constraint jacobian w.r.t x
    Eigen::VectorXd g_h_;     // gradients                   |E| x 1
    Eigen::VectorXd H_h_;     // hessians                    |E| x 1

    // Device data
    vector<double> z_;     // deformation variable        |E| x 1
    vector<double> Z_;     // deformation variable        |E| x 1
    vector<double> delta_; // deformation variable deltas |E| x 1
    vector<double> la_;    // lagrange multipliers        |E| x 1
    vector<double> g_;     // gradients                   |E| x 1
    vector<double> H_;     // hessians                    |E| x 1
    vector<double> rhs_; // RHS for primal condensation system
    vector<double> Gx_; // constraint jacobian w.r.t x
    vector<double> Aloc_;  // local stiffness matrices
    vector<int> T_;   // element map |E| x 4
    vector<int> free_map_;

    Eigen::MatrixXd V0_;    // Vertex positions at beginning of lagged step

    ipc::FrictionConstraints constraints_;
    std::shared_ptr<BlockMatrix<double,DIM,4>> assembler_;
    std::shared_ptr<const gko::Executor> exec_; // ginkgo executor
    std::shared_ptr<gko::matrix::Csr<double, int>> A_;  // ginkgo matrix
   
  };
}