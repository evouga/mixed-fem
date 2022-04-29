#pragma once

#include <EigenTypes.h>

namespace mfem {
  // Initializes the LHS for KKT system. Compliance block initialized with
  // diagonal 1/mu entries.
  //
  // M      - mass matrix
  // Jw     - volume weighted jacobian
  // vols   - per-element volumes
  // mu     - lame parameter
  // ih2    - inverse squared timestep
  // trips  - output triplets for matrix assembly
  //
  // Outputs into vector for A,B,B^T,C blocks for the kkt of form [A B^T; B C]
  void kkt_lhs(const Eigen::SparseMatrixd& M,
      const Eigen::SparseMatrix<double,Eigen::RowMajor>& Jw,
      double ih2, std::vector<Eigen::Triplet<double>>& trips);

  void diagonal_compliance(const Eigen::VectorXd& vols, double mu, int offset,
      std::vector<Eigen::Triplet<double>>& trips);

  void init_compliance_blocks(int N, int offset,
      std::vector<Eigen::Triplet<double>>& trips);

  // Builds a block symmetric matrix of the form
  // P = [A B^T; B C] where C is block diagonal
  template <int N>
  void fill_block_matrix(const Eigen::SparseMatrixd& A,
      const Eigen::SparseMatrixd& B,
      const std::vector<Eigen::Matrix<double, N, N>>& C,
      Eigen::SparseMatrixd& mat) {
    
    using namespace Eigen;

    mat.resize(A.rows()+B.rows(), A.rows()+B.rows());
    std::vector<Triplet<double>> trips;

    // Mass matrix terms
    for (int i = 0; i < A.outerSize(); ++i) {
      for (SparseMatrixd::InnerIterator it(A,i); it; ++it) {
        trips.push_back(Triplet<double>(it.row(),it.col(),it.value()));
      }
    }

    int offset = A.rows(); // offset for off diagonal blocks

    // Jacobian off-diagonal entries
    for (int i = 0; i < B.outerSize(); ++i) {
      for (SparseMatrixd::InnerIterator it(B, i); it; ++it) {
        trips.push_back(Triplet<double>(offset+it.row(),it.col(),it.value()));
        trips.push_back(Triplet<double>(it.col(),offset+it.row(),it.value()));
      }
    }

    // Compliance block entries
    for (int i = 0; i < C.size(); ++i) {
      
      int offset = A.rows() + i * N;

      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
          trips.push_back(Triplet<double>(offset+j, offset+k, C[i](j,k)));
        }
      }

    }
    mat.setFromTriplets(trips.begin(), trips.end());

  }

  template <int R, int C>
  void init_block_diagonal(Eigen::SparseMatrixd& mat, int N) {
    mat.resize(R*N, C*N);
    mat.reserve(Eigen::VectorXi::Constant(C*N,R));

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < C; ++j) {
        for (int k = 0; k < R; ++k) {
          mat.insert(R*i + k, C*i + j) = 0;
        }
      }
    }
  }

  template <int R, int C>
  void update_block_diagonal(std::vector<Eigen::Matrix<double, R, C>> data,
      Eigen::SparseMatrixd& mat) {

    int N = data.size();
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
      
      int start = R*C*i;

      for (int j = 0; j < C; ++j) {
        for (int k = 0; k < R; ++k) {
          mat.valuePtr()[start + j*R + k] = data[i](k,j);
        }
      }
    }
  }
}