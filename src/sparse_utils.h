#pragma once

#include <EigenTypes.h>

namespace mfem {

  // Class for parallel assembly of FEM stiffness matrices
  // Each element's input is a block of size NxN composed of MxM sub-blocks
  // these sub-blocks are scattered to their global nodes positions and
  // summed with duplicates.
  template <typename Scalar, int N, int M>
  class Assembler {

    // Initialize assembler / analyze sparsity of system
    // E        - elements nelem x 4 for tetrahedra
    // free_map - |nnodes| maps node to its position in unpinned vector
    //            equals -1 if node is pinned
    Assembler(const Eigen::MatrixXd& E, const std::vector<int> free_map);

    // Element IDs, global, and local coordinates. Each of these vectors
    // is of the same size.
    std::vector<int> element_ids;
    std::vector<std::pair<int,int>> global_pairs;
    std::vector<std::pair<short,short>> local_pairs;

    int num_nodes; // number of unique pairs / blocks in matrix
    std::vector<short> multiplicity; // number of pairs to sum over for a node
    std::vector<int> row_offsets;
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> A;

  };

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
  //void kkt_lhs(const Eigen::SparseMatrixd& M,
  //    const Eigen::SparseMatrix<double,Eigen::RowMajor>& Jw,
  //    double ih2, std::vector<Eigen::Triplet<double>>& trips);

  //void diagonal_compliance(const Eigen::VectorXd& vols, double mu, int offset,
  //    std::vector<Eigen::Triplet<double>>& trips);

  //void init_compliance_blocks(int N, int offset,
  //    std::vector<Eigen::Triplet<double>>& trips);

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

  // Builds a block symmetric matrix of the form
  // P = [A 0; 0 C] where C is block diagonal
  template <int N>
  void fill_block_matrix(const Eigen::SparseMatrixd& A,
      const std::vector<Eigen::Matrix<double, N, N>>& C,
      Eigen::SparseMatrixd& mat) {
    
    using namespace Eigen;
    int m = N * C.size();
    mat.resize(A.rows()+m, A.rows()+m);
    std::vector<Triplet<double>> trips;

    // Mass matrix terms
    for (int i = 0; i < A.outerSize(); ++i) {
      for (SparseMatrixd::InnerIterator it(A,i); it; ++it) {
        trips.push_back(Triplet<double>(it.row(),it.col(),it.value()));
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

  template <int N>
  void fill_asym_block_matrix(const Eigen::SparseMatrixd& A,
      const Eigen::SparseMatrixd& B,
      const std::vector<Eigen::Matrix<double, N, N>>& C,
      Eigen::SparseMatrixd& mat) {
    
    using namespace Eigen;

    mat.resize(A.rows()+B.cols(), A.rows()+B.cols());
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
        trips.push_back(Triplet<double>(it.row(),offset+it.col(),it.value()));
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

  template <typename Scalar>
  void fill_block_matrix(const Eigen::SparseMatrixd& A,
      const Eigen::SparseMatrixd& B, Eigen::SparseMatrix<Scalar>& mat) {
    
    using namespace Eigen;
    mat.resize(A.rows()+B.rows(), A.cols()+B.cols());
    std::vector<Triplet<double>> trips;

    // Mass matrix terms
    for (int i = 0; i < A.outerSize(); ++i) {
      for (SparseMatrixd::InnerIterator it(A,i); it; ++it) {
        trips.push_back(Triplet<double>(it.row(),it.col(),it.value()));
      }
    }

    int offset = A.rows();
    for (int i = 0; i < B.outerSize(); ++i) {
      for (SparseMatrixd::InnerIterator it(B,i); it; ++it) {
        trips.push_back(Triplet<double>(offset + it.row(),
            offset + it.col(),it.value()));
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
