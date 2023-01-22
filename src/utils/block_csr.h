#pragma once

#include "EigenTypes.h"
#include <thrust/device_vector.h>

namespace mfem {

  // Each block is composed of NxN element matrices with each entry being a
  // DIMxDIM block.
  // Our block matrix needs to support duplicate entries which are sorted in
  // block CSR format.
  template <typename Scalar, int DIM, int N>
  class BlockMatrix {

    // Returns the size of the local blocks for assembly. If N is dynamic
    // M() returns -1
    static constexpr int M() {
      if (N == -1) {
        return -1;
      } else {
        return DIM * N;
      }
    }

    using MatM  = Eigen::Matrix<Scalar, M(), M()>;

  public:

    using MatD = Eigen::Matrix<Scalar, DIM, DIM>;

    // Initialize assembler / analyze sparsity of system
    // None of this wonderfully optimized since we only have to do it once
    // E        - elements nelem x 4 for tetrahedra
    // free_map - |nnodes| maps node to its position in unpinned vector
    //            equals -1 if node is pinned
    BlockMatrix(const Eigen::MatrixXi& E, const std::vector<int>& free_map);

    // Update entries of matrix using per-element blocks
    // blocks   - |nelem| N*DIM x N*DIM blocks to update assembly matrix
    // void update_matrix(const std::vector<MatM>& blocks);

    // Element IDs, global, and local coordinates. Each of these vectors
    // is of the same size.
    // std::vector<int> element_ids;
    // std::vector<std::pair<int,int>> global_pairs;
    // std::vector<std::pair<int,int>> local_pairs;

    // std::vector<int> multiplicity; // number of pairs to sum over for a node
    // std::vector<int> row_offsets;  // each entry is index into new row
    // std::vector<int> offsets;      // unique pairs
    // Eigen::SparseMatrix<Scalar, Eigen::RowMajor> A;

    thrust::device_vector<MatD> blocks_;
    thrust::device_vector<int> col_indices_;
    thrust::device_vector<int> row_offsets_;
  };
}