#pragma once

#include "EigenTypes.h"
#include <thrust/device_vector.h>

namespace mfem {

  /// @brief Block Compressed sparse row matrix for FEM assembly. Each block is
  /// DIMxDIM, and the matrix is N*DIM x N*DIM, where N is the number of nodes
  ///
  /// Useful for FEM stiffness matrices, where each element contributes
  /// multiple DIMxDIM subblocks to the global matrix.
  ///
  /// @tparam Scalar {float, double}
  /// @tparam DIM size of subblocks
  /// @tparam N block size
  template <typename Scalar, int DIM, int N>
  class BlockMatrix {

    using MatD = Eigen::Matrix<Scalar, DIM, DIM>;

  public:

    /// @brief Initialize assembler / analyze sparsity of system
    /// @param E - elements nelem x 4 for tetrahedra
    /// @param free_map - |nnodes| maps node to its position in unpinned vector
    BlockMatrix(const Eigen::MatrixXi& E, const std::vector<int>& free_map);

    // Update entries of matrix using per-element blocks
    // blocks   - |nelem| N*DIM x N*DIM blocks to update assembly matrix
    void update_matrix(const thrust::device_vector<double>& blocks);

    /// @brief Convert to Eigen CSR matrix
    /// @return row major Eigen sparse matrix reference
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor>& to_eigen_csr();

    /// @brief Extract diagonal blocks from the matrix, adding the entries to diag
    /// @param diag - |nnodes| x DIM x DIM diagonal blocks
    void extract_diagonal(double* diag);

    struct update_functor {

      update_functor(const double* _blocks, const int* _block_indices,
          MatD* _blocks_with_duplicates) :
        blocks(_blocks), block_indices(_block_indices),
        blocks_with_duplicates(_blocks_with_duplicates) {}

      void operator()(int i);
      
      const double* blocks;
      const int* block_indices;
      MatD* blocks_with_duplicates;
    };

    struct update_functor2 {

      update_functor2(const double* _blocks, const int* _block_indices,
          MatD* _blocks_with_duplicates, int _nelem) :
        blocks(_blocks), block_indices(_block_indices),
        blocks_with_duplicates(_blocks_with_duplicates), nelem(_nelem) {}

      void operator()(int i);
      
      const double* blocks;
      const int* block_indices;
      MatD* blocks_with_duplicates;
      int nelem;
    };

    struct pairs_functor {

      pairs_functor(int* _E_d, int* _free_map_d, int* _block_row_indices,
          int* _block_col_indices, int* _block_is_free)
          : E_d(_E_d), free_map_d(_free_map_d),
          block_row_indices(_block_row_indices),
          block_col_indices(_block_col_indices),
          block_is_free(_block_is_free)
          {}

      void operator()(int i);

      int * E_d;
      int * free_map_d;
      int * block_row_indices;
      int * block_col_indices;
      int * block_is_free;
    };

    struct zip_comparator {
      __host__ __device__
      bool operator() (const thrust::tuple<int, int>& a,
        const thrust::tuple<int, int>& b) const {
        if(a.head < b.head) return true;
        if(a.head == b.head) return a.tail < b.tail;
        return false;
      }
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

    const int* row_offsets() {
      return thrust::raw_pointer_cast(row_offsets_.data());
    }

    const int* col_indices() {
      return thrust::raw_pointer_cast(col_indices_.data());
    }

    const double* values() { 
      return reinterpret_cast<double *>(thrust::raw_pointer_cast(blocks_.data()));
    }
    
    int num_row_blocks() const {
      return nnodes_ + 1;
    }
    int num_col_indices() const {
      return col_indices_.size();
    }
    int num_values() const {
      return blocks_.size() * DIM * DIM;
    }


    int num_blocks() const {
      return blocks_.size();
    }

    int size() const {
      return nnodes_ * DIM;
    }
  protected:

    /// @brief Eigen sparse matrix for conversion to CSR
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> A_;

    /// @brief All DIMxDIM blocks, row major sorted.
    thrust::device_vector<MatD> blocks_with_duplicates_;

    /// @brief Map from local hessian block to sorted block index in
    /// block_with_duplicates_ vector
    thrust::device_vector<int> block_indices_;

    /// @brief Row indices of each block
    thrust::device_vector<int> block_row_indices_;

    /// @brief Column indices of each block
    thrust::device_vector<int> block_col_indices_;

    /// @brief Column indices of each (unique) block
    thrust::device_vector<int> col_indices_;

    /// @brief Row offsets of each (unique) block
    thrust::device_vector<int> row_offsets_;

    /// @brief Compressed set of DIMxDIM blocks, corresponding to a reduction
    /// over the duplicate entries.
    thrust::device_vector<MatD> blocks_;

    thrust::device_vector<int> row_indices_tmp_;

    /// @brief Number of elements
    int nelem_;

    /// @brief Number of nodes (matrix has nnodes_ x nnodes_ blocks)
    int nnodes_;
  };
}