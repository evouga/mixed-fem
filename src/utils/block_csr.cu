#include "block_csr.h"
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

using namespace Eigen;
using namespace mfem;

typedef thrust::tuple<int, int, int> Tuple;

namespace {

  struct is_not_free {
    __host__ __device__
    bool operator()(const Tuple& tuple) const {
      return tuple.get<0>() == -1 || tuple.get<1>() == -1;
    }
  };
}

template<typename Scalar, int DIM, int N> __device__
void BlockMatrix<Scalar,DIM,N>::pairs_functor::operator()(int i) {
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < N; ++k) {
      // Get the global index of the block
      int block_index = i * N * N + j * N + k;

      // Get the global index of the row and column
      int row = E_d[i * N + j];
      int col = E_d[i * N + k];

      // Check if the row and column are free
      int row_is_free = free_map_d[row] >= 0;
      int col_is_free = free_map_d[col] >= 0;

      // If both are free, set the block to free
      block_is_free[block_index] = row_is_free && col_is_free;

      // Set the row and column indices
      block_row_indices[block_index] = free_map_d[row];
      block_col_indices[block_index] = free_map_d[col];
    }
  }
}

template<typename Scalar, int DIM, int N> __device__
void BlockMatrix<Scalar,DIM,N>::update_functor::operator()(int i) {
  // i is the index of the block in the matrix
  // block_indices[i] is the index of the block in the sorted list of blocks
  int idx = block_indices[i];
  if (idx < 0) return;

  // Copy blocks to blocks_with_duplicates MatD
  MatD& block = blocks_with_duplicates[idx];
  for (int j = 0; j < DIM; ++j) {
    for (int k = 0; k < DIM; ++k) {
      block(j, k) = blocks[i * DIM * DIM + j * DIM + k];
    }
  }
}


template<typename Scalar, int DIM, int N>
BlockMatrix<Scalar,DIM,N>::BlockMatrix(const Eigen::MatrixXi& E,
    const std::vector<int>& free_map) : nelem_(E.rows()) {

  // Row and column index for each block in the matrix.
  // and a vector indicating if this block is free or not
  block_row_indices_.resize(E.rows() * N * N);
  block_col_indices_.resize(E.rows() * N * N);
  thrust::device_vector<int> block_is_free(E.rows() * N * N);

  // ----------------------------------------------------------------------- //
  // 1. Create row, col index values for each DIMxDIM block in the matrix
  // ----------------------------------------------------------------------- //

  // Write element matrix to device. Transpose first for better memory access.
  MatrixXi E_t = E.transpose();
  thrust::device_vector<int> E_d(E_t.data(), E_t.data() + E.rows() * E.cols());
  thrust::device_vector<int> free_map_d(free_map); // free_map to device memory

  // Get maximum value in free_map
  nnodes_ = *thrust::max_element(free_map_d.begin(), free_map_d.end());

  // Fill in the block row, column, and is_free vectors from all pair-wise
  // combinations of nodes in each element.
  thrust::for_each(thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(E.rows()),
      pairs_functor(
        thrust::raw_pointer_cast(E_d.data()),
        thrust::raw_pointer_cast(free_map_d.data()),
        thrust::raw_pointer_cast(block_row_indices_.data()),
        thrust::raw_pointer_cast(block_col_indices_.data()),
        thrust::raw_pointer_cast(block_is_free.data())));

  // sorted block to element block map
  thrust::device_vector<int> sorted_block_map(E.rows() * N * N);
  thrust::sequence(sorted_block_map.begin(), sorted_block_map.end());

  // Block row & col indices may contain -1 values for non-free blocks. The
  // block_map,row,and column tuple for these values need to be removed
  // from the list of blocks.
  auto new_end = thrust::remove_if(
      thrust::make_zip_iterator(
          thrust::make_tuple(
              block_row_indices_.begin(),
              block_col_indices_.begin(),
              sorted_block_map.begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(
              block_row_indices_.end(),
              block_col_indices_.end(),
              sorted_block_map.end())), is_not_free());

  // Erase the removed elements using new_end
  auto Tend =  new_end.get_iterator_tuple();;
  block_row_indices_.erase(thrust::get<0>(Tend), block_row_indices_.end());
  block_col_indices_.erase(thrust::get<1>(Tend), block_col_indices_.end());
  sorted_block_map.erase(thrust::get<2>(Tend), sorted_block_map.end());
  blocks_with_duplicates_.resize(block_row_indices_.size());

  // ----------------------------------------------------------------------- //
  // 2. Row major sort block indices and block map
  // ----------------------------------------------------------------------- //

  // Sort by key of the form (block_row_indices, block_col_indices) with
  // values sorted_block_map
  thrust::sort_by_key(
      thrust::make_zip_iterator(thrust::make_tuple(
          block_row_indices_.begin(),
          block_col_indices_.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(
          block_row_indices_.end(),
          block_col_indices_.end())),
      sorted_block_map.begin(), zip_comparator());

  // For each block in the matrix, get the index of the block in the sorted
  // list of blocks
  block_indices_.resize(E.rows() * N * N, -1.0);
  thrust::device_vector<int> seq(E.rows() * N * N);
  thrust::sequence(seq.begin(), seq.end());
  thrust::gather(sorted_block_map.begin(), sorted_block_map.end(),
      seq.begin(), block_indices_.begin());

  // ----------------------------------------------------------------------- //
  // 2. Building row offsets
  // ----------------------------------------------------------------------- //

  // copies of sorted block_row_indices and block_col_indices
  thrust::device_vector<int> row_indices = block_row_indices_;
  col_indices_ = block_col_indices_;

  // Find the unique row-col pairs
  // NOTE: these are the final row and column indices for the matrix
  auto iter = thrust::unique(
      thrust::make_zip_iterator(thrust::make_tuple(
          row_indices.begin(),
          col_indices_.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(
          row_indices.end(),
          col_indices_.end())));
  
  // Resize unique_rows and unique_cols to the new size
  auto Tend2 = iter.get_iterator_tuple();
  row_indices.resize(thrust::get<0>(Tend2) - row_indices.begin());
  col_indices_.resize(thrust::get<1>(Tend2) - col_indices_.begin());
  blocks_.resize(row_indices.size());

  // Now count the number of occurrences of each row value
  // TODO this will not work if num of offsets is not equal to matrix rows
  // Need to do a prefix sum on the row_offsets to get the final offsets
  row_offsets_.resize(row_indices.size());

  // Count the number of matching row indices and write to row_offsets
  // Returns pair of iterators to the end of the (indices and offsets)
  auto pair_end = thrust::reduce_by_key(row_indices.begin(), row_indices.end(),
      thrust::constant_iterator<int>(1), row_indices.begin(), row_offsets_.begin());

  // Now compute the row offsets by doing a prefix sum
  row_offsets_.resize(pair_end.second - row_offsets_.begin() + 1);
  thrust::inclusive_scan(row_offsets_.begin(), pair_end.second, row_offsets_.begin()+1);
}

template<typename Scalar, int DIM, int N>
void BlockMatrix<Scalar,DIM,N>::update_matrix(
    const thrust::device_vector<double>& blocks) {

  // Blocks is vector of size |E| x (N*N*DIM*DIM)
  // For each DIMxDIM block, write to blocks_with_duplicates using block_indices
  thrust::for_each(thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(nelem_*N*N),
      update_functor(
        thrust::raw_pointer_cast(blocks.data()),
        thrust::raw_pointer_cast(block_indices_.data()),
        thrust::raw_pointer_cast(blocks_with_duplicates_.data())));

  // Using sorted block_row_indices and block_col_indices, do a reduction
  // over the duplicates
  thrust::device_vector<int> row_indices = block_row_indices_;

  thrust::reduce_by_key(
      thrust::make_zip_iterator(thrust::make_tuple(
          block_row_indices_.begin(),
          block_col_indices_.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(
          block_row_indices_.end(),
          block_col_indices_.end())),
      blocks_with_duplicates_.begin(),
      thrust::make_zip_iterator(thrust::make_tuple(
          row_indices.begin(),
          col_indices_.begin())),
      blocks_.begin());
}

template<typename Scalar, int DIM, int N>
Eigen::SparseMatrix<Scalar, Eigen::RowMajor>&
BlockMatrix<Scalar,DIM,N>::to_eigen_csr() {

  thrust::host_vector<int> h_row_offsets = row_offsets_;
  thrust::host_vector<int> h_col_indices = col_indices_;
  thrust::host_vector<MatD> h_blocks = blocks_;

  // Convert to Eigen
  A_.resize(nnodes_*DIM, nnodes_*DIM);

  for(int i=0; i<h_row_offsets.size()-1; i++) {
    for(int j=h_row_offsets[i]; j<h_row_offsets[i+1]; j++) {
      int row = i;
      int col = h_col_indices[j];
      MatD block = h_blocks[j];
      for(int k=0; k<DIM; k++) {
        for(int l=0; l<DIM; l++) {
          A_.insert(row*DIM+k, col*DIM+l) = block(k,l);
        }
      }
    }
  }
  return A_;
}

template class mfem::BlockMatrix<double, 3, 4>;