#include "block_csr.h"
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/iterator/constant_iterator.h>

using namespace Eigen;
using namespace mfem;

typedef thrust::tuple<int, int> Tuple;

namespace {

  struct is_not_free {
    __host__ __device__
    bool operator()(const Tuple& tuple) const {
      return tuple.get<0>() == -1 || tuple.get<1>() == -1;
    }
  };

  template<int N>
  struct pairs_functor {

    pairs_functor(int* _E_d, int* _free_map_d, int* _block_row_indices,
        int* _block_col_indices, int* _block_is_free)
        : E_d(_E_d), free_map_d(_free_map_d),
        block_row_indices(_block_row_indices),
        block_col_indices(_block_col_indices),
        block_is_free(_block_is_free)
        {}

    __host__ __device__
    void operator()(int i) const {
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
    int * E_d;
    int * free_map_d;
    int * block_row_indices;
    int * block_col_indices;
    int * block_is_free;
  };

}

template<typename Scalar, int DIM, int N>
BlockMatrix<Scalar,DIM,N>::BlockMatrix(const Eigen::MatrixXi& E,
    const std::vector<int>& free_map) {

  // Row and column index for each block in the matrix.
  // and a vector indicating if this block is free or not
  thrust::device_vector<int> block_row_indices(E.rows() * N * N);
  thrust::device_vector<int> block_col_indices(E.rows() * N * N);
  thrust::device_vector<int> block_is_free(E.rows() * N * N);

  // ----------------------------------------------------------------------- //
  // 1. Create row, col index values for each DIMxDIM block in the matrix
  // ----------------------------------------------------------------------- //

  // Write element matrix to device. Transpose first for better memory access.
  MatrixXi E_t = E.transpose();
  thrust::device_vector<int> E_d(E_t.data(), E_t.data() + E.rows() * E.cols());
  thrust::device_vector<int> free_map_d(free_map); // free_map to device memory

  // Fill in the block row, column, and is_free vectors from all pair-wise
  // combinations of nodes in each element.
  thrust::for_each(thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(E.rows()),
      pairs_functor<N>(E_d.data().get(), free_map_d.data().get(),
        block_row_indices.data().get(), block_col_indices.data().get(),
        block_is_free.data().get()));

  // sorted block to element block map
  thrust::device_vector<int> sorted_block_map(E.rows() * N * N);
  thrust::sequence(sorted_block_map.begin(), sorted_block_map.end());

  // Block row & col indices may contain -1 values for non-free blocks. The
  // block_map,row,and column tuple for these values need to be removed
  // from the list of blocks.
  auto new_end = thrust::remove_if(
      thrust::make_zip_iterator(
          thrust::make_tuple(
              block_row_indices.begin(),
              block_col_indices.begin(),
              sorted_block_map.begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(
              block_row_indices.end(),
              block_col_indices.end(),
              sorted_block_map.end())), is_not_free());

  // Erase the removed elements using new_end
  auto Tend =  new_end.get_iterator_tuple();;
  block_row_indices.erase(thrust::get<0>(Tend), block_row_indices.end());
  block_col_indices.erase(thrust::get<1>(Tend), block_col_indices.end());
  sorted_block_map.erase(thrust::get<2>(Tend), sorted_block_map.end());

  // ----------------------------------------------------------------------- //
  // 2. Row major sort block indices and block map
  // ----------------------------------------------------------------------- //

  // Sort by key of the form (block_row_indices, block_col_indices) with
  // values sorted_block_map
  thrust::sort_by_key(
      thrust::make_zip_iterator(
          thrust::make_tuple(
              block_row_indices.begin(),
              block_col_indices.begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(
              block_row_indices.end(),
              block_col_indices.end())),
      sorted_block_map);

  // For each block in the matrix, get the index of the block in the sorted
  // list of blocks
  thrust::device_vector<int> block_indices(E.rows() * N * N, -1.0);
  thrust::device_vector<int> seq(E.rows() * N * N);
  thrust::sequence(seq.begin(), seq.end());
  thrust::gather(sorted_block_map.begin(), sorted_block_map.end(),
      seq.begin(), block_indices.begin());

  // ----------------------------------------------------------------------- //
  // 2. Building row offsets
  // ----------------------------------------------------------------------- //

  thrust::device_vector<int> unique_rows, unique_cols;
  thrust::device_vector<int> row_offsets, row_indices, col_indices;

  // copies of sorted block_row_indices and block_col_indices
  thrust::device_vector<int> block_row_indices_copy = block_row_indices;
  thrust::device_vector<int> block_col_indices_copy = block_col_indices;

  // Use thrust::unique_by_key to find the unique row-col pairs
  // NOTE: these are the final row and column indices for the matrix
  auto unique_p = thrust::unique_by_key(
      thrust::make_zip_iterator(
          thrust::make_tuple(
              block_row_indices_copy.begin(),
              block_col_indices_copy.begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(
              block_row_indices_copy.end(),
              block_col_indices_copy.end())),
      thrust::make_zip_iterator(
          thrust::make_tuple(unique_rows.begin(), unique_cols.begin())));
  
  // Resize unique_rows and unique_cols to the new size
  auto Tend2 =  unique_p.second.get_iterator_tuple();;
  unique_rows.resize(thrust::get<0>(Tend2) - unique_rows.begin());
  unique_cols.resize(thrust::get<1>(Tend2) - unique_cols.begin());

  // Now count the number of occurrences of each row value
  // TODO this will not work if num of offsets is not equal to matrix rows
  // Need to do a prefix sum on the row_offsets to get the final offsets
  row_offsets.resize(unique_rows.size());
  row_indices.resize(unique_rows.size());

  // Returns pair of iterators to the end of the (indices and offsets) vectors
  auto pair_end = thrust::reduce_by_key(unique_rows.begin(), unique_rows.end(),
      thrust::constant_iterator<int>(1), row_indices.begin(), row_offsets.begin());

  // Now compute the row offsets by doing a prefix sum
  thrust::inclusive_scan(row_offsets.begin(), pair_end.second, row_offsets.begin());
  row_offsets.resize(pair_end.second - row_offsets.begin());
}