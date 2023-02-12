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
#include "optimizers/optimizer_data.h"

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
void BlockMatrix<Scalar,DIM,N>::extract_diagonal_functor
    ::operator()(int i) const {
  Map<MatD> diag_i(diag + DIM*DIM*i);

  int row_beg = row_offsets[i];
  int row_end = row_offsets[i+1];

  for (int j = row_beg; j < row_end; j++) {
    int col = col_indices[j];
    if (col == i) {
      Map<const MatD> values_diag_i(values + DIM*DIM*j);
      diag_i += values_diag_i;
      return;
    }
  }
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
  // i is the element index
  for (int r = 0; r < N; ++r) {
    for (int c = 0; c < N; ++c) {
      // Get the global index of the block
      int block_index = i * N * N + r * N + c;

      int idx = block_indices[block_index];
      if (idx < 0) continue;

      MatD& block = blocks_with_duplicates[idx];
      // blocks is a |E|*N*DIM*N*DIM array of blocks each of size
      // (N*DIM) x (N*DIM) and i = 1 : |E|*N*N, so we need to extract
      // the DIMxDIM block for the i-th block
      
      int col_start = 
          i * N * N * DIM * DIM // start of element matrix
          + c * DIM * DIM * N    // start of local block column
          + r * DIM;             // row for current dimension
      
      for (int col = 0; col < DIM; ++col) {
        int row_start = col_start + col * N * DIM;
        for (int row = 0; row < DIM; ++row) {
          block(row, col) = blocks[row_start + row];
        }
      }
    }
  }
}


template<typename Scalar, int DIM, int N> __device__
void BlockMatrix<Scalar,DIM,N>::update_functor2::operator()(int i) {
  // i = 1: |E|*N*N
  // Get the element index
  int elem = i / (N * N);
  // Get the local row and column indices
  int r = (i / N) % N;
  int c = i % N;

  // Get the global index of the block
  // int block_index = i * N * N + r * N + c;
  int block_index = i;

  int idx = block_indices[block_index];
  if (idx < 0) return;;

  MatD& block = blocks_with_duplicates[idx];
  // blocks is a |E|*N*DIM*N*DIM array of blocks each of size
  // (N*DIM) x (N*DIM) and i = 1 : |E|*N*N, so we need to extract
  // the DIMxDIM block for the i-th block
  
  int col_start = 
      elem * N * N * DIM * DIM // start of element matrix
      + c * DIM * DIM * N      // start of local block column
      + r * DIM;               // row for current dimension
  
  for (int col = 0; col < DIM; ++col) {
    int row_start = col_start + col * N * DIM;
    for (int row = 0; row < DIM; ++row) {
      block(row, col) = blocks[row_start + row];
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
  nnodes_ = *thrust::max_element(free_map_d.begin(), free_map_d.end()) + 1;

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

  // for (int i = 0; i < 120; ++i) {
  //   std::cout << "i: "<< i << " (r,c,map): " << block_row_indices_[i] << " " << block_col_indices_[i] << " "
  //     << sorted_block_map[i] <<  std::endl; 
  // }

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
  // thrust::device_vector<int> seq(E.rows() * N * N);
  // thrust::sequence(seq.begin(), seq.end());
  // thrust::gather(sorted_block_map.begin(), sorted_block_map.end(),
  //     seq.begin(), block_indices_.begin());
  // sorted_block_indices is a set of indices now form the inverse map
  // that go from the indices to the map to the index in the sorted_block_indices
  thrust::scatter(thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(0) + sorted_block_map.size(),
      sorted_block_map.begin(), block_indices_.begin());

  // for (int i = 0; i < 10; ++i) {
  //   std::cout << "i: "<< i << " (r,c,map,block_indices): " << block_row_indices_[i] << " " << block_col_indices_[i] << " "
  //     << sorted_block_map[i] << " " << block_indices_[sorted_block_map[i]] << std::endl;
  // }

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
  
  // std::cout << "size 0 " << row_indices.size() << std::endl;
  // Resize unique_rows and unique_cols to the new size
  auto Tend2 = iter.get_iterator_tuple();
  row_indices.resize(thrust::get<0>(Tend2) - row_indices.begin());
  col_indices_.resize(thrust::get<1>(Tend2) - col_indices_.begin());
  blocks_.resize(row_indices.size());

  // std::cout << "size 1 " << row_indices.size() << std::endl;
  // for (int i = 0; i < 50; ++i) {
  //   std::cout << "i: "<< i << " (r,c): " << row_indices[i] << " " << col_indices_[i] << std::endl;
  // }
  // Now count the number of occurrences of each row value
  // TODO this will not work if num of offsets is not equal to matrix rows
  // Need to do a prefix sum on the row_offsets to get the final offsets
  // Do a scatter to the full size row offsets, then prefix sum over that
  row_offsets_.resize(row_indices.size());

  // Count the number of matching row indices and write to row_offsets
  // Returns pair of iterators to the end of the (indices and offsets)
  auto pair_end = thrust::reduce_by_key(row_indices.begin(), row_indices.end(),
      thrust::constant_iterator<int>(1), row_indices.begin(), row_offsets_.begin());

  // for (int i = 0; i < 8; ++i) {
  //   std::cout << "i: "<< i << " (row,count): " << row_indices[i] << " " <<  row_offsets_[i] << std::endl;
  // }

  // Now compute the row offsets by doing a prefix sum

  int nrows = pair_end.second - row_offsets_.begin();

  // row_offsets_.resize(pair_end.second - row_offsets_.begin());
  row_offsets_.resize(nnodes_ + 1);

  thrust::device_vector<int> row_offsets_tmp(nnodes_ + 1, 0);

  // If number of unique row offsets is less than actual number of rows
  // in the matrix. Scatter the current row offsets 
  if (nrows < nnodes_) {
    // std::cout << "Scattering in assembler! " << std::endl;
    // Get unique row_indices, to be used for scattering map
    thrust::device_vector<int> row_indices_tmp = row_indices;
    auto it2 = thrust::unique(row_indices_tmp.begin(), row_indices_tmp.end());
    thrust::scatter(row_offsets_.begin(), row_offsets_.begin() + nrows,
        row_indices_tmp.begin(), row_offsets_tmp.begin());
    // for (int i = 0; i < row_indices.size(); ++i) {
    //   std::cout << "i: "<< i << " (row,count): " << row_indices[i] << " " <<  row_offsets_[i] << std::endl;
    // }
    // for (int i = 0; i < row_offsets_tmp.size(); ++i) {
    //   std::cout << "i: "<< i << " (row_offsets_tmp): " << row_offsets_tmp[i] << std::endl;
    // }
    row_offsets_ = row_offsets_tmp;

  } else {

  }
  thrust::inclusive_scan(thrust::device, row_offsets_.begin(), 
      row_offsets_.end() - 1, row_offsets_tmp.begin()+1);
  row_offsets_ = row_offsets_tmp;
  row_offsets_[0] = 0;
  


  // thrust::inclusive_scan(thrust::device, row_offsets_.begin(), pair_end.second, row_offsets_.begin()+1);
  // row_offsets_[0] = 0;

  // std::cout << "row offsets size " << row_offsets_.size() << std::endl;
  // for (int i = 0; i <  row_offsets_.size(); ++i) {
  //   std::cout << "i: "<< i << " offset: " << row_offsets_[i] << std::endl;
  // }
}

template<typename Scalar, int DIM, int N>
void BlockMatrix<Scalar,DIM,N>::update_matrix(
    const thrust::device_vector<double>& blocks) {
  // OptimizerData::get().timer.start("assemble1", "MixedStretchGpu");
  // Blocks is vector of size |E| x (N*N*DIM*DIM)
  // For each DIMxDIM block, write to blocks_with_duplicates using block_indices
  // thrust::for_each(thrust::counting_iterator<int>(0),
  //     thrust::counting_iterator<int>(nelem_),
  //     update_functor(
  //       thrust::raw_pointer_cast(blocks.data()),
  //       thrust::raw_pointer_cast(block_indices_.data()),
  //       thrust::raw_pointer_cast(blocks_with_duplicates_.data())));
  // OptimizerData::get().timer.stop("assemble1", "MixedStretchGpu");

  // OptimizerData::get().timer.start("assemble1v2", "MixedStretchGpu");
  thrust::for_each(thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(nelem_*N*N),
      update_functor2(
        thrust::raw_pointer_cast(blocks.data()),
        thrust::raw_pointer_cast(block_indices_.data()),
        thrust::raw_pointer_cast(blocks_with_duplicates_.data()), nelem_));
  // OptimizerData::get().timer.stop("assemble1v2", "MixedStretchGpu");
  // for (int i = 0; i < 10; ++i) {
  //   std::cout << "i: " << i << std::endl;
  //   std::cout << "  row col: " << block_row_indices_[i] << " " << block_col_indices_[i] << std::endl;
  //   std::cout << "  block \n " << blocks_with_duplicates_[i] << std::endl;
  // }
  // Using sorted block_row_indices and block_col_indices, do a reduction
  // over the duplicates
  if (row_indices_tmp_.size() == 0) {
    row_indices_tmp_.resize(col_indices_.size());
  }
  
  thrust::reduce_by_key(
      thrust::make_zip_iterator(thrust::make_tuple(
          block_row_indices_.begin(),
          block_col_indices_.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(
          block_row_indices_.end(),
          block_col_indices_.end())),
      blocks_with_duplicates_.begin(),
      thrust::make_zip_iterator(thrust::make_tuple(
          row_indices_tmp_.begin(),
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
  // A_.resize(nnodes_*DIM, nnodes_*DIM);
  A_ = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>(nnodes_*DIM, nnodes_*DIM);
  std::vector<Eigen::Triplet<Scalar>> triplets;
  // std::cout << "nnodes: " << nnodes_ << std::endl;
  for(int i=0; i<h_row_offsets.size()-1; ++i) {
    // std::cout << "row: " << i << " offset: " << h_row_offsets[i] << " " << h_row_offsets[i+1] << std::endl;
    for(int j=h_row_offsets[i]; j<h_row_offsets[i+1]; ++j) {
      int row = i;
      int col = h_col_indices[j];
      // std::cout << "row: " << row << " col: " << col << std::endl;
      MatD block = h_blocks[j];
      // std::cout << "block: \n" << block << std::endl;
      for(int k=0; k<DIM; k++) {
        for(int l=0; l<DIM; l++) {
          triplets.push_back(Eigen::Triplet<Scalar>(
              row*DIM+k, col*DIM+l, block(k,l)));
        }
      }
    }
  }
  A_.setFromTriplets(triplets.begin(), triplets.end());
  // std::cout << "BCSR A_ " << A_ << std::endl;
  return A_;
}

template<typename Scalar, int DIM, int N>
void BlockMatrix<Scalar,DIM,N>::extract_diagonal(double* diag) {
  typedef Eigen::Matrix<Scalar, DIM, DIM> MatD;
  MatD* diag_out = (MatD*)diag;

  // Get DIM x DIM diagonals from block CSR matrix
  thrust::for_each(thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(nnodes_),
      extract_diagonal_functor(
          diag, this->values(), this->row_offsets(), this->col_indices()));
}

template class mfem::BlockMatrix<double, 3, 4>;