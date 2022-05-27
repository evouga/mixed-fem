#include "sparse_utils.h"

using namespace mfem;
using namespace Eigen;

template <typename Scalar, int DIM>
Assembler<Scalar,DIM>::Assembler(const MatrixXi& E,
    const std::vector<int>& free_map) : N_(E.cols())
{
  element_ids.reserve(N_*N_*E.rows());
  global_pairs.reserve(N_*N_*E.rows());
  local_pairs.reserve(N_*N_*E.rows());

  int cur_id = 0;
  std::vector<int> ids;
  ids.reserve(N_*N_*E.rows());

  int cols = E.cols();
  // Identify all node pairs for each element
  for (int i = 0; i < E.rows(); ++i) {
    for (int j = 0; j < cols; ++j) {
      int id1 = free_map[E(i,j)];
      for (int k = 0; k < cols; ++k) {
        int id2 = free_map[E(i,k)];

        // If both nodes are unpinned, insert the pair
        if (id1 != -1 && id2 != -1) {
          element_ids.push_back(i);
          global_pairs.push_back(std::make_pair(id1,id2));
          local_pairs.push_back(std::make_pair(j,k));
          ids.push_back(cur_id++);
        }
      }
    }
  }

  // Sort the pairs so that they are in row major ordering
  auto sorter = [&](int i, int j)->bool {
    return global_pairs[i].first < global_pairs[j].first ||
            (global_pairs[i].first == global_pairs[j].first 
            && global_pairs[i].second < global_pairs[j].second);
  };
  //std::cout << "ids: ";
  //for(int i= 0; i < ids.size();++i) { std::cout << ids[i] << ", ";}std::cout << std::endl;
  std::sort(ids.begin(), ids.end(), sorter);
  reorder(global_pairs, ids);
  reorder(element_ids, ids);
  reorder(local_pairs, ids);

  //std::cout << "ids: ";
  //for(int i= 0; i < ids.size();++i) { std::cout << ids[i] << ", ";
  //  std::cout << " global: " << global_pairs[i].first << ", " << global_pairs[i].second << " " << element_ids[i] << std::endl;
  //}std::cout << std::endl;

  // Identify multiplicity of each pair
  multiplicity.resize(element_ids.size(),1);

  int i = 0;
  int npairs = 0;
  int curr_row = 0;
  std::pair<int, int> curr_pair;

  row_offsets.push_back(curr_row);
  num_nodes = 0;

  // Loop over pairs, counting duplicates
  // Purpose of this is so that we can easily sum over duplicate
  // entries when assembling the matrix
  while (i < multiplicity.size()) {
      //&& (i + multiplicity[i]) < multiplicity.size()) {

    curr_pair = global_pairs[i];

    // Mark the offset in which we meet a new row
    if (curr_pair.first != curr_row) {
      row_offsets.push_back(i);
      curr_row = curr_pair.first;
    }

    // Check if duplicate global pair and increase multiplicity if so
    while ((i+multiplicity[i]) < multiplicity.size()
        && curr_pair == global_pairs[i + multiplicity[i]]) {
      ++multiplicity[i];
    }

    offsets.push_back(i);
    i += multiplicity[i];
    ++num_nodes;
  }
  row_offsets.push_back(multiplicity.size());

  // Initialize our sparse matrix
  std::vector<Triplet<Scalar>> trips;
  for (int i = 0; i < num_nodes; ++i) {
    std::pair<int,int>& p = global_pairs[offsets[i]];
    // std::cout << "i: " << i << " offset: " << offsets[i] << " pair: " << p.first << ", " << p.second << std::endl;
    for (int j = 0; j < DIM; ++j) {
      for (int k = 0; k < DIM; ++k) {
        trips.push_back(Triplet<double>(
              p.first*DIM+j, p.second*DIM+k,1.0));

      }
    }
  }

  int m = *std::max_element(free_map.begin(), free_map.end()) + 1;
  A.resize(DIM*m, DIM*m);
  A.setFromTriplets(trips.begin(), trips.end());
  // std::cout << "A\n" << A << std::endl;
}

 
template <typename Scalar, int DIM>
void Assembler<Scalar,DIM>::update_matrix(std::vector<MatrixXx<Scalar>> blocks)
{
  // Iterate over M rows at a time
  #pragma omp parallel for
  for (size_t ii = 0; ii < row_offsets.size() - 1; ++ii) {
    int row_beg = row_offsets[ii];
    int row_end = row_offsets[ii+1];

    // std::cout << "non zeros: " << A.nonZeros() << std::endl;
    // std::cout << "row beg: " << row_beg << " k_start: " << A.outerIndexPtr()[3*ii] << std::endl;;
    // std::cout << "row end: " << row_end << std::endl;
    // The index of the block for this row. 
    int block_col = 0;

    while (row_beg < row_end) {
      // Get number of duplicate blocks for this offset
      int n = multiplicity[row_beg];
      // std::cout << "n: " << n << " block col: " << block_col << std::endl;

      // Get the global positioning of this block
      const std::pair<int,int>& g = global_pairs[row_beg];
      // std::cout << "g: " << g.first << ", " << g.second << std::endl;

      // Compute Local block
      Matrix<double,DIM,DIM> local_block;
      local_block.setZero();
      for (int i = 0; i < n; ++i) {

        int e = element_ids[row_beg + i];

        // Local coordinates within element's block
        const std::pair<int,int>& l = local_pairs[row_beg + i];
        local_block += blocks[e].block(DIM*l.first, DIM*l.second, DIM, DIM);
      }

      // Apologies to the cache locality gods
      // For each row of the DIMxDIM blocks
      for (int i = 0; i < DIM; ++i) {
        // Find the corresponding column position for this block
        int start = A.outerIndexPtr()[DIM*ii + i] + DIM*block_col;
        // std::cout << "\t start : " << i << " start : " << start << std::endl;
        for (int j = 0; j < DIM; ++j) {
          // Assign the value
          int idx = start + j;
          A.valuePtr()[idx] = local_block(i,j);
        }
      }
      
      // Move to next sequence of duplicate blocks
      row_beg += n;
      ++block_col;
    }

  }
}

template <typename Scalar, int DIM>
void Assembler<Scalar,DIM>::assemble(std::vector<Matrix<Scalar,Dynamic,1>> vecs,
    VectorXd& a) {
  a.resize(A.rows());
  a.setZero();

  // Iterate over M rows at a time
  #pragma omp parallel for
  for (size_t ii = 0; ii < row_offsets.size() - 1; ++ii) {
    int row_beg = row_offsets[ii];
    int row_end = row_offsets[ii+1];

    // The index of the block for this row. 
    int block_col = 0;

    while (row_beg < row_end) {
      // Get number of duplicate blocks for this offset
      int n = multiplicity[row_beg];

      // Compute Local block
      Matrix<Scalar,DIM,1> local_vec;
      local_vec.setZero();

      for (int i = 0; i < n; ++i) {
        int e = element_ids[row_beg + i];

        // Local coordinates within element's block
        const std::pair<int,int>& l = local_pairs[row_beg + i];
        const std::pair<int,int>& g = global_pairs[row_beg + i];
        if (l.first == l.second && g.first == ii)
          local_vec += vecs[e].template segment<DIM>(DIM*l.first);
      }

      a.segment<DIM>(DIM*ii) = local_vec;
      
      // Move to next sequence of duplicate blocks
      row_beg += n;
      ++block_col;
    }

  }
}

// Explicit template instantiation
template class mfem::Assembler<double, 3>;
template class mfem::Assembler<double, 2>;
