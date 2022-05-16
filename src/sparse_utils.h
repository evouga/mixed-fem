#pragma once

#include <EigenTypes.h>

namespace mfem {

  // Class for parallel assembly of FEM stiffness matrices
  // Each element's input is a block of size NxN composed of MxM sub-blocks
  // these sub-blocks are scattered to their global nodes positions and
  // summed with duplicates.
  template <typename Scalar, int N, int M>
  class Assembler {
  public:

    template<typename T>
    static void reorder(std::vector<T>& val, const std::vector<int>& indices) {
      std::vector<T> tmp(indices.size());

      for (int i = 0; i < indices.size(); ++i) {
        tmp[i] = val[indices[i]];
        //tmp[indices[i]] = val[i];
      }
      std::copy(tmp.begin(), tmp.end(), val.begin());
    }

    // Initialize assembler / analyze sparsity of system
    // None of this wonderfully optimized since we only have to do it once
    // E        - elements nelem x 4 for tetrahedra
    // free_map - |nnodes| maps node to its position in unpinned vector
    //            equals -1 if node is pinned
    //template <typename Scalar, int N, int M>
    Assembler(const Eigen::MatrixXi& E, const std::vector<int> free_map) {

      element_ids.reserve(N*N*E.rows());
      global_pairs.reserve(N*N*E.rows());
      local_pairs.reserve(N*N*E.rows());

      int cur_id = 0;
      std::vector<int> ids;
      ids.reserve(N*N*E.rows());

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
      while (i < multiplicity.size()
          && (i + multiplicity[i]) < multiplicity.size()) {

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
      std::vector<Eigen::Triplet<Scalar>> trips;
      for (int i = 0; i < num_nodes; ++i) {
        std::pair<int,int>& p = global_pairs[offsets[i]];
        //std::cout << "i: " << i << " offset: " << offsets[i] << " pair: " << p.first << ", " << p.second << std::endl;
        for (int j = 0; j < M; ++j) {
          for (int k = 0; k < M; ++k) {
            trips.push_back(Eigen::Triplet<double>(
                  p.first*M+j, p.second*M+k,1.0));

          }
        }
      }

      int m = *std::max_element(free_map.begin(), free_map.end()) + 1;
      //std::cout << "m: " << m << std::endl;
      A.resize(M*m, M*m);
      A.setFromTriplets(trips.begin(), trips.end());
    }

    // Probably not the ideal way to parallelize this but fuck it
    void update_matrix(std::vector<Eigen::Matrix<double,M*N,M*N>> blocks) {

      // Iterate over M rows at a time
      #pragma omp parallel for
      for (int ii = 0; ii < row_offsets.size() - 1; ++ii) {
        int row_beg = row_offsets[ii];
        int row_end = row_offsets[ii+1];

        //std::cout << "row beg: " << row_beg << " k_start: " << A.outerIndexPtr()[3*ii] << std::endl;;
        // The index of the block for this row. 
        int block_col = 0;

        while (row_beg < row_end) {
          // Get number of duplicate blocks for this offset
          int n = multiplicity[row_beg];
          //std::cout << "n: " << n << " block col: " << block_col << std::endl;

          // Get the global positioning of this block
          const std::pair<int,int>& g = global_pairs[row_beg];

          // Compute Local block
          Eigen::Matrix<double,M,M> local_block;
          local_block.setZero();
          for (int i = 0; i < n; ++i) {

            int e = element_ids[row_beg + i];

            // Local coordinates within element's block
            const std::pair<int,int>& l = local_pairs[row_beg + i];
            local_block += blocks[e].block(M*l.first, M*l.second, M, M);
          }

          // Apologies to the cache locality gods
          // For each row of the MxM blocks
          for (int i = 0; i < M; ++i) {
            // Find the corresponding column position for this block
            int start = A.outerIndexPtr()[M*ii + i] + M*block_col;
            for (int j = 0; j < M; ++j) {
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

    // Element IDs, global, and local coordinates. Each of these vectors
    // is of the same size.
    std::vector<int> element_ids;
    std::vector<std::pair<int,int>> global_pairs;
    std::vector<std::pair<int,int>> local_pairs;

    int num_nodes; // number of unique pairs / blocks in matrix
    std::vector<int> multiplicity; // number of pairs to sum over for a node
    std::vector<int> row_offsets;
    std::vector<int> offsets;
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
