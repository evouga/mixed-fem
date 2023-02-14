#pragma once

#include <ginkgo/ginkgo.hpp>
#include "block_csr.h"

namespace mfem {

  /// @brief Apply a block CSR matrix to a vector using Ginkgo
  ///
  /// @param exec Ginkgo executor
  /// @param mat block CSR matrix
  /// @param x output dense matrix (stored in row major format)
  /// @param b input dense matrix (stored in row major format)
  /// @param ncols number of columns in x and b
  /// @tparam N block size
  template<int N>
  void bcsr_apply(
      std::shared_ptr<const gko::Executor> exec,
      std::shared_ptr<BlockMatrix<double,N,4>> mat,
      double* x, const double* b, int ncols = 1);

}