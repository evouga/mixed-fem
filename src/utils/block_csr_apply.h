#pragma once

#include <ginkgo/ginkgo.hpp>
#include "block_csr.h"

namespace mfem {

  /// @brief Apply a block CSR matrix to a vector using Ginkgo
  ///
  /// @param exec Ginkgo executor
  /// @param mat block CSR matrix
  /// @param x output vector
  /// @param b input vector
  /// @tparam N block size
  template<int N>
  void bcsr_apply(
      std::shared_ptr<const gko::Executor> exec,
      std::shared_ptr<BlockMatrix<double,N,4>> mat,
      double* x, const double* b);

}