#include "block_csr_apply.h"
#include "optimizers/optimizer_data.h"

using namespace mfem;

template<int N> 
void mfem::bcsr_apply(std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<BlockMatrix<double,N,4>> matrix,
    double* x, const double* b, int ncols) {

  if (exec == nullptr) {
    std::cout << "Executor is null" << std::endl;
    return;
  }
  if (matrix == nullptr) {
    std::cout << "Matrix is null" << std::endl;
    return;
  }
  size_t size = (size_t) matrix->size();

  // Create gko vectors for x and b
  auto x_gko = gko::matrix::Dense<double>::create(exec,
      gko::dim<2>{size, ncols},
      gko::array<double>::view(exec, size*ncols, x), ncols);
  auto b_gko = gko::matrix::Dense<double>::create_const(exec,
      gko::dim<2>{size, ncols},
      gko::array<double>::const_view(exec, size*ncols, b), ncols);

  // Create gko fbcsr matrix
  auto hessian = gko::matrix::Fbcsr<double,int>::create_const(exec,
      gko::dim<2>{size}, N, 
      gko::array<double>::const_view(exec, matrix->num_values(),
          matrix->values()),
      gko::array<int>::const_view(exec, matrix->num_col_indices(),
          matrix->col_indices()),
      gko::array<int>::const_view(exec, matrix->num_row_blocks(),
          matrix->row_offsets()));
  hessian->apply(lend(b_gko), lend(x_gko));
}

template void mfem::bcsr_apply<3>(
    std::shared_ptr<const gko::Executor>,
    std::shared_ptr<BlockMatrix<double,3,4>>,
    double*, const double*, int);