#include "mixed_stretch_gpu.h"
#include "mesh/mesh.h"


using namespace Eigen;
using namespace mfem;

#include <ginkgo/ginkgo.hpp>


template<int DIM, StorageType STORAGE> 
void MixedStretchGpu<DIM,STORAGE>::apply(double* x, const double* b) {
  // Create gko view of x and y

  // std::cout << "!!!!!!!!!!! ASSIGN EXECUTOR !!!!!!!!!!!!!!! " << std::endl;
  // exit(1);
  size_t size = (size_t) assembler2_->size();
  // std::cout << "size: " << size << std::endl;
  // Create gko vectors for x and b
  auto x_gko = gko::matrix::Dense<double>::create(exec_,
      gko::dim<2>{size, 1},
      gko::array<double>::view(exec_, size, x), 1);
  auto b_gko = gko::matrix::Dense<double>::create_const(exec_,
      gko::dim<2>{size, 1},
      gko::array<double>::const_view(exec_, size, b), 1);
  // std::cout << " num row blocks: " << assembler2_->num_row_blocks() << std::endl;
  // Create gko fbcsr matrix
  auto hessian = gko::matrix::Fbcsr<double,int>::create_const(exec_,
      gko::dim<2>{size}, DIM, 
      gko::array<double>::const_view(exec_, assembler2_->num_values(),
          assembler2_->values()),
      gko::array<int>::const_view(exec_, assembler2_->num_col_indices(),
          assembler2_->col_indices()),
      gko::array<int>::const_view(exec_, assembler2_->num_row_blocks(),
          assembler2_->row_offsets()));
  hessian->apply(lend(b_gko), lend(x_gko));
 
}

// template class mfem::MixedStretchGpu<3,STORAGE_THRUST>; // 3D
// template class mfem::MixedStretchGpu<3,STORAGE_EIGEN>; // 3D
template void mfem::MixedStretchGpu<3,STORAGE_THRUST>::apply(double*, const double*);
template void mfem::MixedStretchGpu<3,STORAGE_EIGEN>::apply(double*, const double*);
