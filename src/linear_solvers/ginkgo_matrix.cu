#include "ginkgo_matrix.h"
#include "EigenTypes.h"
#include <thrust/execution_policy.h>

using namespace mfem;
using namespace Eigen;

namespace {

  template<typename Scalar, int DIM>
  struct inv_functor {

    double* diag;

    inv_functor(double* diag) : diag(diag) {}

    __device__
    void operator()(int i) {
      // Get the diagonal block
      Map<Matrix<Scalar,DIM,DIM>> diag_block(diag + i*DIM*DIM);

      diag_block = diag_block.inverse();
    }
  };

  template<typename Scalar, int DIM>
  struct apply_functor {

    const double* diag;
    const double* b;
    double* x;

    apply_functor(const double* diag, const double* b,
        double* x) : diag(diag), b(b), x(x) {}

    __device__
    void operator()(int i) {
      // Get the diagonal block
      Map<const Matrix<Scalar,DIM,DIM>> diag_block(diag + i*DIM*DIM);
      Map<const Matrix<Scalar,DIM,1>> b_block(b + i*DIM);
      Map<Matrix<Scalar,DIM,1>> x_block(x + i*DIM);
      x_block = diag_block * b_block;
    }

  };

}
template<typename Scalar, int DIM>
void mfem::GkoBlockJacobiPreconditioner<Scalar,DIM>::compute_inv(double* diag, int nrows) {
  thrust::for_each(thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(nrows),
      inv_functor<Scalar,DIM>(diag));
}
      
template<typename Scalar, int DIM>
void mfem::GkoBlockJacobiPreconditioner<Scalar,DIM>::apply_inv_diag(
    const double* diag, const double* b, double* x, int nrows) {

  // Iterate over each row and apply DIMxDIM inverse on diag solved against b
  thrust::for_each(thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(nrows),
      apply_functor<Scalar,DIM>(diag, b, x));
}

template<typename Scalar, int DIM>
void mfem::GkoBlockJacobiPreconditioner<Scalar,DIM>::init_block_csr(
    int* row_offsets, int* col_indices, int nrows) {
  thrust::sequence(thrust::device, row_offsets, row_offsets + nrows + 1);
  thrust::sequence(thrust::device, col_indices, col_indices + nrows);
}


template void mfem::GkoBlockJacobiPreconditioner<double,3>::compute_inv(double*,int);
template void mfem::GkoBlockJacobiPreconditioner<double,3>::apply_inv_diag(const double*, const double*, double*,int);
template void mfem::GkoBlockJacobiPreconditioner<double,3>::init_block_csr(int*,int*,int);