#include "sparse_matrix_gpu.h"
#include <thrust/device_vector.h>
#include <cusparse.h>
#include <cusolverDn.h>

using namespace Eigen;
using namespace mfem;

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)

namespace {

  template<int N>
  struct psd_fix_functor {
      psd_fix_functor(double* V, double* W, double* A,
      double* Ainv) : V_(V), W_(W), A_(A), Ainv_(Ainv) {}

      __device__
      void operator()(int i) const {
          // Map<Matrix<double, N, N>> Ainvi(Ainv_ + i*N*N);
          Map<Matrix<double, N, N>> Vi(V_ + i*N*N);
          Map<Matrix<double, N, N>> Ai(A_ + i*N*N);
          Map<Matrix<double, N, 1>> Wi(W_ + i*N);
          
          Matrix<double, N, 1> Wi_inv;
          for (int j = 0; j < N; ++j) {
            // PSD fix
            if (Wi(j) < 1e-8) {
              Wi(j) = 1e-8;
            }
            Wi_inv(j) = 1.0 / Wi(j);
          }
          Ai = Vi * Wi.asDiagonal() * Vi.transpose();
          // Ainvi = Vi * Wi_inv.asDiagonal() * Vi.transpose();
          // printf("Wi %f", Wi(0));
          // printf("Ai %f", Ai(0, 0));
          // printf("Vi %f", Vi(0, 0));
      }

  private:
      double *V_;
      double *W_;
      double *A_;
      double *Ainv_;
  };

}

SparseMatrixGpu::SparseMatrixGpu(const SparseMatrix<double, RowMajor>& A) {
  init(A);
}

void SparseMatrixGpu::init(SparseMatrix<double, RowMajor> A) {
  A.makeCompressed();

  rows_ = A.rows();
  cols_ = A.cols();

  // Sparse matrix CSR data
  int nnz = A.nonZeros();
  cudaMalloc((void**)&d_csr_row_offsets, sizeof(int) * (rows_ + 1));
  cudaMalloc((void**)&d_csr_columns, sizeof(int) * nnz);
  cudaMalloc((void**)&d_csr_values, sizeof(double) * nnz);

  // Copy CSR data to device
  cudaMemcpy(d_csr_row_offsets, A.outerIndexPtr(),
      sizeof(int) * (rows_+1),
      cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr_columns, A.innerIndexPtr(), sizeof(int) * nnz,
      cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr_values, A.valuePtr(), sizeof(double) * nnz,
      cudaMemcpyHostToDevice);

  // Create a cusparse handle and matrix descriptor
  cusparseCreate(&handle);
  cusparseCreateCsr(&matA, rows_, cols_, nnz,
      d_csr_row_offsets, d_csr_columns, d_csr_values,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

  // Create dense output vector
  cudaMalloc((void**)&d_y, sizeof(double) * rows_);

  // Create dense vector X & Y vectors
  cusparseCreateDnVec(&vecY, rows_, d_y, CUDA_R_64F);
}

void SparseMatrixGpu::product(double* dx, double** y) {
  cusparseDnVecDescr_t vecX;
  cusparseCreateDnVec(&vecX, cols_, dx, CUDA_R_64F);

  // allocate an external buffer if needed
  if (dBuffer == NULL) {
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
  }

  // execute SpMV
  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
              &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
              CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

  cusparseDestroyDnVec(vecX);
  *y = d_y;
}

void SparseMatrixGpu::product(const double* dx, double** y) {
  cusparseConstDnVecDescr_t vecX;
  cusparseCreateConstDnVec(&vecX, cols_, dx, CUDA_R_64F);

  // allocate an external buffer if needed
  if (dBuffer == NULL) {
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
  }

  // execute SpMV
  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
              &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
              CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

  cusparseDestroyDnVec(vecX);
  *y = d_y;
}

template<int N>
MatrixBatchInverseGpu<N>::MatrixBatchInverseGpu(int batch_size, double solver_tol,
    int max_sweeps) 
    : batch_size_(batch_size), solver_tol_(solver_tol), max_sweeps_(max_sweeps) {

  std::cout << "MatrixBatchInverseGpu::MatrixBatchInverseGpu()" << std::endl;
  std::vector<int> info(batch_size, 0); 

  /* step 1: create cusolver handle, bind a stream */
  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

  /* step 2: configuration of syevj */
  CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));

  /* default value of tolerance is machine zero */
  CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params, solver_tol_));

  /* default value of max. sweeps is 100 */
  CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps_));

  /* disable sorting */
  CUSOLVER_CHECK(cusolverDnXsyevjSetSortEig(syevj_params, sort_eig));

  /* step 3: create device vectors */
  int size_A = N * N * batch_size;
  int size_W = N * batch_size;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * size_A));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_V), sizeof(double) * size_A));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(double) * size_W));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int) * batch_size));

  /* step 4: query working space of syevj */
  CUSOLVER_CHECK(cusolverDnDsyevjBatched_bufferSize(cusolverH, jobz, uplo, N,
      d_A, N, d_W, &lwork, syevj_params, batch_size));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork)); 
  std::cout << "MatrixBatchInverseGpu() done" << std::endl;
}

template<int N>
void MatrixBatchInverseGpu<N>::compute(double* A, double* Ainv) {
  // Copy device input to d_A (which will be overwritten)
  int size_A = N * N * batch_size_;
  int size_W = N * batch_size_;
  CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeof(double) * size_A,
      cudaMemcpyDeviceToDevice, stream));
  
  // Compute eigen-pairs. Eigenvectors are in d_A and eigenvalues in d_W
  CUSOLVER_CHECK(cusolverDnDsyevjBatched(
      cusolverH, jobz, uplo, N, d_A, N, d_W, d_work, lwork,
      d_info, syevj_params, batch_size_));

  // std::vector<int> info(batch_size_, 0);
  // CUDA_CHECK(cudaMemcpy(info.data(), d_info, sizeof(int) * batch_size_,
  //       cudaMemcpyDeviceToHost));

  // std::vector<double> W(size_W, 0.0);
  // CUDA_CHECK(
  //       cudaMemcpyAsync(W.data(), d_W, sizeof(double) * W.size(), cudaMemcpyDeviceToHost, stream));
  // // OKAY we have eigenvector and eigenvalues. Now we compute
  // // the inverse
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // for (int i = 0; i < batch_size_; i++) {
  //   if (0 == info[i]) {
  //       // std::printf("matrix %d: syevj converges \n", i);
  //   } else if (0 > info[i]) {
  //       /* only info[0] shows if some input parameter is wrong.
  //         * If so, the error is CUSOLVER_STATUS_INVALID_VALUE.
  //         */
  //       std::printf("Error: %d-th parameter is wrong \n", -info[i]);
  //       exit(1);
  //   } else { /* info = m+1 */
  //             /* if info[i] is not zero, Jacobi method does not converge at i-th matrix. */
  //       std::printf("WARNING: matrix %d, info = %d : sygvj does not converge \n", i, info[i]);
  //   }
  // }

  // for (int i = 0; i < W.size(); ++i) {
  //   std::cout << "W[i] " << W[i] << std::endl;
  // }

  thrust::for_each(thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(batch_size_),
      psd_fix_functor<N>(d_A, d_W, A, Ainv));
}

template class mfem::MatrixBatchInverseGpu<6>; // 6x6 matrices
template class mfem::MatrixBatchInverseGpu<3>; // 6x6 matrices
