#pragma once

#include "EigenTypes.h"
#include <cusparse.h>
#include <cusolverDn.h>

namespace mfem {

  // Wrapper for Eigen sparse matrix on the gpu with cusparse spare
  // matrix-vector product support
  class SparseMatrixGpu {
  public:
    SparseMatrixGpu() {}

    SparseMatrixGpu(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A);

    void init(Eigen::SparseMatrix<double, Eigen::RowMajor> A);

    void product(double* dx, double** y);
    void product(const double* dx, double** y);
    void product_mat(const double* B, double* C, int ncols);

    ~SparseMatrixGpu() {
      // destroy matrix/vector descriptors
      cusparseDestroySpMat(matA);
      cusparseDestroyDnVec(vecY);
      cusparseDestroy(handle);
      cudaFree(d_csr_row_offsets);
      cudaFree(d_csr_columns);
      cudaFree(d_csr_values);
      cudaFree(d_y);
      cudaFree(dBuffer);

    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }

  protected:
    // Cusparse handle and matrix/vector descrptors
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecY;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    int rows_, cols_;
    double alpha = 1.0f;
    double beta  = 0.0f;

    // Cuda device CSR pointers
    int *d_csr_row_offsets;
    int *d_csr_columns;
    double *d_csr_values;

    double *d_y; // device output vector
  };

  template <int N>
  class MatrixBatchInverseGpu {


  public:

    MatrixBatchInverseGpu(int batch_size, double solver_tol=1e-7,
        int max_sweeps=15);

    void compute(double* A, double* Ainv);
    // void init(Eigen::SparseMatrix<double, Eigen::RowMajor> A);
    // void product(double* dx, double** y);

    ~MatrixBatchInverseGpu() {
      // cudaFree(d_A);
      // cudaFree(d_W);
      // cudaFree(d_V);
      // cudaFree(d_info);
      // cudaFree(d_work);
      // cusolverDnDestroySyevjInfo(syevj_params);
      // cusolverDnDestroy(cusolverH);
    }

    // int rows() const { return rows_; }
    // int cols() const { return cols_; }
  private:

    void invert(int i, double* V, double* W, double* A, double* Ainv);

    int batch_size_;
    double *d_A = nullptr;    /* N-by-N-by-batchSize */
    double *d_W = nullptr;    /* N-by-batchSize */
    double *d_V = nullptr;    /* N-by-batchSize */
    int *d_info = nullptr;    /* batchSize */
    double *d_work = nullptr; /* device workspace for syevjBatched */
    int lwork = 0;            /* size of workspace */

    /* configuration of syevj  */
    const double solver_tol_ = 1.e-7;
    const int max_sweeps_ = 15;
    const int sort_eig = 0;    /* don't sort eigenvalues */
    
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    syevjInfo_t syevj_params = NULL;

    /* compute eigenvectors */
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; 
    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

  };

}