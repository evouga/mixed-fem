#include "mixed_stretch_gpu.h"
#include "mesh/mesh.h"
#include <cusparse.h>

using namespace Eigen;
using namespace mfem;
using namespace thrust::placeholders;

template<int DIM>
MixedStretchGpu<DIM>::MixedStretchGpu(std::shared_ptr<Mesh> mesh)
    : MixedVariable<DIM>(mesh) {
  nelem_ = mesh_->T_.rows();

  s_.resize(N()*nelem_);
  g_.resize(N()*nelem_);
  ds_.resize(N()*nelem_);
  la_.resize(N()*nelem_);
  // la_.setZero();
  R_.resize(M()*nelem_);
  S_.resize(N()*N()*nelem_);
  H_.resize(N()*N()*nelem_);
  dSdF_.resize(N()*M()*nelem_);
  Hinv_.resize(N()*N()*nelem_);
  Hloc_.resize(N()*N()*nelem_);

  std::cout << "Assuming triangles in 2D and tetrahedra in 3D" << std::endl;
  int N_loc; // size of local stiffness matrix
  if constexpr (DIM == 2) {
    N_loc = DIM * 3;
  } else {
    N_loc = DIM * 4;
  }
  Aloc_.resize(N_loc*N_loc*nelem_);
}

template<int DIM>
__device__ 
void MixedStretchGpu<DIM>::init_variables(int i, double* si_data) {
  Map<VecN> si(si_data + N()*i);
  if constexpr (DIM == 2) {
    si << 1, 1, 0;
  } else {
    si << 1, 1, 1, 0, 0, 0;
  }
}

template<int DIM>
void MixedStretchGpu<DIM>::reset() {

  double* si_data = thrust::raw_pointer_cast(s_.data());
  thrust::for_each(thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(nelem_),
      [this, si_data] __device__ (const int i) {
        Map<VecN> si(si_data + N()*i);
        if constexpr (DIM == 2) {
            si << 1, 1, 0;
        } else {
            si << 1, 1, 1, 0, 0, 0;
        }
  });

  MatrixXd tmp = mesh_->V_.transpose();
  VectorXd x = Map<VectorXd>(tmp.data(), mesh_->V_.size());
  VectorXd def_grad;
  mesh_->deformation_gradient(x, def_grad);

  Eigen::SparseMatrix<double, RowMajor> A =
      mesh_->template jacobian<JacobianType::FULL>();
  A.makeCompressed();

  // Create a cusparse handle and matrix descriptor
  cusparseHandle_t     handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;
  double alpha = 1.0f;
  double beta  = 0.0f;

  // Sparse matrix CSR data
  int nnz = A.nonZeros();
  int *d_csr_row_offsets, *d_csr_columns;
  double *d_csr_values;
  cudaMalloc((void**)&d_csr_row_offsets, sizeof(int) * (A.rows() + 1));
  cudaMalloc((void**)&d_csr_columns, sizeof(int) * nnz);
  cudaMalloc((void**)&d_csr_values, sizeof(double) * nnz);

  // Copy CSR data to device
  cudaMemcpy(d_csr_row_offsets, A.outerIndexPtr(), sizeof(int) * (A.rows()+1),
      cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr_columns, A.innerIndexPtr(), sizeof(int) * nnz,
      cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr_values, A.valuePtr(), sizeof(double) * nnz,
      cudaMemcpyHostToDevice);

  // Create a cusparse handle and matrix descriptor
  cusparseCreate(&handle);
  cusparseCreateCsr(&matA, A.rows(), A.cols(), nnz,
                    d_csr_row_offsets, d_csr_columns, d_csr_values,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

  // Create dense input vector
  double* dx;
  cudaMalloc((void**)&dx, sizeof(double) * A.cols());
  cudaMemcpy(dx, x.data(), sizeof(double) * A.cols(),
      cudaMemcpyHostToDevice);
  // Create dense output vector
  double *dy;
  cudaMalloc((void**)&dy, sizeof(double) * A.rows());

  // Create dense vector X & Y vectors
  cusparseCreateDnVec(&vecX, A.cols(), dx, CUDA_R_64F);
  cusparseCreateDnVec(&vecY, A.rows(), dy, CUDA_R_64F);

  // allocate an external buffer if needed
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                          CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  cudaMalloc(&dBuffer, bufferSize);

  // cusparseStatus_t status;
  // FINALLY perform SPMV
  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
              &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
              CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

  // destroy matrix/vector descriptors
  cusparseDestroySpMat(matA);
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  cusparseDestroy(handle);

  // Copy the result back to the host
  VectorXd F(A.rows());
  cudaMemcpy(F.data(), dy, sizeof(double)*A.rows(), cudaMemcpyDeviceToHost);
  VectorXd F2 = A * x;
  std::cout << "F: " << F << std::endl;
  // std::cout << "def_grad: " << def_grad << std::endl;
  std::cout << "F - def_grad: " << (F - def_grad).norm() << std::endl;
  std::cout << "F2 - def_grad: " << (F2 - def_grad).norm() << std::endl;
    std::cout << "x size: " << x.size() << " A rows: " << A.rows() << " A cols: " << A.cols() << std::endl;
    std::cout << "bufferSize" << bufferSize << std::endl;
  // Copy Eigen matrix to GPU with CUDA and perform sparse
  // matrix-vector multiplication with CUSPARSE

  cudaFree(dBuffer);
  cudaFree(d_csr_row_offsets);
  cudaFree(d_csr_columns);
  cudaFree(d_csr_values);
  cudaFree(dx);
  cudaFree(dy);


  // std::cout << " Nelem: " << nelem_ << 
  // Aloc_.resize(nelem_);
  // evals_.resize(nelem_);
  // assembler_ = std::make_shared<Assembler<double,DIM,-1>>(
  //     mesh_->T_, mesh_->free_map_);
}
template class mfem::MixedStretchGpu<3>; // 3D
template class mfem::MixedStretchGpu<2>; // 2D
