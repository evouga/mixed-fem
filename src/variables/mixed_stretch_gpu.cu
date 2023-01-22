#include "mixed_stretch_gpu.h"
#include "mesh/mesh.h"
#include "utils/sparse_matrix_gpu.h"
#include "energies/material_model.h"
#include "optimizers/optimizer_data.h"
#include "svd/svd3_cuda_polar.cuh"
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include "energies/arap.cuh"

using namespace Eigen;
using namespace mfem;
using namespace thrust::placeholders;

template<int DIM, StorageType STORAGE> __device__
double MixedStretchGpu<DIM,STORAGE>::energy_functor::operator()(int i) const {
  Map<VecN> si(s + N()*i);
  Map<MatD> Fi(F + M()*i);
  Map<VecN> lai(la + N()*i);
  double vol = vols[i];
  // double mu = 10344.8;
  double mui = mu[i];
  double lambdai = lambda[i];
  double h2 = 0.0333 * 0.0333;

  double e = 0.0;
  // void svd_polar(const Eigen::Matrix3d& Ad, Eigen::Matrix3f& R) {
  if constexpr (DIM == 3) {
    Matrix<float,DIM,DIM> Rf;

    svd_polar(Fi, Rf);
    MatD Ri = Rf.template cast<double>();

    //  = Rf.cast<double>();
    MatD S3D = Ri.transpose() * Fi;
    VecN Si;
    Si << S3D(0,0), S3D(1,1), S3D(2,2),
      0.5*(S3D(1,0) + S3D(0,1)),
      0.5*(S3D(2,0) + S3D(0,2)),
      0.5*(S3D(2,1) + S3D(1,2));

    MatN sym = Sym();
    VecN diff = sym * (Si - si);
    e = vol * (lai.dot(diff)/h2 + local_energy<double>(si, mui));
  }
  return e;
  
}

template<int DIM, StorageType STORAGE> 
template<bool COMPUTE_GRADIENTS> __device__
void MixedStretchGpu<DIM,STORAGE>::
rotation_functor<COMPUTE_GRADIENTS>::operator()(int i) const {

  Map<MatD> Fi(F + M()*i);
  Map<MatD> Ri(R + M()*i);
  Map<VecN> Si(S + N()*i);
  Map<MatMN> dSdFi(dSdF + M()*N()*i);

  if constexpr (DIM == 3) {
    MatM dRdF;
    Matrix3f Rf;
    Matrix<float, 6, 9> dSdFf;
    // svd_polar(Fi, Rf);
    svd_deriv_S(Fi, Rf, dSdFf);
    Ri = Rf.cast<double>();

    MatD S3D = Ri.transpose() * Fi;
    Si << S3D(0,0), S3D(1,1), S3D(2,2),
        0.5*(S3D(1,0) + S3D(0,1)),
        0.5*(S3D(2,0) + S3D(0,2)),
        0.5*(S3D(2,1) + S3D(1,2));
    
    dSdFi = dSdFf.transpose().cast<double>();
  }
}

template<int DIM, StorageType STORAGE> __device__
void MixedStretchGpu<DIM,STORAGE>::derivative_functor::operator()(int i)
    const {
  double vol = vols[i];
  double mui = mu[i];
  double lambdai = lambda[i];
  Map<VecN> si(s + N()*i);
  Map<VecN> gi(g + N()*i);
  Map<MatN> Hi(H + N()*N()*i);
  Map<MatMN> dSdFi(dSdF + M()*N()*i);

  Map<Matrix<double, Aloc_N(), Aloc_N()>> Aloci(Aloc + Aloc_N()*Aloc_N()*i);
  Map<Matrix<double, M(), Aloc_N()>> Jloci(Jloc + M()*Aloc_N()*i);
  
  // TODO better selection
  // Look at how AMGCL or GINKGO does runtime selection of GPU stuff...
  double h2 = 0.0333 * 0.0333;
  // mui = 10344.8;

  Hi = h2 * local_hessian<double>(si, mui);
  gi = h2 * local_gradient<double>(si, mui);
  Aloci = vol * Jloci.transpose() * (dSdFi * Hi * dSdFi.transpose()) * Jloci;
}

template<int DIM, StorageType STORAGE> __device__
void MixedStretchGpu<DIM,STORAGE>::rhs_functor::operator()(int i) const {
  double vol = vols[i];
  Map<VecN> si(s + N()*i);
  Map<VecN> Si(S + N()*i);
  Map<VecN> gi(g + N()*i);
  Map<MatN> Hi(H + N()*N()*i);
  Map<MatMN> dSdFi(dSdF + M()*N()*i);
  Map<VecM> rhsi(rhs + M()*i);
  // MatN syminv = Syminv();
  // VecN gl = syminv * (Hi * (Si - si) + gi);
  VecN gl = (Hi * (Si - si) + gi);
  rhsi = -dSdFi * gl;
}

template<int DIM, StorageType STORAGE> __device__
void MixedStretchGpu<DIM,STORAGE>::solve_functor::operator()(int i) const {
  double vol = vols[i];
  Map<VecN> si(s + N()*i);
  Map<VecN> Si(S + N()*i);
  Map<VecN> gi(g + N()*i);
  Map<VecN> lai(la + N()*i);
  Map<VecN> dsi(ds + N()*i);
  Map<MatN> Hi(H + N()*N()*i);
  Map<VecM> Jdxi(Jdx + M()*i);
  Map<MatMN> dSdFi(dSdF + M()*N()*i);
  MatN syminv = Syminv();
  // MatN sym = Sym();

  dsi = (Si - si + dSdFi.transpose()*Jdxi/vol);
  lai = syminv * (Hi * dsi + gi);
}

template<int DIM, StorageType STORAGE>
double MixedStretchGpu<DIM,STORAGE>::energy(VectorType& x,
    VectorType& s) {
  OptimizerData::get().timer.start("energy-transfer", "MixedStretchGpu");
  // copy x and s to the GPU
  double* d_x; // device x vector
  double* d_s; // device s vector
  if constexpr (STORAGE == STORAGE_EIGEN) {
    cudaMalloc((void**)&d_x, sizeof(double) * x.size());
    cudaMalloc((void**)&d_s, sizeof(double) * s.size());
    cudaMemcpy(d_x, x.data(), sizeof(double) * x.size(),
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, s.data(), sizeof(double) * s.size(),
        cudaMemcpyHostToDevice);
  } else {
    d_x = thrust::raw_pointer_cast(x.data());
    d_s = thrust::raw_pointer_cast(s.data());
  }

  OptimizerData::get().timer.stop("energy-transfer", "MixedStretchGpu");

  // compute deformation gradient
  OptimizerData::get().timer.start("energy-F", "MixedStretchGpu");
  double* F;
  J_gpu_.product(d_x, &F);
  OptimizerData::get().timer.stop("energy-F", "MixedStretchGpu");
  OptimizerData::get().timer.start("energy", "MixedStretchGpu");

  // // call energy functor
  double e = thrust::transform_reduce(thrust::counting_iterator<int>(0),
    thrust::counting_iterator<int>(nelem_),
    energy_functor(d_s, F,
        thrust::raw_pointer_cast(la_.data()), 
        thrust::raw_pointer_cast(vols_.data()),
        thrust::raw_pointer_cast(params_.mu.data()),
        thrust::raw_pointer_cast(params_.lambda.data())),
        0.0, thrust::plus<double>());
  OptimizerData::get().timer.stop("energy", "MixedStretchGpu");
  if constexpr (STORAGE == STORAGE_EIGEN) {
    cudaFree(d_x);
    cudaFree(d_s);
  }
  return e;
}

template<int DIM, StorageType STORAGE>
void MixedStretchGpu<DIM,STORAGE>::update(VectorType& x, double dt) {

  vector<double> h2(1, dt*dt);
  double* d_x; // device x vector

  // copy host s to gpu s
  if constexpr (STORAGE == STORAGE_EIGEN) {
    cudaMemcpy(thrust::raw_pointer_cast(s_.data()), s_h_.data(), sizeof(double) * s_.size(),
        cudaMemcpyHostToDevice);

    OptimizerData::get().timer.start("def_grad_transfer", "MixedStretchGpu");
    cudaMalloc((void**)&d_x, sizeof(double) * x.size());
    cudaMemcpy(d_x, x.data(), sizeof(double) * x.size(),
        cudaMemcpyHostToDevice);
    OptimizerData::get().timer.stop("def_grad_transfer", "MixedStretchGpu");
  } else {
    d_x = thrust::raw_pointer_cast(x.data());
  }

  // First compute the deformation gradient
  OptimizerData::get().timer.start("def_grad", "MixedStretchGpu");
  double* F;
  J_gpu_.product(d_x, &F);
  OptimizerData::get().timer.stop("def_grad", "MixedStretchGpu");

  // make the host block until the device is finished with foo
  // check for error
  // cudaDeviceSynchronize();
  // cudaError_t error = cudaGetLastError();
  // if(error != cudaSuccess) {
  //   // print the CUDA error message and exit
  //   printf("CUDA error: %s\n", cudaGetErrorString(error));
  //   exit(-1);
  // }

  // Compute rotations and rotation derivatives (dSdF)
  OptimizerData::get().timer.start("rotations", "MixedStretchGpu");
  thrust::for_each(thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(nelem_),
      rotation_functor<true>(F,
          thrust::raw_pointer_cast(R_.data()), 
          thrust::raw_pointer_cast(S_.data()), 
          thrust::raw_pointer_cast(dSdF_.data())
      ));
  OptimizerData::get().timer.stop("rotations", "MixedStretchGpu");
  std::cout << "Update local hessians and gradient" << std::endl;
  OptimizerData::get().timer.start("gradients", "MixedStretchGpu");

  // Compute local hessians and gradients
  thrust::for_each(thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(nelem_),
      derivative_functor(
          thrust::raw_pointer_cast(s_.data()), 
          thrust::raw_pointer_cast(g_.data()), 
          thrust::raw_pointer_cast(H_.data()), 
          thrust::raw_pointer_cast(dSdF_.data()), 
          thrust::raw_pointer_cast(Jloc_.data()), 
          thrust::raw_pointer_cast(Aloc_.data()),
          thrust::raw_pointer_cast(vols_.data()),
          thrust::raw_pointer_cast(params_.mu.data()),
          thrust::raw_pointer_cast(params_.lambda.data())));
  OptimizerData::get().timer.stop("gradients", "MixedStretchGpu");

  // for (int i = 0; i < 50; ++i) {
  //   std::cout << "s_[i]: " << s_[i] << " vols_[i]: " << vols_[i] << " H_[i]: " << H_[i] << " g_[i] " << g_[i] << std::endl;
  // } exit(1);
  OptimizerData::get().timer.start("invert H", "MixedStretchGpu");
  // TODO Hinv is just 1/H (diagonal) for ARAP
  Hinv_gpu_.compute(
      thrust::raw_pointer_cast(H_.data()), 
      thrust::raw_pointer_cast(Hinv_.data()));
  OptimizerData::get().timer.stop("invert H", "MixedStretchGpu");

  // Compute RHS
  OptimizerData::get().timer.start("rhs-1", "MixedStretchGpu");
  thrust::for_each(thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(nelem_),
      rhs_functor(
          thrust::raw_pointer_cast(rhs_tmp_.data()), 
          thrust::raw_pointer_cast(s_.data()), 
          thrust::raw_pointer_cast(S_.data()), 
          thrust::raw_pointer_cast(g_.data()), 
          thrust::raw_pointer_cast(H_.data()), 
          thrust::raw_pointer_cast(dSdF_.data()), 
          thrust::raw_pointer_cast(vols_.data())));
  OptimizerData::get().timer.stop("rhs-1", "MixedStretchGpu");
  OptimizerData::get().timer.start("rhs-2", "MixedStretchGpu");
  double* rhs;
  JW_gpu_.product(thrust::raw_pointer_cast(rhs_tmp_.data()), &rhs);
  cudaMemcpy(thrust::raw_pointer_cast(rhs_.data()),
      rhs, rhs_.size()*sizeof(double), cudaMemcpyDeviceToDevice);
  OptimizerData::get().timer.stop("rhs-2", "MixedStretchGpu");

  // OptimizerData::get().timer.start("assemble1", "MixedStretchGpu");
  // // Copy Aloc from device to host
  // std::vector<double> Aloc_tmp(Aloc_.size());
  // cudaMemcpy(Aloc_tmp.data(), thrust::raw_pointer_cast(Aloc_.data()),
  //     Aloc_.size()*sizeof(double), cudaMemcpyDeviceToHost);

  // std::vector<MatrixXd> Aloc(nelem_);
  // for (int i = 0; i < nelem_; ++i) {
  //   Aloc[i] = Map<MatrixXd>(Aloc_tmp.data() + Aloc_N()*Aloc_N()*i,
  //       Aloc_N(), Aloc_N());
  // }
  // OptimizerData::get().timer.stop("assemble1", "MixedStretchGpu");
  // OptimizerData::get().timer.start("assemble2", "MixedStretchGpu");
  // assembler_->update_matrix(Aloc);
  // OptimizerData::get().timer.stop("assemble2", "MixedStretchGpu");

  OptimizerData::get().timer.start("assemble3", "MixedStretchGpu");
  assembler2_->update_matrix(Aloc_);
  OptimizerData::get().timer.stop("assemble3", "MixedStretchGpu");

  // std::cout << "Top left1 : \n" << assembler_->A.block(0,0,10,10) << std::endl;
  // std::cout << "Top left2: \n" << assembler2_->to_eigen_csr().block(0,0,10,10) << std::endl;
  // std::cout << "DIFF: " <<  (assembler_->A - assembler2_->to_eigen_csr()).norm() << std::endl;
  // OptimizerData::get().print_data(true);
  if constexpr (STORAGE == STORAGE_EIGEN) {
    cudaFree(d_x);
  }
}

template<int DIM, StorageType STORAGE>
void MixedStretchGpu<DIM,STORAGE>::solve(VectorType& dx) {
  OptimizerData::get().timer.start("solve", "MixedStretchGpu");
  // Copy dx to device
  double* d_dx;
  if constexpr (STORAGE == STORAGE_EIGEN) {
    cudaMalloc(&d_dx, dx.size()*sizeof(double));
    cudaMemcpy(d_dx, dx.data(), dx.size()*sizeof(double), cudaMemcpyHostToDevice);
  } else {
    d_dx = thrust::raw_pointer_cast(dx.data());
  }

  // Compute Jdx
  double* Jdx;
  JWT_gpu_.product(d_dx, &Jdx);

  // Compute ds
  thrust::for_each(thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(nelem_),
      solve_functor(Jdx, 
          thrust::raw_pointer_cast(S_.data()), 
          thrust::raw_pointer_cast(s_.data()), 
          thrust::raw_pointer_cast(g_.data()), 
          thrust::raw_pointer_cast(H_.data()), 
          thrust::raw_pointer_cast(dSdF_.data()), 
          thrust::raw_pointer_cast(ds_.data()), 
          thrust::raw_pointer_cast(la_.data()),
          thrust::raw_pointer_cast(vols_.data())));
  OptimizerData::get().timer.stop("solve", "MixedStretchGpu");

  if constexpr (STORAGE == STORAGE_EIGEN) {
    // Copy ds to host
    ds_h_.resize(ds_.size());
    cudaMemcpy(ds_h_.data(), thrust::raw_pointer_cast(ds_.data()),
        ds_.size()*sizeof(double), cudaMemcpyDeviceToHost);

    // Copy la to host
    la_h_.resize(la_.size());
    cudaMemcpy(la_h_.data(), thrust::raw_pointer_cast(la_.data()),
        la_.size()*sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "ds norm: " << ds_h_.norm() << " la norm: " << la_h_.norm() << std::endl;
    cudaFree(d_dx);
  }
}

template<int DIM, StorageType STORAGE>
MixedStretchGpu<DIM,STORAGE>::MixedStretchGpu(std::shared_ptr<Mesh> mesh)
    : MixedVariable<DIM,STORAGE>(mesh), Hinv_gpu_(mesh->T_.rows()) {
  nelem_ = mesh_->T_.rows();

  s_.resize(N()*nelem_);
  g_.resize(N()*nelem_);
  ds_.resize(N()*nelem_);
  la_.resize(N()*nelem_);
  R_.resize(M()*nelem_);
  S_.resize(N()*nelem_);
  H_.resize(N()*N()*nelem_);
  dSdF_.resize(N()*M()*nelem_);
  Hinv_.resize(N()*N()*nelem_);
  energy_tmp_.resize(nelem_);

  std::cout << "Assuming triangles in 2D and tetrahedra in 3D" << std::endl;
  Aloc_.resize(Aloc_N()*Aloc_N()*nelem_);
}

template<int DIM, StorageType STORAGE>
void MixedStretchGpu<DIM,STORAGE>::reset() {

  // Init s_h_ to identity
  s_h_.resize(N()*nelem_);
  for (int i = 0; i < nelem_; i++) {
    if constexpr (DIM == 2) {
      s_h_[N()*i] = 1;
      s_h_[N()*i + 1] = 1;
      s_h_[N()*i + 2] = 0;
    } else {
      s_h_[N()*i] = 1;
      s_h_[N()*i + 1] = 1;
      s_h_[N()*i + 2] = 1;
      s_h_[N()*i + 3] = 0;
      s_h_[N()*i + 4] = 0;
      s_h_[N()*i + 5] = 0;
    }
  }

  // Initialize mixed stretch variables to identity
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
  // For Tets each Jloc is 9 x 12
  Jloc_.resize(DIM*DIM*Aloc_N()*nelem_);
  const std::vector<MatrixXd>& Jloc = mesh_->local_jacobians();
  MatrixXd JlocHost(DIM*DIM*Aloc_N(), nelem_);
  for (int i = 0; i < nelem_; i++) {
    Map<const MatrixXd> Ji(Jloc[i].data(), DIM*DIM*Aloc_N(), 1);
    JlocHost.col(i) = Ji;
  }
  std::vector<double> JlocVec(JlocHost.data(),
      JlocHost.data() + DIM*DIM*Aloc_N()*nelem_);
  Jloc_ = JlocVec;

  // Initialize volumes thrust vector
  const VectorXd& vols = mesh_->volumes();
  vols_ = thrust::device_vector<double>(vols.begin(), vols.end());

  // Initialize deformation gradient jacobian matrix on the gpu
  const Eigen::SparseMatrix<double, RowMajor>& A =
      mesh_->template jacobian<JacobianType::FULL>();
  J_gpu_.init(A);

  // Weight jacobian matrix transposed initialization
  const Eigen::SparseMatrix<double, RowMajor>& PJW = mesh_->jacobian();
  JW_gpu_.init(PJW);
  JWT_gpu_.init(PJW.transpose());
  
  // initialize RHS vector
  rhs_.resize(PJW.rows());
  rhs_tmp_.resize(M() * nelem_);

  // initialize material parameters
  // Copy material parameters to host vectors
  thrust::host_vector<double> mu_h(nelem_);
  thrust::host_vector<double> lambda_h(nelem_);
  for (int i = 0; i < nelem_; i++) {
    // mesh_->material(i)->get_parameters(mu_h[i], lambda_h[i]);
    mu_h[i] = mesh_->elements_[i].material_->config()->mu;
    lambda_h[i] = mesh_->elements_[i].material_->config()->la;
  }
  params_.mu = mu_h;
  params_.lambda = lambda_h;

  // Init CPU assembler
  assembler_ = std::make_shared<Assembler<double,DIM,-1>>(
      mesh_->T_, mesh_->free_map_);

  std::cout << "Assembler 2: " <<std::endl;
  assembler2_ = std::make_shared<BlockMatrix<double,DIM,4>>(
      mesh_->T_, mesh_->free_map_);
      std::cout << "Assembler 2 done" <<std::endl;
  
}

template<int DIM, StorageType STORAGE>
MixedStretchGpu<DIM,STORAGE>::VectorType& MixedStretchGpu<DIM,STORAGE>::rhs()
{ 
  if constexpr (STORAGE == STORAGE_EIGEN) {
    rhs_h_.resize(rhs_.size());
    cudaMemcpy(rhs_h_.data(), thrust::raw_pointer_cast(rhs_.data()),
        rhs_.size()*sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << " rhs norm: " << rhs_h_.norm() << std::endl;
    return rhs_h_;
  } else {
    return rhs_;
  }
}

template<int DIM, StorageType STORAGE>
MixedStretchGpu<DIM,STORAGE>::VectorType&
MixedStretchGpu<DIM,STORAGE>::delta() {
  if constexpr (STORAGE == STORAGE_EIGEN) {
    ds_h_.resize(ds_.size());
    cudaMemcpy(ds_h_.data(), thrust::raw_pointer_cast(ds_.data()),
        ds_.size()*sizeof(double), cudaMemcpyDeviceToHost);
    return ds_h_;
  } else {
    return ds_;
  }
}

template<int DIM, StorageType STORAGE>
MixedStretchGpu<DIM,STORAGE>::VectorType&
MixedStretchGpu<DIM,STORAGE>::value() {
  if constexpr (STORAGE == STORAGE_EIGEN) {
    s_h_.resize(s_.size());
    cudaMemcpy(s_h_.data(), thrust::raw_pointer_cast(s_.data()),
        s_.size()*sizeof(double), cudaMemcpyDeviceToHost);
    return s_h_;
  } else {
    return s_;
  }
}

template<int DIM, StorageType STORAGE>
MixedStretchGpu<DIM,STORAGE>::VectorType&
 MixedStretchGpu<DIM,STORAGE>::lambda() {
  if constexpr (STORAGE == STORAGE_EIGEN) {
    std::cout << "Lambda not implemented for MixedStretchGpu" << std::endl;
    return dummy_;
  } else {
    return la_;
  }
}

template<int DIM, StorageType STORAGE>
MixedStretchGpu<DIM,STORAGE>::VectorType
MixedStretchGpu<DIM,STORAGE>::gradient() { 

  if constexpr (STORAGE == STORAGE_EIGEN) {
    VectorXd rhs(rhs_.size());
    cudaMemcpy(rhs.data(), thrust::raw_pointer_cast(rhs_.data()),
        rhs_.size()*sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "GPU - gradient is negated RHS norm: " << rhs.norm() << std::endl;

    return rhs;
  } else {
    return rhs_;
  }
}


template<int DIM, StorageType STORAGE>
void MixedStretchGpu<DIM,STORAGE>::post_solve() {
  thrust::fill(la_.begin(), la_.end(), 0.0);
}

template class mfem::MixedStretchGpu<3,STORAGE_THRUST>; // 3D
template class mfem::MixedStretchGpu<3,STORAGE_EIGEN>; // 3D