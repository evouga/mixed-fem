#include "mixed_stretch_gpu.h"
#include "mesh/mesh.h"
#include "utils/sparse_matrix_gpu.h"
#include "energies/material_model.h"
#include "optimizers/optimizer_data.h"
#include "svd/iARAP_gpu.cuh"
#include "svd/svd3_cuda_polar.cuh"
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

using namespace Eigen;
using namespace mfem;
using namespace thrust::placeholders;

namespace {
}

// Right now just ARAP energy supported on GPU
template<> __device__  
double MixedStretchGpu<3>::local_energy(const Vector6d& S, double mu) {
  return (mu*(pow(S(0)-1.0,2.0)+pow(S(1)-1.0,2.0)+pow(S(2)-1.0,2.0)
      +(S(3)*S(3))*2.0+(S(4)*S(4))*2.0+(S(5)*S(5))*2.0))/2.0;
}

template<> __device__  
Vector6d MixedStretchGpu<3>::local_gradient(const Vector6d& S, double mu) {
  Vector6d g;
  g(0) = (mu*(S(0)*2.0-2.0))/2.0;
  g(1) = (mu*(S(1)*2.0-2.0))/2.0;
  g(2) = (mu*(S(2)*2.0-2.0))/2.0;
  g(3) = S(3)*mu*2.0;
  g(4) = S(4)*mu*2.0;
  g(5) = S(5)*mu*2.0;
  return g;
}

template<> __device__  
Matrix6d MixedStretchGpu<3>::local_hessian(const Vector6d& S, double mu) {
  Vector6d tmp; tmp << 1,1,1,2,2,2;
  Matrix6d tmp2; tmp2.setZero();
  tmp2.diagonal() = tmp;
  return tmp2 * mu;
}

template<> __device__  
double MixedStretchGpu<2>::local_energy(const Vector3d& s, double mu) {
  return (mu*(pow(s(0)-1.0,2.0)+pow(s(1)-1.0,2.0)+(s(2)*s(2))*2.0))/2.0;
}

template<> __device__ 
Vector3d MixedStretchGpu<2>::local_gradient(const Vector3d& s, double mu) {
  Vector3d g;
  g(0) = (mu*(s(0)*2.0-2.0))/2.0;
  g(1) = (mu*(s(1)*2.0-2.0))/2.0;
  g(2) = s(2)*mu*2.0;
  return g;
}

template<> __device__ 
Matrix3d MixedStretchGpu<2>::local_hessian(const Vector3d& s, double mu) {
  Matrix3d H;
  H.setZero();
  H(0,0) = mu;
  H(1,1) = mu;
  H(2,2) = mu*2.0;
  return H;
}


template<int DIM> __device__
double MixedStretchGpu<DIM>::energy_functor::operator()(int i) const {
  Map<VecN> si(s + N()*i);
  Map<MatD> Fi(F + M()*i);
  Map<VecN> lai(la + N()*i);
  double vol = vols[i];
  double mu = 10344.8;
  double h2 = 0.0333 * 0.0333;

  double e = 0.0;
  // void svd_polar(const Eigen::Matrix3d& Ad, Eigen::Matrix3f& R) {
  if constexpr (DIM == 3) {
    Matrix<float,DIM,DIM> Rf;
// void svd_polar(const Eigen::Matrix3d& Ad, Eigen::Matrix3f& R) {

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
    // e = local_energy(si,mu);
    // e = vol * local_energy(si,mu); // fails
    // e = lai.dot(diff); // fails
    // e = vol * (lai.dot(diff)));
    e = vol * (lai.dot(diff)/h2 + local_energy(si, mu));
  }
  return e;
  
}

template<int DIM> template<bool COMPUTE_GRADIENTS> __device__
void MixedStretchGpu<DIM>::
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
    // if (i < 3) {
    //   printf("Si = %f %f %f %f %f %f\n", Si(0), Si(1), Si(2), Si(3), Si(4), Si(5));
    //   // Print dSdFf
    //   printf("dSdFf:\n");
    //   printf("1%f %f %f %f %f %f %f %f %f\n", dSdFf(0,0), dSdFf(0,1), dSdFf(0,2), dSdFf(0,3), dSdFf(0,4), dSdFf(0,5), dSdFf(0,6), dSdFf(0,7), dSdFf(0,8));
    //   printf("2%f %f %f %f %f %f %f %f %f\n", dSdFf(1,0), dSdFf(1,1), dSdFf(1,2), dSdFf(1,3), dSdFf(1,4), dSdFf(1,5), dSdFf(1,6), dSdFf(1,7), dSdFf(1,8));
    //   printf("3%f %f %f %f %f %f %f %f %f\n", dSdFf(2,0), dSdFf(2,1), dSdFf(2,2), dSdFf(2,3), dSdFf(2,4), dSdFf(2,5), dSdFf(2,6), dSdFf(2,7), dSdFf(2,8));
    //   printf("4%f %f %f %f %f %f %f %f %f\n", dSdFf(3,0), dSdFf(3,1), dSdFf(3,2), dSdFf(3,3), dSdFf(3,4), dSdFf(3,5), dSdFf(3,6), dSdFf(3,7), dSdFf(3,8));
    //   printf("5%f %f %f %f %f %f %f %f %f\n", dSdFf(4,0), dSdFf(4,1), dSdFf(4,2), dSdFf(4,3), dSdFf(4,4), dSdFf(4,5), dSdFf(4,6), dSdFf(4,7), dSdFf(4,8));
    //   printf("6%f %f %f %f %f %f %f %f %f\n", dSdFf(5,0), dSdFf(5,1), dSdFf(5,2), dSdFf(5,3), dSdFf(5,4), dSdFf(5,5), dSdFf(5,6), dSdFf(5,7), dSdFf(5,8));
    //   printf("\n\n");
    // }
  }
}

template<int DIM> __device__
void MixedStretchGpu<DIM>::derivative_functor::operator()(int i) const {
  double vol = vols[i];
  Map<VecN> si(s + N()*i);
  Map<VecN> gi(g + N()*i);
  Map<MatN> Hi(H + N()*N()*i);
  Map<MatN> Hinvi(Hinv + N()*N()*i);
  Map<MatMN> dSdFi(dSdF + M()*N()*i);

  Map<Matrix<double, Aloc_N(), Aloc_N()>> Aloci(Aloc + Aloc_N()*Aloc_N()*i);
  Map<Matrix<double, M(), Aloc_N()>> Jloci(Jloc + M()*Aloc_N()*i);
  
  // TODO input material parameters
  // Look at how AMGCL does runtime selection of GPU stuff...
  double mu = 10344.8;
  double h2 = 0.0333 * 0.0333;
  // MatN syminv = Syminv();
  Hi = h2 * local_hessian(si, mu);
  gi = h2 * local_gradient(si, mu);
  // Aloci = vol * Jloci.transpose() * (dSdFi 
    // * (syminv * Hi * syminv) * dSdFi.transpose()) * Jloci;
  Aloci = vol * Jloci.transpose() * (dSdFi 
    * Hi * dSdFi.transpose()) * Jloci;
}

template<int DIM> __device__
void MixedStretchGpu<DIM>::rhs_functor::operator()(int i) const {
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

template<int DIM> __device__
void MixedStretchGpu<DIM>::solve_functor::operator()(int i) const {
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

template<int DIM>
double MixedStretchGpu<DIM>::energy(const VectorXd& x, const VectorXd& s) {
  OptimizerData::get().timer.start("energy-transfer", "MixedStretchGpu");

  // copy x and s to the GPU
  double* d_x; // device x vector
  double* d_s; // device s vector
  cudaMalloc((void**)&d_x, sizeof(double) * x.size());
  cudaMalloc((void**)&d_s, sizeof(double) * s.size());
  cudaMemcpy(d_x, x.data(), sizeof(double) * x.size(),
      cudaMemcpyHostToDevice);
  cudaMemcpy(d_s, s.data(), sizeof(double) * s.size(),
      cudaMemcpyHostToDevice);
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
        thrust::raw_pointer_cast(vols_.data())),
        0.0, thrust::plus<double>());
  OptimizerData::get().timer.stop("energy", "MixedStretchGpu");
  cudaFree(d_x);
  cudaFree(d_s);

  return e;
}

template<int DIM>
void MixedStretchGpu<DIM>::update(const VectorXd& x, double dt) {

  // copy host s to gpu s
  cudaMemcpy(thrust::raw_pointer_cast(s_.data()), s_h_.data(), sizeof(double) * s_.size(),
    cudaMemcpyHostToDevice);

  VectorXd def_grad;
  mesh_->deformation_gradient(x, def_grad);

  vector<double> h2(1, dt*dt);

  // First compute the deformation gradient
  OptimizerData::get().timer.start("def_grad_transfer", "MixedStretchGpu");
  double* d_x; // device x vector
  cudaMalloc((void**)&d_x, sizeof(double) * x.size());
  cudaMemcpy(d_x, x.data(), sizeof(double) * x.size(),
      cudaMemcpyHostToDevice);
  OptimizerData::get().timer.stop("def_grad_transfer", "MixedStretchGpu");
  OptimizerData::get().timer.start("def_grad", "MixedStretchGpu");
  double* F;
  J_gpu_.product(d_x, &F);
  OptimizerData::get().timer.stop("def_grad", "MixedStretchGpu");

  // Copy the result back to the host
  VectorXd F3(J_gpu_.rows());
  cudaMemcpy(F3.data(), F, sizeof(double)*J_gpu_.rows(), cudaMemcpyDeviceToHost);
  std::cout << "F3 - def_grad: " << (F3 - def_grad).norm() << std::endl;

  // make the host block until the device is finished with foo
  // check for error
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

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
  // for (int i = 0; i < 100; ++i) {
  //   std::cout << "R: " << R_[i] << " F: " << F3[i] << std::endl;
  // }
  // exit(0);

  std::cout << "Update local hessians and gradient" << std::endl;
  OptimizerData::get().timer.start("gradients", "MixedStretchGpu");
  // Compute local hessians and gradients
  thrust::for_each(thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(nelem_),
      derivative_functor(
          thrust::raw_pointer_cast(s_.data()), 
          thrust::raw_pointer_cast(g_.data()), 
          thrust::raw_pointer_cast(H_.data()), 
          thrust::raw_pointer_cast(Hinv_.data()), 
          thrust::raw_pointer_cast(dSdF_.data()), 
          thrust::raw_pointer_cast(Jloc_.data()), 
          thrust::raw_pointer_cast(Aloc_.data()),
          thrust::raw_pointer_cast(vols_.data())));
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

  OptimizerData::get().timer.start("assemble1", "MixedStretchGpu");
  // Copy Aloc from device to host
  std::vector<double> Aloc_tmp(Aloc_.size());
  cudaMemcpy(Aloc_tmp.data(), thrust::raw_pointer_cast(Aloc_.data()),
      Aloc_.size()*sizeof(double), cudaMemcpyDeviceToHost);

  std::vector<MatrixXd> Aloc(nelem_);
  for (int i = 0; i < nelem_; ++i) {
    Aloc[i] = Map<MatrixXd>(Aloc_tmp.data() + Aloc_N()*Aloc_N()*i,
        Aloc_N(), Aloc_N());
  }
  OptimizerData::get().timer.stop("assemble1", "MixedStretchGpu");
  OptimizerData::get().timer.start("assemble2", "MixedStretchGpu");
  assembler_->update_matrix(Aloc);
  OptimizerData::get().timer.stop("assemble2", "MixedStretchGpu");


  std::cout << "Done update " << std::endl;
  OptimizerData::get().print_data(true);

  cudaFree(d_x);
}


template<int DIM>
void MixedStretchGpu<DIM>::solve(const Eigen::VectorXd& dx) {
  OptimizerData::get().timer.start("solve", "MixedStretchGpu");
  // Copy dx to device
  double* d_dx;
  cudaMalloc(&d_dx, dx.size()*sizeof(double));
  cudaMemcpy(d_dx, dx.data(), dx.size()*sizeof(double), cudaMemcpyHostToDevice);

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

  // Copy ds to host
  ds_h_.resize(ds_.size());
  cudaMemcpy(ds_h_.data(), thrust::raw_pointer_cast(ds_.data()),
      ds_.size()*sizeof(double), cudaMemcpyDeviceToHost);

  // std::cout << " gpu solve norm: " << ds_h_.norm() << "\n " << ds_h_.head(100).transpose() << std::endl;
  cudaFree(d_dx);
}



template<int DIM>
MixedStretchGpu<DIM>::MixedStretchGpu(std::shared_ptr<Mesh> mesh)
    : MixedVariable<DIM>(mesh), Hinv_gpu_(mesh->T_.rows()) {
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
  int N_loc; // size of local stiffness matrix
  if constexpr (DIM == 2) {
    N_loc = DIM * 3;
  } else {
    N_loc = DIM * 4;
  }
  Aloc_.resize(N_loc*N_loc*nelem_);

  // For Tets each Jloc is 9 x 12
  Jloc_.resize(DIM*DIM*N_loc*nelem_);
  const std::vector<MatrixXd>& Jloc = mesh_->local_jacobians();
  MatrixXd JlocHost(DIM*DIM*N_loc, nelem_);
  for (int i = 0; i < nelem_; i++) {
    Map<const MatrixXd> Ji(Jloc[i].data(), DIM*DIM*N_loc, 1);
    JlocHost.col(i) = Ji;
  }
  std::vector<double> JlocVec(JlocHost.data(), JlocHost.data() + DIM*DIM*N_loc*nelem_);
  Jloc_ = JlocVec;

  // Initialize volumes thrust vector
  const VectorXd& vols = mesh_->volumes();
  vols_ = thrust::device_vector<double>(vols.begin(), vols.end());

  // Initialize deformation gradient matrix on the gpu
  const Eigen::SparseMatrix<double, RowMajor>& A =
      mesh_->template jacobian<JacobianType::FULL>();
  J_gpu_.init(A);

  const Eigen::SparseMatrix<double, RowMajor>& PJW = mesh_->jacobian();
  JW_gpu_.init(PJW);
  JWT_gpu_.init(PJW.transpose());

  rhs_.resize(PJW.rows());
  rhs_tmp_.resize(M() * nelem_);

  // Init CPU assembler
  assembler_ = std::make_shared<Assembler<double,DIM,-1>>(
      mesh_->T_, mesh_->free_map_);
}

template<int DIM>
void MixedStretchGpu<DIM>::reset() {

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
}

template<int DIM>
Eigen::VectorXd MixedStretchGpu<DIM>::rhs() { 
  VectorXd rhs(rhs_.size());
  cudaMemcpy(rhs.data(), thrust::raw_pointer_cast(rhs_.data()),
      rhs_.size()*sizeof(double), cudaMemcpyDeviceToHost);
  
  std::cout << "GPU RHS norm: " << rhs.norm() << std::endl;
  return rhs;
}

template<int DIM>
Eigen::VectorXd& MixedStretchGpu<DIM>::delta() {
  // std::cout << "Delta not implemented for MixedStretchGpu" << std::endl;
  ds_h_.resize(ds_.size());
  cudaMemcpy(ds_h_.data(), thrust::raw_pointer_cast(ds_.data()),
      ds_.size()*sizeof(double), cudaMemcpyDeviceToHost);
  return ds_h_;
}

template<int DIM>
Eigen::VectorXd& MixedStretchGpu<DIM>::value() {
  // std::cout << "Value not implemented for MixedStretchGpu" << std::endl;
  s_h_.resize(s_.size());
  cudaMemcpy(s_h_.data(), thrust::raw_pointer_cast(s_.data()),
      s_.size()*sizeof(double), cudaMemcpyDeviceToHost);
  return s_h_;
}

template<int DIM>
Eigen::VectorXd& MixedStretchGpu<DIM>::lambda() {
  std::cout << "Lambda not implemented for MixedStretchGpu" << std::endl;
  return dummy_;
}

template<int DIM>
void MixedStretchGpu<DIM>::post_solve() {
  thrust::fill(la_.begin(), la_.end(), 0.0);
}

template class mfem::MixedStretchGpu<3>; // 3D
template class mfem::MixedStretchGpu<2>; // 2D