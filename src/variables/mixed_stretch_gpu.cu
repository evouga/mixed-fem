#include "mixed_stretch_gpu.h"
#include "mesh/mesh.h"
#include "utils/sparse_matrix_gpu.h"
#include "energies/material_model.h"
#include "optimizers/optimizer_data.h"
#include "svd/iARAP_gpu.cuh"

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


template<int DIM> template<bool COMPUTE_GRADIENTS> __device__
void MixedStretchGpu<DIM>::
rotation_functor<COMPUTE_GRADIENTS>::operator()(int i) const {

  Map<MatD> Fi(F + M()*i);
  Map<MatD> Ri(R + M()*i);
  Map<VecN> Si(S + N()*i);
  Map<MatMN> dSdFi(dSdF + M()*N()*i);

  if constexpr (DIM == 3) {
    MatM dRdF;
    // if (i < 100)
        // printf("COMPUTE_GRADIENTS = %d", COMPUTE_GRADIENTS);
    Ri = rotation<double>(Fi, COMPUTE_GRADIENTS, dRdF);
    MatD S3D = Ri.transpose() * Fi;
    Si << S3D(0,0), S3D(1,1), S3D(2,2),
        0.5*(S3D(1,0) + S3D(0,1)),
        0.5*(S3D(2,0) + S3D(0,2)),
        0.5*(S3D(2,1) + S3D(1,2));
    // if (i < 100) {
    //   printf("Si = %f %f %f %f %f %f", Si(0), Si(1), Si(2), Si(3), Si(4), Si(5));
    // }
  }
  // Ri.setIdentity();
  // Eigen::Matrix3d rotation(const Eigen::Matrix3d &F);

  // double J = Fi.determinant();
  // printf("J = %f",J);
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
  MatN syminv = Syminv();
  Hi = h2 * (syminv * local_hessian(si, mu) * syminv) / vol;
  gi = h2 * local_gradient(si, mu);
  Aloci = vol * vol * Jloci.transpose() 
      * (dSdFi * Hi * dSdFi.transpose()) * Jloci;
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
}

template<int DIM>
void MixedStretchGpu<DIM>::update(const Eigen::VectorXd& x, double dt) {
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
  // Copy the result back to the host
  double* F;
  J_gpu_.product(d_x, &F);
  OptimizerData::get().timer.stop("def_grad", "MixedStretchGpu");


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
  for (int i = 0; i < 100; ++i) {
    std::cout << "R: " << R_[i] << " F: " << F3[i] << std::endl;
  }
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
  std::cout << "Done update " << std::endl;
  OptimizerData::get().timer.stop("invert H", "MixedStretchGpu");
  OptimizerData::get().print_data(true);

  cudaFree(d_x);
}


template<int DIM>
void MixedStretchGpu<DIM>::reset() {

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

  // Initialize volumes thrust vector
  const VectorXd& vols = mesh_->volumes();
  vols_ = thrust::device_vector<double>(vols.begin(), vols.end());

  // Initialize deformation gradient matrix on the gpu
  Eigen::SparseMatrix<double, RowMajor> A =
      mesh_->template jacobian<JacobianType::FULL>();
  J_gpu_.init(A);

  // Init CPU assembler
  assembler_ = std::make_shared<Assembler<double,DIM,-1>>(
      mesh_->T_, mesh_->free_map_);
}
template class mfem::MixedStretchGpu<3>; // 3D
template class mfem::MixedStretchGpu<2>; // 2D