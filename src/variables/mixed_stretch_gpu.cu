#include "mixed_stretch_gpu.h"
#include "mesh/mesh.h"
#include "utils/sparse_matrix_gpu.h"
#include "energies/material_model.h"
#include "optimizers/optimizer_data.h"

using namespace Eigen;
using namespace mfem;
using namespace thrust::placeholders;

// Right now just ARAP energy supported on GPU
template<> __device__  
double MixedStretchGpu<3>::local_energy(const Vector6d& S, double mu) {
  return (mu*(pow(S(0)-1.0,2.0)+pow(S(1)-1.0,2.0)+pow(S(2)-1.0,2.0)+(S(3)*S(3))*2.0
      +(S(4)*S(4))*2.0+(S(5)*S(5))*2.0))/2.0;
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
  return tmp.asDiagonal() * mu;
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

template<int DIM>
MixedStretchGpu<DIM>::MixedStretchGpu(std::shared_ptr<Mesh> mesh)
    : MixedVariable<DIM>(mesh), Hinv_gpu_(mesh->T_.rows()) {
  nelem_ = mesh_->T_.rows();

  s_.resize(N()*nelem_);
  g_.resize(N()*nelem_);
  ds_.resize(N()*nelem_);
  la_.resize(N()*nelem_);
  R_.resize(M()*nelem_);
  S_.resize(N()*N()*nelem_);
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
__device__ 
void MixedStretchGpu<DIM>::local_derivatives(int i, double* s, double* g,
    double* H, double* Hinv, double* dSdF, double* Jloc, double* Aloc,
    double* vols) {

  double vol = vols[i];
  Map<VecN> si(s + N()*i);
  Map<VecN> gi(g + N()*i);
  Map<MatN> Hi(H + N()*N()*i);
  Map<MatN> Hinvi(H + N()*N()*i);
  Map<MatN> dSdFi(H + M()*N()*i);
  
  // TODO input material parameters
  // Look at how AMGCL does runtime selection of GPU stuff...
  double mu = 10344.8;
  double h2 = 0.0333 * 0.0333;

  Hi = h2 * (Syminv() * local_hessian(si, mu) * Syminv()) / vol;
  gi = h2 * local_gradient(si, mu);
  // Hinvi = Hi.inverse();
  // Inverse of 6x6 matrix in cuda device code

  // Hi = mesh_->elements_[i].material_->hessian(si);

    // MatN H = h2 * mesh_->elements_[i].material_->hessian(si);
    // g_[i] = h2 * mesh_->elements_[i].material_->gradient(si);
    // H_[i] = (1.0 / vol) * (Syminv() * H * Syminv());
}


template<int DIM>
void MixedStretchGpu<DIM>::update(const Eigen::VectorXd& x, double dt) {
  std::cout << "Update 1" << std::endl;
  VectorXd def_grad;
  mesh_->deformation_gradient(x, def_grad);

  vector<double> h2(1, dt*dt);

  // First compute the deformation gradient
  OptimizerData::get().timer.start("def_grad", "MixedStretchGpu");

  double* d_x; // device x vector
  cudaMalloc((void**)&d_x, sizeof(double) * x.size());
  cudaMemcpy(d_x, x.data(), sizeof(double) * x.size(),
      cudaMemcpyHostToDevice);

  // Copy the result back to the host
  double* F;
  J_gpu_.product(d_x, &F);
  OptimizerData::get().timer.stop("def_grad", "MixedStretchGpu");


  VectorXd F3(J_gpu_.rows());
  cudaMemcpy(F3.data(), F, sizeof(double)*J_gpu_.rows(), cudaMemcpyDeviceToHost);
  std::cout << "F3 - def_grad: " << (F3 - def_grad).norm() << std::endl;

  // TODO rotations
  // 
  std::cout << "Update local hessians and gradient" << std::endl;
  OptimizerData::get().timer.start("gradients", "MixedStretchGpu");
  // Compute local hessians and gradients
  thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(nelem_),
      std::bind(&MixedStretchGpu::local_derivatives, this, std::placeholders::_1,
          thrust::raw_pointer_cast(s_.data()), 
          thrust::raw_pointer_cast(g_.data()), 
          thrust::raw_pointer_cast(H_.data()), 
          thrust::raw_pointer_cast(Hinv_.data()), 
          thrust::raw_pointer_cast(dSdF_.data()), 
          thrust::raw_pointer_cast(Jloc_.data()), 
          thrust::raw_pointer_cast(Aloc_.data()),
          thrust::raw_pointer_cast(vols_.data())
      ));
  OptimizerData::get().timer.stop("gradients", "MixedStretchGpu");

  OptimizerData::get().timer.start("invert H", "MixedStretchGpu");

  Hinv_gpu_.compute(thrust::raw_pointer_cast(H_.data()), 
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

  // std::cout << " Nelem: " << nelem_ << 
  // Aloc_.resize(nelem_);
  // evals_.resize(nelem_);
  // assembler_ = std::make_shared<Assembler<double,DIM,-1>>(
  //     mesh_->T_, mesh_->free_map_);
}
template class mfem::MixedStretchGpu<3>; // 3D
template class mfem::MixedStretchGpu<2>; // 2D
