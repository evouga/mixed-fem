#include "mixed_friction_gpu.h"
#include "mesh/mesh.h"
#include "optimizers/optimizer_data.h"
#include <ipc/barrier/barrier.hpp>
#include <thrust/host_vector.h>

using namespace Eigen;
using namespace mfem;
\
template<int DIM, StorageType STORAGE> __device__
void MixedFrictionGpu<DIM,STORAGE>::derivative_functor::operator()(int i)
    const {
  Map<Matrix<double, Aloc_N(), Aloc_N()>> Aloc_i(Aloc + Aloc_N()*Aloc_N()*i);
  Map<Matrix<double, Aloc_N(), 1>> Gx_i(Gx + Aloc_N()*i);
  Aloc_i = Gx_i * Gx_i.transpose() * H[i];
}

template<int DIM, StorageType STORAGE> __device__
void MixedFrictionGpu<DIM,STORAGE>::rhs_functor::operator()(int i)
    const {
  Map<Matrix<double, Aloc_N(), 1>> Gx_i(Gx + Aloc_N()*i);
  Map<Vector4i> Ti(T + 4*i);

  double tmp = -(H[i] * (D[i] - d[i]) + g[i]);

  Matrix<double, Aloc_N(), 1> gloc = Gx_i * tmp;

  // Atomic add to rhs
  for (int j = 0; j < 4; ++j) {
    if (Ti[j] < 0) {
      break;
    }
    int idx = free_map[Ti(j)];

    // Atomic add to rhs
    if (idx >= 0) {
      atomicAdd(rhs + DIM*idx, gloc(j*DIM));
      atomicAdd(rhs + DIM*idx + 1, gloc(j*DIM + 1));
      if constexpr (DIM == 3) {
        atomicAdd(rhs + DIM*idx + 2, gloc(j*DIM + 2));
      }
    }
  }
}

template<int DIM, StorageType STORAGE> __device__
void MixedFrictionGpu<DIM,STORAGE>::solve_functor::operator()(int i)
    const {
  Map<Vector4i> Ti(T + 4*i);

  Map<Matrix<double, Aloc_N(), 1>> Gx_i(Gx + Aloc_N()*i);

  // Compute Gdx
  Vector12d qi = Vector12d::Zero();
  for (int j = 0; j < 4; ++j) {
    if (Ti[j] < 0) {
      break;
    }
    int idx = free_map[Ti(j)];
    if (idx >= 0) {
      qi(j*DIM) = dx[DIM*idx];
      qi(j*DIM + 1) = dx[DIM*idx + 1];
      if constexpr (DIM == 3) {
        qi(j*DIM + 2) = dx[DIM*idx + 2];
      }
    }
  }
  double Gdx = Gx_i.dot(qi);
  delta[i] = (D[i] - d[i] + Gdx);
  la[i] = H[i] * delta[i] + g[i];
}

template<int DIM, StorageType STORAGE>
MixedFrictionGpu<DIM,STORAGE>::MixedFrictionGpu(
    std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
    : MixedVariable<DIM,STORAGE>(mesh), config_(config) {
  // nelem_ = mesh_->T_.rows();

}

template<int DIM, StorageType STORAGE>
double MixedFrictionGpu<DIM,STORAGE>::energy(VectorType& x,
    VectorType& d) {

  if (constraints_.size() == 0) {
    return 0.0;
  }
  OptimizerData::get().timer.start("energy", "MixedFrictionGpu");

  Eigen::VectorXd x_h;
  Eigen::VectorXd d_h;

  if constexpr(STORAGE == STORAGE_THRUST) {
    // copy x to host
    x_h.resize(x.size());
    cudaMemcpy(x_h.data(), thrust::raw_pointer_cast(x.data()),
        x.size()*sizeof(double), cudaMemcpyDeviceToHost);

    // copy d to host
    d_h.resize(d.size());
    cudaMemcpy(d_h.data(), thrust::raw_pointer_cast(d.data()),
        d.size()*sizeof(double), cudaMemcpyDeviceToHost);
  } else {
    x_h = x;
    d_h = d;
  }

  // Convert configuration vector to matrix form
  MatrixXd V = Map<const MatrixXd>(x_h.data(), DIM, mesh_->V_.rows());
  V.transposeInPlace();
  
  // Get IPC mesh and face/edge/vertex data
  const auto& ipc_mesh = mesh_->collision_mesh();
  const Eigen::MatrixXi& E = ipc_mesh.edges();
  const Eigen::MatrixXi& F = ipc_mesh.faces();
  MatrixXd U = ipc_mesh.vertices(V) - V0_; // surface vertex set


  VectorXd e_vec(constraints_.size());
  #pragma omp parallel for
  for (size_t i = 0; i < e_vec.size(); ++i) {
    double u_norm = constraints_[i].u_norm(U,E,F);
    e_vec(i) = constraints_[i].potential(d_h(i), config_->espv * dt_ * dt_)
        + la_h_(i) * (u_norm - d_h(i));
  }
  double e = e_vec.sum() / dt_ / dt_;
  OptimizerData::get().timer.stop("energy", "MixedFrictionGpu");
  return e;
}

template<int DIM, StorageType STORAGE>
void MixedFrictionGpu<DIM,STORAGE>::update(VectorType& x, double dt) {
  // Get collision frames
  dt_ = dt;

  int num_frames = constraints_.size();
  if (num_frames == 0) {
    return;
  }

  OptimizerData::get().timer.start("update-constraints", "MixedFrictionGpu");

  Eigen::VectorXd x_h;
  if constexpr(STORAGE == STORAGE_THRUST) {
    // copy x to host
    x_h.resize(x.size());
    cudaMemcpy(x_h.data(), thrust::raw_pointer_cast(x.data()),
        x.size()*sizeof(double), cudaMemcpyDeviceToHost);
  } else {
    x_h = x;
  }

  // Convert configuration vector to matrix form
  MatrixXd V = Map<const MatrixXd>(x_h.data(), DIM, mesh_->V_.rows());
  V.transposeInPlace();

  // Getting IPC mesh data
  const auto& ipc_mesh = mesh_->collision_mesh();
  const Eigen::MatrixXi& E = ipc_mesh.edges();
  const Eigen::MatrixXi& F = ipc_mesh.faces();

  // Get displacements
  MatrixXd U = ipc_mesh.vertices(V) - V0_;

  // Copy device d_ and la_ to host
  if constexpr(STORAGE == STORAGE_THRUST) {
    cudaMemcpy(z_h_.data(), thrust::raw_pointer_cast(z_.data()),
        z_.size()*sizeof(double), cudaMemcpyDeviceToHost);
  }
  Gx_h_.setZero();

  double epsv_h = config_->espv * dt_ * dt_;
  // Rebuilding mixed variables for the new set of collision frames.
  #pragma omp parallel for
  for (int i = 0; i < num_frames; ++i) {
    Z_h_(i) = constraints_[i].u_norm(U, E, F);
    ipc::VectorMax12d grad = constraints_[i].u_norm_gradient(U, E, F);
    Gx_h_.segment(12*i, grad.size()) = grad;
    g_h_(i) = constraints_[i].potential_gradient(z_h_(i), epsv_h);
    H_h_(i) = constraints_[i].potential_hessian(z_h_(i), epsv_h);
  }
  // std::cout << "GPU \n" << std::endl;
  // std::cout << "Z_h_ \n" << Z_h_.transpose() << std::endl;
  // std::cout << "z_h_ \n" << z_h_.transpose() << std::endl;

  // Copy Z_h_ and Gx_ to device
  cudaMemcpy(thrust::raw_pointer_cast(Z_.data()), Z_h_.data(),
      Z_h_.size()*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(thrust::raw_pointer_cast(Gx_.data()), Gx_h_.data(),
      Gx_h_.size()*sizeof(double), cudaMemcpyHostToDevice);
  // Copy g_ and H_ to host
  cudaMemcpy(thrust::raw_pointer_cast(g_.data()), g_h_.data(),
      g_h_.size()*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(thrust::raw_pointer_cast(H_.data()), H_h_.data(),
      H_h_.size()*sizeof(double), cudaMemcpyHostToDevice);
  
  OptimizerData::get().timer.stop("update-constraints", "MixedFrictionGpu");

  // std::cout << "d_ size " << d_.size() << std::endl;
  // std::cout << "la_ size " << la_.size() << std::endl;
  // std::cout << "Gx_ size " << Gx_.size() << std::endl;
  // std::cout << "Aloc_ size " << Aloc_.size() << std::endl;
  // std::cout << "params size " << params.size() << std::endl;
  // std::cout << "g_ size " << g_.size() << std::endl;
  // std::cout << "H_ size " << H_.size() << std::endl;

  // Compute derivatives and right-hand-side
  OptimizerData::get().timer.start("derivatives", "MixedFrictionGpu");
  thrust::for_each(thrust::counting_iterator(0),
      thrust::counting_iterator(num_frames),
      derivative_functor(
        thrust::raw_pointer_cast(H_.data()),
        thrust::raw_pointer_cast(Gx_.data()),
        thrust::raw_pointer_cast(Aloc_.data())
      ));

  OptimizerData::get().timer.stop("derivatives", "MixedFrictionGpu");

  OptimizerData::get().timer.start("rhs", "MixedFrictionGpu");
  thrust::fill(rhs_.begin(), rhs_.end(), 0.0);
  thrust::for_each(thrust::counting_iterator(0),
      thrust::counting_iterator(num_frames),
      rhs_functor(
        thrust::raw_pointer_cast(T_.data()),
        thrust::raw_pointer_cast(free_map_.data()),
        thrust::raw_pointer_cast(z_.data()),
        thrust::raw_pointer_cast(Z_.data()),
        thrust::raw_pointer_cast(g_.data()),
        thrust::raw_pointer_cast(H_.data()),
        thrust::raw_pointer_cast(Gx_.data()),
        thrust::raw_pointer_cast(rhs_.data())
      ));
  OptimizerData::get().timer.stop("rhs", "MixedFrictionGpu");

  OptimizerData::get().timer.start("assemble", "MixedFrictionGpu");
  assembler_->update_matrix(Aloc_);
  OptimizerData::get().timer.stop("assemble", "MixedFrictionGpu");

  OptimizerData::get().timer.start("csr", "MixedFrictionGpu");
  if (exec_) {
    if (!A_) {
      A_ = gko::matrix::Csr<double,int>::create(exec_);
    }
    // Create gko fbcsr matrix
    auto hessian = gko::matrix::Fbcsr<double,int>::create_const(exec_,
      gko::dim<2>{assembler_->size()}, DIM, 
      gko::array<double>::const_view(exec_, assembler_->num_values(),
          assembler_->values()),
      gko::array<int>::const_view(exec_, assembler_->num_col_indices(),
          assembler_->col_indices()),
      gko::array<int>::const_view(exec_, assembler_->num_row_blocks(),
          assembler_->row_offsets()));
    hessian->convert_to(A_.get());
  } else {
    std::cout << "No exec!" << std::endl;
  }
  OptimizerData::get().timer.stop("csr", "MixedFrictionGpu");
}

template<int DIM, StorageType STORAGE>
void MixedFrictionGpu<DIM,STORAGE>::apply_submatrix(double* x, const double* b,
    int cols, int start, int end) {
  if (exec_ && A_ && constraints_.size() > 0) {
    auto sub_A = A_->create_submatrix({start,end},{start,end});
    int nrows = end - start;
    auto x_gko = gko::matrix::Dense<double>::create(exec_,
      gko::dim<2>{nrows, cols},
      gko::array<double>::view(exec_, nrows*cols, x), cols);
    auto b_gko = gko::matrix::Dense<double>::create_const(exec_,
      gko::dim<2>{nrows, cols},
      gko::array<double>::const_view(exec_, nrows*cols, b), cols);
    sub_A->apply(b_gko.get(), x_gko.get());
  }
}


template<int DIM, StorageType STORAGE>
void MixedFrictionGpu<DIM,STORAGE>::solve(VectorType& dx) {
  int num_frames = constraints_.size();

  if (num_frames == 0) {
    return;
  }
  OptimizerData::get().timer.start("solve", "MixedFrictionGpu");

  // Copy dx to device
  double* d_dx;
  if constexpr (STORAGE == STORAGE_EIGEN) {
    cudaMalloc(&d_dx, dx.size()*sizeof(double));
    cudaMemcpy(d_dx, dx.data(), dx.size()*sizeof(double),
        cudaMemcpyHostToDevice);
  } else {
    d_dx = thrust::raw_pointer_cast(dx.data());
  }
  thrust::for_each(thrust::counting_iterator(0),
      thrust::counting_iterator(num_frames),
      solve_functor(
        thrust::raw_pointer_cast(T_.data()),
        thrust::raw_pointer_cast(free_map_.data()),
        thrust::raw_pointer_cast(z_.data()),
        thrust::raw_pointer_cast(Z_.data()),
        thrust::raw_pointer_cast(g_.data()),
        thrust::raw_pointer_cast(H_.data()),
        thrust::raw_pointer_cast(Gx_.data()),
        thrust::raw_pointer_cast(la_.data()),
        thrust::raw_pointer_cast(delta_.data()), d_dx
      ));
  // Copy la to la_h_
  cudaMemcpy(la_h_.data(),  thrust::raw_pointer_cast(la_.data()),
      la_.size()*sizeof(double), cudaMemcpyDeviceToHost);
  OptimizerData::get().timer.stop("solve", "MixedFrictionGpu");
}

template<int DIM, StorageType STORAGE>
void MixedFrictionGpu<DIM,STORAGE>::pre_solve() {
  // Convert configuration vector to matrix form
  V0_ = mesh_->V_;

  // Get IPC mesh and face/edge/vertex data
  const auto& ipc_mesh = mesh_->collision_mesh();
  const Eigen::MatrixXi& E = ipc_mesh.edges();
  const Eigen::MatrixXi& F = ipc_mesh.faces();
  V0_ = ipc_mesh.vertices(V0_);

  // Computing collision constraints
  ipc::Constraints constraints;
  ipc::construct_constraint_set(mesh_->collision_candidates(),
      ipc_mesh, V0_, config_->dhat, constraints);
  ipc::construct_friction_constraint_set(
      ipc_mesh, V0_, constraints,
      config_->dhat, config_->kappa,
      config_->mu, constraints_);

  size_t num_frames = constraints_.size();

  // Create element matrix for assemblers
  T_h_.resize(num_frames, 4);
  for (size_t i = 0; i < constraints_.size(); ++i) {
    std::array<long, 4> ids = constraints_[i].vertex_indices(E, F);
    for (int j = 0; j < 4; ++j) {
      T_h_(i,j) = -1;
      if (ids[j] != -1) {
        T_h_(i,j) = ipc_mesh.to_full_vertex_id(ids[j]);
      }
    }
  }

  // Initialize host mixed variables
  z_h_.resize(num_frames);
  z_h_.setZero();
  Z_h_.resize(num_frames);
  la_h_.resize(num_frames);
  Gx_h_.resize(num_frames*12);
  delta_h_.resize(num_frames);
  g_h_.resize(num_frames);
  H_h_.resize(num_frames);

  // Initialize device mixed variables
  z_.resize(num_frames);
  thrust::fill(z_.begin(), z_.end(), 0.0);
  Z_.resize(num_frames);
  thrust::fill(Z_.begin(), Z_.end(), 0.0);
  delta_.resize(num_frames);
  la_.resize(num_frames);
  g_.resize(num_frames);
  H_.resize(num_frames);
  Gx_.resize(num_frames*12);
  Aloc_.resize(num_frames*12*12);

  // Remaking assemblers since collision frames change.
  // std::cout << "Rebuilding assemblers..." << std::endl;
  // std::cout << "Number of collision frames: " << num_frames << std::endl;
  // std::cout << "T_h: \n" << T_h_ << std::endl;
  assembler_ = std::make_shared<BlockMatrix<double,DIM,4>>(
      T_h_, mesh_->free_map_);  

  // Copy T_h_ to device
  MatrixXi T_tmp = T_h_.transpose();
  T_.resize(T_tmp.size());
  cudaMemcpy(thrust::raw_pointer_cast(T_.data()), T_tmp.data(),
      T_tmp.size()*sizeof(int), cudaMemcpyHostToDevice);
}


template<int DIM, StorageType STORAGE>
void MixedFrictionGpu<DIM,STORAGE>::reset() {

  rhs_.resize(mesh_->jacobian().rows());
  rhs_h_.resize(mesh_->jacobian().rows());
  rhs_h_.setZero();
  thrust::fill(rhs_.begin(), rhs_.end(), 0.0);

  // Copy free map to device
  free_map_.resize(mesh_->free_map_.size());
  thrust::copy(mesh_->free_map_.begin(), mesh_->free_map_.end(),
               free_map_.begin());
}

template<int DIM, StorageType STORAGE>
MixedFrictionGpu<DIM,STORAGE>::VectorType& MixedFrictionGpu<DIM,STORAGE>::rhs()
{ 
  if constexpr (STORAGE == STORAGE_EIGEN) {
    rhs_h_.resize(rhs_.size());
    cudaMemcpy(rhs_h_.data(), thrust::raw_pointer_cast(rhs_.data()),
        rhs_.size()*sizeof(double), cudaMemcpyDeviceToHost);
    return rhs_h_;
  } else {
    return rhs_;
  }
}

template<int DIM, StorageType STORAGE>
MixedFrictionGpu<DIM,STORAGE>::VectorType&
MixedFrictionGpu<DIM,STORAGE>::delta() {
  if constexpr (STORAGE == STORAGE_EIGEN) {
    delta_h_.resize(delta_.size());
    cudaMemcpy(delta_h_.data(), thrust::raw_pointer_cast(delta_.data()),
        delta_.size()*sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "MFGpu delta: " << delta_h_.norm() << std::endl;
    return delta_h_;
  } else {
    return delta_;
  }
}

template<int DIM, StorageType STORAGE>
MixedFrictionGpu<DIM,STORAGE>::VectorType&
MixedFrictionGpu<DIM,STORAGE>::value() {
  if constexpr (STORAGE == STORAGE_EIGEN) {
    z_h_.resize(z_.size());
    cudaMemcpy(z_h_.data(), thrust::raw_pointer_cast(z_.data()),
        z_.size()*sizeof(double), cudaMemcpyDeviceToHost);
    return z_h_;
  } else {
    return z_;
  }
}

template<int DIM, StorageType STORAGE>
MixedFrictionGpu<DIM,STORAGE>::VectorType&
MixedFrictionGpu<DIM,STORAGE>::lambda() {
  if constexpr (STORAGE == STORAGE_EIGEN) {
    std::cout << "Lambda not implemented for MixedFrictionGpu" << std::endl;
    return dummy_;
  } else {
    return la_;
  }
}

template<int DIM, StorageType STORAGE>
MixedFrictionGpu<DIM,STORAGE>::VectorType
MixedFrictionGpu<DIM,STORAGE>::gradient() { 

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

template class mfem::MixedFrictionGpu<3,STORAGE_THRUST>; // 3D
template class mfem::MixedFrictionGpu<3,STORAGE_EIGEN>; // 3D