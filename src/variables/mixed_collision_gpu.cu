#include "mixed_collision_gpu.h"
#include "mesh/mesh.h"
#include "optimizers/optimizer_data.h"
#include <ipc/barrier/barrier.hpp>
#include <thrust/host_vector.h>

using namespace Eigen;
using namespace mfem;

namespace {
  __host__ __device__
  double barrier_gradient(double d, double dhat)
  {
    if (d <= 0.0 || d >= dhat) {
      return 0.0;
    }
    // b(d) = -(d - d̂)²ln(d / d̂)
    // b'(d) = -2(d - d̂)ln(d / d̂) - (d-d̂)²(1 / d)
    //       = (d - d̂) * (-2ln(d/d̂) - (d - d̂) / d)
    //       = (d̂ - d) * (2ln(d/d̂) - d̂/d + 1)
    return (dhat - d) * (2 * log(d / dhat) - dhat / d + 1);
  }

  __host__ __device__
  double barrier_hessian(double d, double dhat) {
    if (d <= 0.0 || d >= dhat) {
      return 0.0;
    }
    double dhat_d = dhat / d;
    return (dhat_d + 2) * dhat_d - 2 * log(d / dhat) - 3;
  }
}

template<int DIM, StorageType STORAGE> __device__
void MixedCollisionGpu<DIM,STORAGE>::derivative_functor::operator()(int i)
    const {
  double kappa = params[0];
  double dhat_sqr = params[1];

  Map<Matrix<double, Aloc_N(), Aloc_N()>> Aloc_i(Aloc + Aloc_N()*Aloc_N()*i);
  Map<Matrix<double, Aloc_N(), 1>> Gx_i(Gx + Aloc_N()*i);

  double di = d[i];
  double d2 = di*di;
  double grad = barrier_gradient(d2 , dhat_sqr);
  g[i] = kappa * (grad * di * 2);
  H[i] = kappa * (barrier_hessian(d2, dhat_sqr) * d2 * 4 + grad * 2);
  Aloc_i = Gx_i * Gx_i.transpose() * H[i];
}

template<int DIM, StorageType STORAGE> __device__
void MixedCollisionGpu<DIM,STORAGE>::rhs_functor::operator()(int i)
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
void MixedCollisionGpu<DIM,STORAGE>::solve_functor::operator()(int i)
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
MixedCollisionGpu<DIM,STORAGE>::MixedCollisionGpu(
    std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
    : MixedVariable<DIM,STORAGE>(mesh), config_(config) {
  // nelem_ = mesh_->T_.rows();

}

template<int DIM, StorageType STORAGE>
double MixedCollisionGpu<DIM,STORAGE>::energy(VectorType& x,
    VectorType& d) {

  ipc::Candidates& candidates = mesh_->collision_candidates();

  if (candidates.size() == 0) {
    return 0.0;
  }
  OptimizerData::get().timer.start("energy", "MixedCollisionGpu");

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
  MatrixXd V_srf = ipc_mesh.vertices(V); // surface vertex set

  // Computing collision constraints
  ipc::MixedConstraints constraints = constraints_;
  constraints.update_distances(d_h);
      OptimizerData::get().timer.start("energyconstruct_constraint_set", "MixedCollisionGpu");

  ipc::construct_constraint_set(candidates, ipc_mesh, V_srf, config_->dhat,
      constraints);
          OptimizerData::get().timer.stop("energyconstruct_constraint_set", "MixedCollisionGpu");

  // std::cout << "constraints: " << constraints.size() << std::endl;
  if (constraints.size() == 0) {
    return 0.0;
  }
  VectorXd e_vec(constraints.size());
  #pragma omp parallel for
  for (size_t i = 0; i < constraints.size(); ++i) {
    double di = constraints.distance(i);
    double la = constraints.lambda(i);
    if (di <= 0.0) {
      e_vec(i) = std::numeric_limits<double>::infinity();
    } else {
      // V1: SQRT
      e_vec(i) = (config_->kappa 
          * ipc::barrier(di*di, config_->dhat*config_->dhat));
    }
    double D = constraints[i].compute_distance(V_srf, E, F,
      ipc::DistanceMode::SQRT);
    if (D <= config_->dhat || di <= config_->dhat) {
      e_vec(i) += la * (D - di);
    }
  }
  double e = e_vec.sum() / dt_ / dt_;
  OptimizerData::get().timer.stop("energy", "MixedCollisionGpu");
  return e;
}

template<int DIM, StorageType STORAGE>
void MixedCollisionGpu<DIM,STORAGE>::update(VectorType& x, double dt) {
  // Get collision frames
  dt_ = dt;

  ipc::Candidates& candidates = mesh_->collision_candidates();
  if (candidates.size() == 0) {
    return;
  }

  Eigen::VectorXd x_h;
  OptimizerData::get().timer.start("update-constraints", "MixedCollisionGpu");

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

  // Convert to reduced (surface only) vertex set
  MatrixXd V_srf = ipc_mesh.vertices(V);

  // Copy device d_ and la_ to host
  if constexpr(STORAGE == STORAGE_THRUST) {
    d_h_.resize(d_.size());
    cudaMemcpy(d_h_.data(), thrust::raw_pointer_cast(d_.data()),
        d_.size()*sizeof(double), cudaMemcpyDeviceToHost);
    la_h_.resize(la_.size());
    cudaMemcpy(la_h_.data(), thrust::raw_pointer_cast(la_.data()),
        la_.size()*sizeof(double), cudaMemcpyDeviceToHost);
  }
   
  // Update mixed variables for current constraint set
  constraints_.update_distances(d_h_);
  constraints_.update_lambdas(la_h_);

  // Computing new collision constraints
  ipc::construct_constraint_set(candidates, ipc_mesh, V_srf,
      config_->dhat, constraints_);

  // Initializing new variables
  int num_frames = constraints_.size();
  T_h_.resize(num_frames, 4);
  D_h_.resize(num_frames);
  Gx_h_.resize(num_frames*12);
  Gx_h_.setZero();
  g_.resize(num_frames);
  H_.resize(num_frames);

  std::cout << "Num frames: " << num_frames << std::endl;

  // From new constraint set, get mixed variables
  d_h_ = constraints_.get_distances();
  la_h_ = constraints_.get_lambdas();
  std::cout << "LA norm" << la_h_.norm() << std::endl;
  // Rebuilding mixed variables for the new set of collision frames.
  #pragma omp parallel for
  for (int i = 0; i < num_frames; ++i) {
    // Getting collision frame, and computing squared distance.
    std::array<long, 4> ids = constraints_[i].vertex_indices(E, F);

    // Add entry in element matrix. Used for assembly.
    for (int j = 0; j < 4; ++j) {
      T_h_(i,j) = -1;
      if (ids[j] != -1) {
        T_h_(i,j) = ipc_mesh.to_full_vertex_id(ids[j]);
      }
    }
    // TODO should be doing dtype based on config
    D_h_(i) = constraints_[i].compute_distance(V_srf, E, F,
        ipc::DistanceMode::SQRT);

    ipc::VectorMax12d grad = constraints_[i].compute_distance_gradient(
        V_srf, E, F, ipc::DistanceMode::SQRT);
    Gx_h_.segment(12*i, grad.size()) = grad;
  }

  // Copy d_h_ and la_h_ to device
  d_.resize(d_h_.size());
  cudaMemcpy(thrust::raw_pointer_cast(d_.data()), d_h_.data(),
      d_h_.size()*sizeof(double), cudaMemcpyHostToDevice);
  
  la_.resize(la_h_.size());
  cudaMemcpy(thrust::raw_pointer_cast(la_.data()), la_h_.data(),
      la_h_.size()*sizeof(double), cudaMemcpyHostToDevice);

  // Copy T_h_ to device
  MatrixXi T_tmp = T_h_.transpose();
  T_.resize(T_tmp.size());
  cudaMemcpy(thrust::raw_pointer_cast(T_.data()), T_tmp.data(),
      T_tmp.size()*sizeof(int), cudaMemcpyHostToDevice);

  // Copy D_h_ to device
  D_.resize(D_h_.size());
  cudaMemcpy(thrust::raw_pointer_cast(D_.data()), D_h_.data(),
      D_h_.size()*sizeof(double), cudaMemcpyHostToDevice);

  // Copy Gx_h_ to device
  Gx_.resize(Gx_h_.size());
  cudaMemcpy(thrust::raw_pointer_cast(Gx_.data()), Gx_h_.data(),
      Gx_h_.size()*sizeof(double), cudaMemcpyHostToDevice);

  Aloc_.resize(num_frames*12*12);
  delta_.resize(num_frames);

  // std::cout << "MC: " << num_frames << std::endl;
  // std::cout << "D: " << D_h_.transpose() << std::endl;
  // std::cout << "d: " << d_h_.transpose() << std::endl;
  // std::cout << "la: " << la_h_.transpose() << std::endl;
  OptimizerData::get().timer.stop("update-constraints", "MixedCollisionGpu");

  // Create new assembler, since collision frame set changes.
  OptimizerData::get().timer.start("assembler-init", "MixedCollisionGpu");
  assembler_ = std::make_shared<BlockMatrix<double,DIM,4>>(T_h_,
      mesh_->free_map_);

  OptimizerData::get().timer.stop("assembler-init", "MixedCollisionGpu");

  thrust::host_vector<double> params_h(2);
  params_h[0] = config_->kappa;
  params_h[1] = config_->dhat * config_->dhat;
  thrust::device_vector<double> params(params_h);

  // std::cout << "d_ size " << d_.size() << std::endl;
  // std::cout << "la_ size " << la_.size() << std::endl;
  // std::cout << "Gx_ size " << Gx_.size() << std::endl;
  // std::cout << "Aloc_ size " << Aloc_.size() << std::endl;
  // std::cout << "params size " << params.size() << std::endl;
  // std::cout << "g_ size " << g_.size() << std::endl;
  // std::cout << "H_ size " << H_.size() << std::endl;

  // Compute derivatives and right-hand-side
  OptimizerData::get().timer.start("derivatives", "MixedCollisionGpu");
  thrust::for_each(thrust::counting_iterator(0),
      thrust::counting_iterator(num_frames),
      derivative_functor(
        thrust::raw_pointer_cast(d_.data()),
        thrust::raw_pointer_cast(g_.data()),
        thrust::raw_pointer_cast(H_.data()),
        thrust::raw_pointer_cast(Gx_.data()),
        thrust::raw_pointer_cast(Aloc_.data()),
        thrust::raw_pointer_cast(params.data())
      ));

  OptimizerData::get().timer.stop("derivatives", "MixedCollisionGpu");

  OptimizerData::get().timer.start("rhs", "MixedCollisionGpu");
  thrust::fill(rhs_.begin(), rhs_.end(), 0.0);
  thrust::for_each(thrust::counting_iterator(0),
      thrust::counting_iterator(num_frames),
      rhs_functor(
        thrust::raw_pointer_cast(T_.data()),
        thrust::raw_pointer_cast(free_map_.data()),
        thrust::raw_pointer_cast(d_.data()),
        thrust::raw_pointer_cast(D_.data()),
        thrust::raw_pointer_cast(g_.data()),
        thrust::raw_pointer_cast(H_.data()),
        thrust::raw_pointer_cast(Gx_.data()),
        thrust::raw_pointer_cast(rhs_.data())
      ));
  OptimizerData::get().timer.stop("rhs", "MixedCollisionGpu");

  OptimizerData::get().timer.start("assemble", "MixedStretchGpu");
  assembler_->update_matrix(Aloc_);
  OptimizerData::get().timer.stop("assemble", "MixedStretchGpu");
}


template<int DIM, StorageType STORAGE>
void MixedCollisionGpu<DIM,STORAGE>::solve(VectorType& dx) {
  if (constraints_.empty()) {
    return;
  }
  OptimizerData::get().timer.start("solve", "MixedCollisionGpu");

  // Copy dx to device
  double* d_dx;
  if constexpr (STORAGE == STORAGE_EIGEN) {
    cudaMalloc(&d_dx, dx.size()*sizeof(double));
    cudaMemcpy(d_dx, dx.data(), dx.size()*sizeof(double), cudaMemcpyHostToDevice);
  } else {
    d_dx = thrust::raw_pointer_cast(dx.data());
  }
  int num_frames = constraints_.size();
  thrust::for_each(thrust::counting_iterator(0),
      thrust::counting_iterator(num_frames),
      solve_functor(
        thrust::raw_pointer_cast(T_.data()),
        thrust::raw_pointer_cast(free_map_.data()),
        thrust::raw_pointer_cast(d_.data()),
        thrust::raw_pointer_cast(D_.data()),
        thrust::raw_pointer_cast(g_.data()),
        thrust::raw_pointer_cast(H_.data()),
        thrust::raw_pointer_cast(Gx_.data()),
        thrust::raw_pointer_cast(la_.data()),
        thrust::raw_pointer_cast(delta_.data()), d_dx
      ));
  OptimizerData::get().timer.stop("solve", "MixedCollisionGpu");
}

template<int DIM, StorageType STORAGE>
void MixedCollisionGpu<DIM,STORAGE>::reset() {
  constraints_.clear();
  d_.resize(0);
  d_h_.resize(0);
  D_.resize(0);
  D_h_.resize(0);
  delta_.resize(0);
  delta_h_.resize(0);
  la_.resize(0);
  la_h_.resize(0);

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
MixedCollisionGpu<DIM,STORAGE>::VectorType& MixedCollisionGpu<DIM,STORAGE>::rhs()
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
MixedCollisionGpu<DIM,STORAGE>::VectorType&
MixedCollisionGpu<DIM,STORAGE>::delta() {
  if constexpr (STORAGE == STORAGE_EIGEN) {
    delta_h_.resize(delta_.size());
    cudaMemcpy(delta_h_.data(), thrust::raw_pointer_cast(delta_.data()),
        delta_.size()*sizeof(double), cudaMemcpyDeviceToHost);
    return delta_h_;
  } else {
    return delta_;
  }
}

template<int DIM, StorageType STORAGE>
MixedCollisionGpu<DIM,STORAGE>::VectorType&
MixedCollisionGpu<DIM,STORAGE>::value() {
  if constexpr (STORAGE == STORAGE_EIGEN) {
    d_h_.resize(d_.size());
    cudaMemcpy(d_h_.data(), thrust::raw_pointer_cast(d_.data()),
        d_.size()*sizeof(double), cudaMemcpyDeviceToHost);
    return d_h_;
  } else {
    return d_;
  }
}

template<int DIM, StorageType STORAGE>
MixedCollisionGpu<DIM,STORAGE>::VectorType&
MixedCollisionGpu<DIM,STORAGE>::lambda() {
  if constexpr (STORAGE == STORAGE_EIGEN) {
    std::cout << "Lambda not implemented for MixedCollisionGpu" << std::endl;
    return dummy_;
  } else {
    return la_;
  }
}

template<int DIM, StorageType STORAGE>
MixedCollisionGpu<DIM,STORAGE>::VectorType
MixedCollisionGpu<DIM,STORAGE>::gradient() { 

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

template class mfem::MixedCollisionGpu<3,STORAGE_THRUST>; // 3D
template class mfem::MixedCollisionGpu<3,STORAGE_EIGEN>; // 3D