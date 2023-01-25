#include "displacement_gpu.h"
#include <Eigen/Core>
#include "factories/integrator_factory.h"
#include <thrust/inner_product.h>

using namespace mfem;
using namespace Eigen;
using namespace thrust::placeholders;

template<int DIM>
double* DisplacementGpu<DIM>::to_full(const VectorType& x,
    ProjectionType type) {
  double* x_full;
  PT_gpu_.product(thrust::raw_pointer_cast(x.data()), &x_full);

  // If dirichlet BCs are included, add the 'b' vector to the result
  if (type == ProjectionType::WITH_DIRICHLET) {
    thrust::device_ptr<double> x_full_ptr(x_full);
    thrust::transform(x_full_ptr, x_full_ptr + b_.size(), b_.begin(),
        x_full_ptr, thrust::plus<double>());
  }
  return x_full;
}

template<int DIM>
void DisplacementGpu<DIM>::to_full(const VectorType& x, VectorType& out,
    ProjectionType type) {
  double* x_full = to_full(x, type);

  // Copy cuda x_full to thrust vector
  cudaMemcpy(thrust::raw_pointer_cast(out.data()), x_full,
      mesh_->V_.size() * sizeof(double),
      cudaMemcpyDeviceToDevice);
}

template<int DIM>
DisplacementGpu<DIM>::DisplacementGpu(std::shared_ptr<Mesh> mesh,
    std::shared_ptr<SimConfig> config)
    : Variable<DIM,STORAGE_THRUST>(mesh), config_(config) {
}

template<int DIM>
double DisplacementGpu<DIM>::energy(VectorType& x_full) {
  OptimizerData::get().timer.start("energy", "DisplacementGpu");

  double h2 = integrator_->dt() * integrator_->dt();

  // Project to full space and add Dirichlet BCs
  // NOTE: using data from to_full(), which is safe
  // double* x_full_ptr = to_full(x, ProjectionType::WITH_DIRICHLET);
  // thrust::device_ptr<double> x_full(x_full_ptr);
  // double* x_full_ptr = thrust::raw_pointer_cast(x_full.data());

  // Compute x_full - x_tilde, storing result in x_full
  thrust::transform(x_full.begin(), x_full.end(), x_tilde_.begin(),
      tmp_.begin(), thrust::minus<double>());

  // - h^2 fext_
  thrust::transform(tmp_.begin(), tmp_.end(), f_ext_.begin(),
      tmp_.begin(), _1 - h2 * _2);

  // Compute M * (x_full - x_tilde - h^2 fext_)
  double* prod;
  M_gpu_.product(thrust::raw_pointer_cast(tmp_.data()), &prod);
  thrust::device_ptr<double> prod_ptr(prod);

  // Dot this against tmp_ to get energy
  double energy = 0.5 * thrust::inner_product(tmp_.begin(),
      tmp_.end(), prod_ptr, 0.0);

  OptimizerData::get().timer.stop("energy", "DisplacementGpu");
  return energy;
}

template<int DIM>
void DisplacementGpu<DIM>::post_solve() {
  // TODO b_,/BCs should not be here, boundary conditions
  // probably should be owned by either optimizer or mesh
  #pragma omp parallel for
  for (int i = 0; i < mesh_->V_.rows(); ++i) {
    if (mesh_->is_fixed_(i)) {
      b_h_.segment<DIM>(DIM*i) = mesh_->V_.row(i).transpose();
    }
  }
  cudaMemcpy(thrust::raw_pointer_cast(b_.data()),
      b_h_.data(), b_h_.size()*sizeof(double),
      cudaMemcpyHostToDevice);

  double* x_full_ptr = to_full(x_);
  integrator_->update(x_full_ptr);
  double* const x_tilde = integrator_->x_tilde();
  double* x_tilde_ptr = thrust::raw_pointer_cast(x_tilde_.data());
  cudaMemcpy(x_tilde_ptr, x_tilde, full_size_*sizeof(double),
      cudaMemcpyDeviceToDevice);

  // Copy x to CPU and write to mesh->V_
  MatrixXd V(mesh_->V_.cols(), mesh_->V_.rows());
  cudaMemcpy(V.data(), x_full_ptr, V.size()*sizeof(double),
      cudaMemcpyDeviceToHost);
  mesh_->V_ = V.transpose();
}

template<int DIM>
void DisplacementGpu<DIM>::update(VectorType&, double) {}

template<int DIM>
DisplacementGpu<DIM>::VectorType& DisplacementGpu<DIM>::rhs() {
  OptimizerData::get().timer.start("rhs", "DisplacementGpu");

  double h2 = integrator_->dt() * integrator_->dt();

  // Project to full space and add Dirichlet BCs
  // NOTE: using data from to_full(), which is safe
  double* x_full_ptr = to_full(x_, ProjectionType::WITH_DIRICHLET);
  thrust::device_ptr<double> x_full(x_full_ptr);

  // Compute x_full - x_tilde, storing result in x_full
  thrust::transform(x_full, x_full + x_tilde_.size(), x_tilde_.begin(),
      x_full, thrust::minus<double>());

  // - h^2 fext_
  thrust::transform(x_full, x_full + f_ext_.size(), f_ext_.begin(),
      x_full, _1 - h2 * _2);

  double* prod;
  PM_gpu_.product(x_full_ptr, &prod);

  // Copy result to rhs_ and negate
  thrust::device_ptr<double> prod_ptr(prod);
  thrust::transform(prod_ptr, prod_ptr + rhs_.size(), rhs_.begin(),
      thrust::negate<double>());

  OptimizerData::get().timer.stop("rhs", "DisplacementGpu");
  return rhs_;
}

template<int DIM>
DisplacementGpu<DIM>::VectorType DisplacementGpu<DIM>::gradient() {
  OptimizerData::get().timer.start("gradient", "DisplacementGpu");
  std::cout << "Displacement gradient not implemented yet" << std::endl;
  OptimizerData::get().timer.stop("gradient", "DisplacementGpu");
  return grad_;
}

template<int DIM>
void DisplacementGpu<DIM>::extract_diagonal(double* diag) {
  // Copy diag_gpu to diag
  cudaMemcpy(diag, thrust::raw_pointer_cast(diag_gpu_.data()),
      diag_gpu_.size()*sizeof(double), cudaMemcpyDeviceToDevice);
}
template<int DIM>
void DisplacementGpu<DIM>::reset() {
  // Initial positions from mesh
  MatrixXd tmp = mesh_->V_.transpose();
  VectorXd x = Map<VectorXd>(tmp.data(), mesh_->V_.size());

  thrust::device_vector<double> x_tmp;
  x_tmp.resize(x.size());
  double* x_ptr = thrust::raw_pointer_cast(x_tmp.data());
  cudaMemcpy(x_ptr, x.data(), x.size()*sizeof(double),
      cudaMemcpyHostToDevice);

  // Initial velocities from mesh
  tmp = mesh_->initial_velocity_.transpose();
  VectorXd v0 = Map<VectorXd>(tmp.data(), mesh_->V_.size());

  // Copy host v0 to device d_v0
  thrust::device_vector<double> d_v0(v0.size());
  double* v0_ptr = thrust::raw_pointer_cast(d_v0.data());
  cudaMemcpy(v0_ptr, v0.data(), v0.size()*sizeof(double),
      cudaMemcpyHostToDevice);

  IntegratorFactory<STORAGE_THRUST> factory;
  integrator_ = factory.create(config_->ti_type, x_ptr, v0_ptr,
      x.size(), config_->h);  
  
  const auto& P = mesh_->projection_matrix();
  full_size_ = P.cols();
  reduced_size_ = P.rows();
  // std::cout << "Full size: " << full_size_ << std::endl;
  // std::cout << "Reduced size: " << reduced_size_ << std::endl;
  P_gpu_.init(P);
  PT_gpu_.init(P.transpose());

  b_h_ = x - P.transpose()*P*x;
  b_.resize(b_h_.size());
  double* b_ptr = thrust::raw_pointer_cast(b_.data());
  cudaMemcpy(b_ptr, b_h_.data(), b_h_.size()*sizeof(double),
      cudaMemcpyHostToDevice);
  tmp_.resize(b_.size());

  // Project out Dirichlet boundary conditions
  double* x_reduce;
  P_gpu_.product(x_ptr, &x_reduce);

  // Copy result of product to x_
  x_.resize(P.rows());
  cudaMemcpy(thrust::raw_pointer_cast(x_.data()),
      x_reduce, x_.size()*sizeof(double), cudaMemcpyDeviceToDevice);

  // Initialize dx to 0
  dx_ = VectorType(x_.size(), 0);
  rhs_.resize(x_.size(), 0);
  lhs_ = mesh_->template mass_matrix<MatrixType::PROJECTED>();

  // Block diagonal host
  thrust::host_vector<MatD> diag_h;
  diag_h.resize(lhs_.rows() / DIM);
  for (int i = 0; i < diag_h.size(); ++i) {
    diag_h[i] = lhs_.block(i*DIM, i*DIM, DIM, DIM);
  }
  // Copy diag_h to diag_gpu
  diag_gpu_.resize(diag_h.size()*DIM*DIM);

  // Copy diag_h to diag gpu
  cudaMemcpy(thrust::raw_pointer_cast(diag_gpu_.data()),
      diag_h.data(), diag_h.size()*DIM*DIM*sizeof(double),
      cudaMemcpyHostToDevice);

  // Copy mesh external force to f_ext_
  const VectorXd& f_ext = mesh_->external_force();
  f_ext_.resize(f_ext.size());
  double* f_ext_ptr = thrust::raw_pointer_cast(f_ext_.data());
  cudaMemcpy(f_ext_ptr, f_ext.data(), f_ext.size()*sizeof(double),
      cudaMemcpyHostToDevice);

  // Copy x_tilde to local member
  double* const x_tilde = integrator_->x_tilde();
  x_tilde_.resize(full_size_);
  double* x_tilde_ptr = thrust::raw_pointer_cast(x_tilde_.data());
  cudaMemcpy(x_tilde_ptr, x_tilde, full_size_*sizeof(double),
      cudaMemcpyDeviceToDevice);

  const auto& MM = mesh_->template mass_matrix<MatrixType::FULL>();
  M_gpu_.init(MM);

  const auto& PM = mesh_->template mass_matrix<MatrixType::PROJECT_ROWS>();
  PM_gpu_.init(PM);

  const auto& PMP = mesh_->template mass_matrix<MatrixType::PROJECTED>();
  PMP_gpu_.init(PMP);
}

template<int DIM>
void DisplacementGpu<DIM>::apply(double* x, const double* b) {
  // Multiply rhs (b) against the mass matrix with Dirichlet BCs projected out
  double* out;
  PMP_gpu_.product(b, &out);

  // Copy result of product to x
  cudaMemcpy(x, out, reduced_size_*sizeof(double), cudaMemcpyDeviceToDevice);
}

template class mfem::DisplacementGpu<3>; // 3D
