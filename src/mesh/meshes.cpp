#include "meshes.h"
#include "config.h"
#include "utils/sparse_utils.h"
#include "utils/pinning_matrix.h"
#include "igl/boundary_facets.h"

using namespace Eigen;
using namespace mfem;


Meshes::Meshes(const std::vector<std::shared_ptr<Mesh>>& meshes)
  : meshes_(meshes) {

  size_t num_V = 0;
  size_t num_T = 0;

  std::cout << "Meshes doesn't support mesh with different elements!" << std::endl;

  for (size_t i = 0; i < meshes_.size(); ++i) {
    num_V += meshes_[i]->V_.rows();
    num_T += meshes_[i]->T_.rows();
  }
  // TODO just template it with DIM
  int m = meshes_[0]->V_.cols();
  int n = meshes_[0]->T_.cols();
  V_.resize(num_V, m);
  initial_velocity_.resize(num_V, m);
  Vref_.resize(num_V, m);
  Vinit_.resize(num_V, m);
  T_.resize(num_T, n);
  mat_ids_.resize(num_T);
  mat_ids_.setZero();

  size_t start_V = 0;
  size_t start_T = 0;
  for (size_t i = 0; i < meshes_.size(); ++i) {
    size_t sz_V = meshes_[i]->V_.rows();
    size_t sz_T = meshes_[i]->T_.rows();
    V_.block(start_V,0, sz_V,m) = meshes_[i]->V_;
    initial_velocity_.block(start_V,0, sz_V,m) = meshes_[i]->initial_velocity_;
    Vref_.block(start_V,0, sz_V,m) = meshes_[i]->Vref_;
    Vinit_.block(start_V,0, sz_V,m) = meshes_[i]->Vinit_;
    T_.block(start_T,0, sz_T,n) = meshes_[i]->T_;
    T_.block(start_T,0, sz_T,n).array() += start_V;
    if (meshes_[i]->mat_ids_.size() > 0) {
      mat_ids_.segment(start_T, sz_T) = meshes_[i]->mat_ids_;
    }
    start_V += sz_V;
    start_T += sz_T;

    elements_.insert(elements_.end(), meshes_[i]->elements_.begin(),
        meshes_[i]->elements_.end());
  }
  igl::boundary_facets(T_, F_);
}

void Meshes::init() {
  // Initialize each individual mesh before initialize this object
  for (size_t i = 0; i < meshes_.size(); ++i) {
    meshes_[i]->init();
  }
  Mesh::init();
  // TODO copy starts... don't capture by reference
  std::vector<size_t> starts(meshes_.size() + 1);
  starts[0] = 0;
  for (size_t i = 0; i < meshes_.size(); ++i) {
    starts[i+1] = starts[i] + meshes_[i]->V_.rows();
  }

  // Disable self-collision
  //ipc_mesh_.can_collide = [starts](size_t a, size_t b)->bool {
  //  int ma = 0, mb = 0;
  //  for (size_t i = 0; i < starts.size(); ++i) {
  //    if (a > starts[i] && a < starts[i+1])
  //      ma = i;
  //    if (b > starts[i] && b < starts[i+1])
  //      mb = i;
  //  }
  //  return ma != mb;
  //};
}

void Meshes::volumes(Eigen::VectorXd& vol) {

  size_t total_size = 0;
  for (size_t i = 0; i < meshes_.size(); ++i) {
    assert(meshes_[i]->volumes().size() > 0);
    total_size += meshes_[i]->volumes().size();
  }

  // Concatenate volumes into one large vector
  vol.resize(total_size);
  size_t start = 0;
  for (size_t i = 0; i < meshes_.size(); ++i) {
    size_t sz = meshes_[i]->volumes().size();
    vol.segment(start, sz) = meshes_[i]->volumes();
    start += sz;
  }
}

void Meshes::mass_matrix(SparseMatrixdRowMajor& M,
    const VectorXd& vols) {
  std::vector<Eigen::SparseMatrixdRowMajor> Ms(meshes_.size());
  for (size_t i = 0; i < meshes_.size(); ++i) {
    Ms[i] = meshes_[i]->mass_matrix<MatrixType::FULL>();
  }
  build_block_diagonal(M_, Ms);
  M = M_;
}

void Meshes::init_jacobian() {
  //Jloc_.resize(T_.rows());
  Jloc_.clear();
  for (size_t i = 0; i < meshes_.size(); ++i) {
    Jloc_.insert(Jloc_.end(), meshes_[i]->local_jacobians().begin(),
        meshes_[i]->local_jacobians().end());
  }

  std::vector<Triplet<double>> trips;
// std::cerr << "MESHES init_jacobian wrong" << std::endl;

  int m = V_.cols();
  // std::cout << "M ::: " << m << std::endl;

  // #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) { 

    // Inserting triplets
    for (int j = 0; j < m*m; ++j) {
      // k-th vertex of the tetrahedra
      for (int k = 0; k < m+1; ++k) {
        int vid = T_(i,k); // vertex index

        // x,y,z index for the k-th vertex
        for (int l = 0; l < m; ++l) {
          double val = Jloc_[i](j,m*k+l);
          trips.push_back(Triplet<double>(m*m*i+j, m*vid+l, val));
        }
      }
    }
  }
  J_.resize(m*m*T_.rows(), V_.size());
  J_.setFromTriplets(trips.begin(),trips.end());
}

void Meshes::deformation_gradient(const VectorXd& x, VectorXd& F) {
  assert(x.size() == J_.cols());
  F = J_ * x;
}

void Meshes::init_bcs() {

  external_force_.resize(V_.size());

  size_t start_V = 0;
  int dim = V_.cols();
  for (size_t i = 0; i < meshes_.size(); ++i) {
    size_t sz_V = meshes_[i]->V_.rows();

    // Set boolean values indicating whether vertex is fixed
    is_fixed_.conservativeResize(start_V + sz_V);
    is_fixed_.segment(start_V, sz_V) = meshes_[i]->is_fixed_;

    // Set segment of external force for each mesh
    external_force_.segment(dim*start_V, dim*sz_V) = 
        meshes_[i]->bc_ext_->force();

    //TODO not sure this should be here.
    V_.block(start_V, 0, sz_V, dim) = meshes_[i]->V_;

    start_V += sz_V;
  }

  // Create reduce vertex set selection matrix
  P_ = pinning_matrix(V_, T_, is_fixed_);

  // Map from full vertex set to reduced set indices. Value
  // equals -1 if vertex is fixed.
  free_map_.resize(is_fixed_.size(), -1);
  int curr = 0;
  for (int i = 0; i < is_fixed_.size(); ++i) {
    if (is_fixed_(i) == 0) {
      free_map_[i] = curr++;
    }
  }  
}

void Meshes::update_bcs(double dt) {
  size_t start_V = 0;
  int dim = V_.cols();
  for (size_t i = 0; i < meshes_.size(); ++i) {
    size_t sz_V = meshes_[i]->V_.rows();

    // Ugh!
    MatrixXd tmp = V_.block(start_V, 0, sz_V, dim);
    meshes_[i]->bc_->step(tmp, dt);

    // Update segment of external force for each mesh
    if (!meshes_[i]->bc_ext_->is_constant()) {
      meshes_[i]->bc_ext_->step(tmp, dt);
      external_force_.segment(dim*start_V, dim*sz_V) = 
          meshes_[i]->bc_ext_->force();
    }

    V_.block(start_V, 0, sz_V, dim) = tmp;
    start_V += sz_V;
  }
}
