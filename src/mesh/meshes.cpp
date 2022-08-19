#include "meshes.h"
#include "config.h"
#include "utils/sparse_utils.h"
#include "boundary_conditions.h"
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

  V_.resize(num_V, 2);
  T_.resize(num_T, 3);

  size_t start_V = 0;
  size_t start_T = 0;
  for (size_t i = 0; i < meshes_.size(); ++i) {
    size_t sz_V = meshes_[i]->V_.rows();
    size_t sz_T = meshes_[i]->T_.rows();
    V_.block(start_V,0, sz_V,2) = meshes_[i]->V_;
    T_.block(start_T,0, sz_T,3) = meshes_[i]->T_;
    T_.block(start_T,0, sz_T,3).array() += start_V;
    start_V += sz_V;
    start_T += sz_T;

    elements_.insert(elements_.end(), meshes_[i]->elements_.begin(),
        meshes_[i]->elements_.end());
  }
  V0_ = V_;

  is_fixed_.resize(V_.rows());
  is_fixed_.setZero();
  bbox.setZero();
  int cols = V0_.cols();
  bbox.block(0,0,1,cols) = V0_.row(0);
  bbox.block(1,0,1,cols) = V0_.row(0);
  for(int i = 1; i < V0_.rows(); i++) {
    const Eigen::RowVectorXd& v = V0_.row(i);
    for(int d = 0; d < cols; d++) {
      if(v[d] < bbox(0, d)) {
          bbox(0, d) = v[d];
      }
      if(v[d] > bbox(1, d)) {
          bbox(1, d) = v[d];
      }
    }
  }
  BoundaryConditions<3>::init_boundary_groups(V0_, bc_groups_, 0.01);
  P_ = pinning_matrix(V_, T_, is_fixed_);

  igl::boundary_facets(T_, F_);
  assert(F_.cols() == cols);
}
void Meshes::init() {
  std::cout << "Meshes::init()" << std::endl;

  // Initialize each individual mesh before initialize this object
  for (size_t i = 0; i < meshes_.size(); ++i) {
    meshes_[i]->init();
  }
  Mesh::init();
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
  std::cout << "NOTE! mass_matrix() may be projected!" << std::endl;
  std::vector<Eigen::SparseMatrixdRowMajor> Ms(meshes_.size());
  for (size_t i = 0; i < meshes_.size(); ++i) {
    Ms[i] = meshes_[i]->mass_matrix();
  }
  std::cout << "meshes size: " << meshes_.size() << std::endl;
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
std::cerr << "MESHES init_jacobian wrong" << std::endl;
  // #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) { 

    // Inserting triplets
    for (int j = 0; j < 4; ++j) {
      // k-th vertex of the tetrahedra
      for (int k = 0; k < 3; ++k) {
        int vid = T_(i,k); // vertex index

        // x,y,z index for the k-th vertex
        for (int l = 0; l < 2; ++l) {
          double val = Jloc_[i](j,2*k+l);
          trips.push_back(Triplet<double>(4*i+j, 2*vid+l, val));
        }
      }
    }
  }
  J_.resize(4*T_.rows(), V_.size());
  J_.setFromTriplets(trips.begin(),trips.end());
}

void Meshes::deformation_gradient(const VectorXd& x, VectorXd& F) {
  assert(x.size() == J_.cols());
  F = J_ * x;
}

void Meshes::jacobian(SparseMatrixdRowMajor& J, const VectorXd& vols,
      bool weighted) {
  std::cerr << "Tri2DMesh::jacobian(J, vols, weighted) unimplemented" << std::endl;
}

void Meshes::jacobian(std::vector<MatrixXd>& J) {
  std::cerr << "Tri2DMesh::jacobian(J) unimplemented" << std::endl;
}
