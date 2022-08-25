#include "mesh.h"
#include "boundary_conditions.h"
#include "energies/material_model.h"
#include "config.h"
#include "energies/stable_neohookean.h"
#include "utils/pinning_matrix.h"
#include "igl/boundary_facets.h"
#include "igl/oriented_facets.h"
#include "igl/edges.h"

using namespace mfem;
using namespace Eigen;

Mesh::Mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    const std::vector<VectorXi>& subsets,
    const std::vector<std::shared_ptr<MaterialModel>>& materials)
    : V_(V), V0_(V), T_(T) {
  assert(materials.size() > 0);
  assert(subsets.size() == materials.size());
  material_ = materials[0];

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

  for (Eigen::Index i = 0; i < T_.rows(); ++i) {
    elements_.push_back(Element(material_));
  }

  for (size_t i = 0; i < subsets.size(); ++i) {
    for (int j = 0; j < subsets[i].size(); ++j) {
      elements_[subsets[i](j)].material_ = materials[i];
    }
  }


}


Mesh::Mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    std::shared_ptr<MaterialModel> material)
    : V_(V), V0_(V), T_(T), material_(material) {

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

  for (int i = 0; i < T_.rows(); ++i) {
    elements_.push_back(Element(material));
  }

  igl::boundary_facets(T_, F_);
  assert(F_.cols() == cols);
}

void Mesh::init() {
  volumes(vols_);

  int M = std::pow(V_.cols(),2);

  // Initialize volume sparse matrix
  W_.resize(T_.rows()*M, T_.rows()*M);
  std::vector<Triplet<double>> trips;
  for (int i = 0; i < T_.rows(); ++i) {
    for (int j = 0; j < M; ++j) {
      trips.push_back(Triplet<double>(M*i+j, M*i+j,vols_[i]));
    }
  }
  W_.setFromTriplets(trips.begin(),trips.end());

  init_jacobian();
  PJW_ = P_ * J_.transpose() * W_;

  mass_matrix(M_, vols_);

  PM_ = P_ * M_;
  PMP_ = PM_ * P_.transpose();
  
  int dim = V_.cols();

  MatrixXi E, F;
  if (dim == 2) {
    igl::oriented_facets(T_, E);
  } else {
    igl::boundary_facets(T_, F);
    // igl::oriented_facets(F, E);
    igl::edges(F, E);
  }
  ipc_mesh_ = ipc::CollisionMesh::build_from_full_mesh(V_, E, F);
  // TODO can_collide

}

void Mesh::clear_fixed_vertices() {
  fixed_vertices_.clear();
  is_fixed_.setZero();
}

void Mesh::free_vertex(int id) {
  fixed_vertices_.erase(fixed_vertices_.begin() + id);
  is_fixed_(id) = 0;
}

void Mesh::set_fixed(int id) {
  is_fixed_(id) = 1;
  fixed_vertices_.push_back(id);
}

void Mesh::set_fixed(const std::vector<int>& ids) {
  for (size_t i = 0; i < ids.size(); ++i) {
    is_fixed_(ids[i]) = 1;
  }
  fixed_vertices_.insert(fixed_vertices_.end(), ids.begin(), ids.end());
}

void Mesh::update_free_map() {
  free_map_.resize(is_fixed_.size());
  int curr = 0;
  for (int i = 0; i < is_fixed_.size(); ++i) {
    if (is_fixed_(i) == 0) {
      free_map_[i] = curr++;
    } else {
      free_map_[i] = -1;
    }
  }
  P_ = pinning_matrix(V_, T_, is_fixed_);

}
