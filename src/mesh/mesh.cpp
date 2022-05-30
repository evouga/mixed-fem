#include "mesh.h"
#include "boundary_conditions.h"

using namespace mfem;
using namespace Eigen;


Mesh::Mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    std::shared_ptr<MaterialModel> material,
    std::shared_ptr<MaterialConfig> material_config)
    : V_(V), V0_(V), T_(T), material_(material),
      config_(material_config) {
  is_fixed_.resize(V_.rows());
  is_fixed_.setZero();

  bbox.block(0,0,1,3) = V0_.row(0);
  bbox.block(1,0,1,3) = V0_.row(0);
  for(int i = 1; i < V0_.rows(); i++) {
    const Eigen::RowVector3d& v = V0_.row(i);
    for(int d = 0; d < 3; d++) {
      if(v[d] < bbox(0, d)) {
          bbox(0, d) = v[d];
      }
      if(v[d] > bbox(1, d)) {
          bbox(1, d) = v[d];
      }
    }
  }
  BoundaryConditions<3>::init_boundary_groups(V0_, bc_groups_, 0.1);
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
      free_map_(i) = curr++;
    } else {
      free_map_(i) = -1;
    }
  }
}