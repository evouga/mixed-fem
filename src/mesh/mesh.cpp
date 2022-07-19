#include "mesh.h"
#include "boundary_conditions.h"
#include "energies/material_model.h"
#include "config.h"
#include "pinning_matrix.h"
#include "energies/stable_neohookean.h"

using namespace mfem;
using namespace Eigen;


Mesh::Mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    std::shared_ptr<MaterialModel> material,
    std::shared_ptr<MaterialConfig> material_config)
    : V_(V), V0_(V), T_(T), material_(material),
      config_(material_config) {

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

  for (size_t i = 0; i < T_.rows(); ++i) {
    elements_.push_back(Element(material,material_config));
    RowVectorXd centroid = (V_.row(T_(i,0)) + V_.row(T_(i,1)) + V_.row(T_(i,2)) + V_.row(T_(i,3))) / 4;
    RowVectorXd offset = bbox.colwise().mean();

    double left_x1 = bbox(0,0);
    double left_x2 = (offset(0) - bbox(0,0))*0.8 + bbox(0,0);

    double right_x1 = (bbox(1,0) - offset(0))*0.2 + offset(0);
    double right_x2 = (offset(0) - bbox(0,0))*0.8 + bbox(0,0);
    Matrix23x<double> range1;
    Matrix23x<double> range2;

    if ( (centroid(0) < left_x2 && centroid(0) > left_x1) || (centroid(0) > right_x1) ) {
      MaterialConfig cfg(*material_config.get());
      double ym = 1e12;
      double pr = 0.45;
      cfg.mu = ym/(2.0*(1.0+pr));
      cfg.la = (ym*pr)/((1.0+pr)*(1.0-2.0*pr));
      elements_[i].config_ = std::make_shared<MaterialConfig>(cfg);
      elements_[i].material_ = std::make_shared<StableNeohookean>(elements_[i].config_);
    }


    //double ym = 1e6;
    //double pr = 0.45;
    //double mu = ym/(2.0*(1.0+pr));
    //double la = (ym*pr)/((1.0+pr)*(1.0-2.0*pr));
  }

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
  std::cout << "Mesh::init() " << M_.rows() << ", " << M_.cols() << std::endl;

  PMP_ = P_ * M_ * P_.transpose();
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
