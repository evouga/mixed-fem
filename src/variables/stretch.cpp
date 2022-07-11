#include "stretch.h"
#include "mesh/mesh.h"
#include "energies/material_model.h"
#include "svd/newton_procrustes.h"

using namespace Eigen;
using namespace mfem;

template class mfem::Stretch<3>; // 3D
//template class mfem::Stretch<2>; // 2D

template<int DIM>
double Stretch<DIM>::energy(const VectorXd& s) {

  double e = 0;

  #pragma omp parallel for reduction( + : e )
  for (int i = 0; i < nelem_; ++i) {
    const VecN& si = s.segment<N()>(N()*i);
    double e_psi = mesh_->material_->energy(si) * mesh_->volumes()[i];
    e += e_psi;
  }
  return e;
}

template<int DIM>
double Stretch<DIM>::constraint_value(const VectorXd& x,
    const VectorXd& s) {

  VectorXd def_grad;
  mesh_->deformation_gradient(x, def_grad);

  double e = 0;

  #pragma omp parallel for reduction( + : e )
  for (int i = 0; i < nelem_; ++i) {

    const VecM& F = def_grad.segment<M()>(M()*i);
  
    MatD R = R_[i];
    newton_procrustes(R, MatD::Identity(), Map<MatD>(
        Map<MatD>(def_grad.segment<M()>(M()*i).data())));
    MatD S = R.transpose()*Map<MatD>(def_grad.segment<M()>(M()*i).data());
  
    Vector6d stmp;
    stmp << S(0,0), S(1,1), S(2,2),
            0.5*(S(1,0) + S(0,1)),
            0.5*(S(2,0) + S(0,2)),
            0.5*(S(2,1) + S(1,2));

    const VecN& si = s.segment<N()>(N()*i);
    VecN diff = Sym() * (stmp - si);
    double e_l = la_.segment<N()>(N()*i).dot(diff) * mesh_->volumes()[i];
    e += e_l;
  }
  return e;
}

template<int DIM>
void Stretch<DIM>::update(const Eigen::VectorXd& x, double dt) {
  update_rotations(x);
  update_derivatives(dt);
}

template<int DIM>
void Stretch<DIM>::update_rotations(const Eigen::VectorXd& x) {
  VectorXd def_grad;
  mesh_->deformation_gradient(x, def_grad);

  #pragma omp parallel for 
  for (int i = 0; i < nelem_; ++i) {

    // Orthogonality sanity check
    // TODO switch to assert :)
    if((R_[i].transpose()*R_[i] - MatD::Identity()).norm() > 1e-6) {
      std::cerr << "Stretch<DIM> rotation failure!" << std::endl;
      exit(1);
    }

    // TODO wrap newton procrustes in some sort of thing
    // Newton's method orthogonal procrustes to compute rotations and
    // rotation derivatives
    Matrix9d dRdF;
    newton_procrustes(R_[i], MatD::Identity(), 
        sim::unflatten<3,3>(def_grad.segment(9*i,9)), true, dRdF, 1e-6, 100);
 
    MatD Sf = R_[i].transpose()
        * sim::unflatten<3,3>(def_grad.segment(9*i,9));

    Sf = 0.5*(Sf+Sf.transpose());
    S_[i] << Sf(0,0), Sf(1,1), Sf(2,2), Sf(1,0), Sf(2,0), Sf(2,1);
    
    Matrix9d J = sim::flatten_multiply<MatD>(R_[i].transpose())
      * (Matrix9d::Identity() - sim::flatten_multiply_right<MatD>(Sf)*dRdF);

    Matrix<double, 6, 9> Js;
    Js.row(0) = J.row(0);
    Js.row(1) = J.row(4);
    Js.row(2) = J.row(8);
    Js.row(3) = 0.5*(J.row(1) + J.row(3));
    Js.row(4) = 0.5*(J.row(2) + J.row(6));
    Js.row(5) = 0.5*(J.row(5) + J.row(7));
    dSdF_[i] = Js.transpose()*Sym();
  }
}

template<int DIM>
void Stretch<DIM>::update_derivatives(double dt) {

  double h2 = dt * dt;

  data_.timer.start("Hinv");
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    double vol = mesh_->volumes()[i];
    const VecN& si = s_.segment<N()>(N()*i);
    MatN H = h2 * mesh_->material_->hessian(si);
    Hinv_[i] = H.inverse();
    g_[i] = h2 * mesh_->material_->gradient(si);
    H_[i] = (1.0 / vol) * (Syminv() * H * Syminv());
  }
  data_.timer.stop("Hinv");
  
  data_.timer.start("Local H");
  const std::vector<MatrixXd>& Jloc = mesh_->local_jacobians();
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    double vol = mesh_->volumes()[i];
    Aloc_[i] = (Jloc[i].transpose() * (dSdF_[i] * H_[i]
        * dSdF_[i].transpose()) * Jloc[i]) * (vol*vol);
  }
  data_.timer.stop("Local H");
  //saveMarket(assembler_->A, "lhs2.mkt");
  data_.timer.start("Update LHS");
  assembler_->update_matrix(Aloc_);
  data_.timer.stop("Update LHS");
  A_ = assembler_->A;

  // Gradient with respect to x variable
  grad_x_.resize(mesh_->jacobian().rows());

  VectorXd tmp(M()*nelem_);

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    tmp.segment<M()>(M()*i) = dSdF_[i]*la_.segment<N()>(N()*i);
  }
  grad_x_ = -mesh_->jacobian() * tmp;

  // Gradient with respect to mixed variable
  grad_.resize(N()*nelem_);

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    double vol = mesh_->volumes()[i];
    grad_.segment<N()>(N()*i) = vol * (g_[i] + Sym()*la_.segment<N()>(N()*i));
  }
}

template<int DIM>
VectorXd Stretch<DIM>::rhs() {
  data_.timer.start("RHS - s");

  rhs_.resize(mesh_->jacobian().rows());
  rhs_.setZero();
  gl_.resize(N()*nelem_);

  VectorXd tmp(M()*nelem_);
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    double vol = mesh_->volumes()[i];
    const VecN& si = s_.segment<N()>(N()*i);
    gl_.segment<N()>(N()*i) = vol * H_[i] * Sym() * (S_[i] - si)
        + Syminv() * g_[i];
    tmp.segment<M()>(M()*i) = dSdF_[i] * gl_.segment<N()>(N()*i);
  }
  rhs_ = -mesh_->jacobian() * tmp;
  data_.timer.stop("RHS - s");
  return rhs_;
}

template<int DIM>
VectorXd Stretch<DIM>::gradient() {
  return grad_x_;
}

template<int DIM>
VectorXd Stretch<DIM>::gradient_mixed() {
  return grad_;
}

template<int DIM>
void Stretch<DIM>::solve(const VectorXd& dx) {
  data_.timer.start("local");
  Jdx_ = -mesh_->jacobian().transpose() * dx;
  la_ = -gl_;

  ds_.resize(N()*nelem_);

  #pragma omp parallel for 
  for (int i = 0; i < nelem_; ++i) {
    la_.segment<N()>(N()*i) += H_[i] * (dSdF_[i].transpose()
        * Jdx_.segment<M()>(M()*i));
    ds_.segment<N()>(N()*i) = -Hinv_[i]
        * (Sym() * la_.segment<N()>(N()*i) + g_[i]);
  }
  data_.timer.stop("local");
}

template<int DIM>
void Stretch<DIM>::reset() {
  nelem_ = mesh_->T_.rows();

  s_.resize(N()*nelem_);
  la_.resize(N()*nelem_);
  la_.setZero();
  R_.resize(nelem_);
  S_.resize(nelem_);
  H_.resize(nelem_);
  g_.resize(nelem_);
  dSdF_.resize(nelem_);
  Hinv_.resize(nelem_);
  Aloc_.resize(nelem_);
  assembler_ = std::make_shared<Assembler<double,DIM,-1>>(
      mesh_->T_, mesh_->free_map_);

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    R_[i].setIdentity();
    H_[i].setIdentity();
    Hinv_[i].setIdentity();
    g_[i].setZero();
    S_[i] = Ivec();
    s_.segment<N()>(N()*i) = Ivec();
  }
}

template<int DIM>
void Stretch<DIM>::post_solve() {
  la_.setZero();
}
