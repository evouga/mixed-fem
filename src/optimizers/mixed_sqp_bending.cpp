#include "mixed_sqp_bending.h"

#include <chrono>
#include "sparse_utils.h"
#include "svd/dsvd.h"
#include "svd/svd3x3_sse.h"
#include "svd/svd_eigen.h"
#include "linesearch.h"
#include "pcg.h"
#include "svd/newton_procrustes.h"
#include "energies/material_model.h"
#include "mesh/mesh.h"

#include <iomanip>
#include <fstream>
#include "unsupported/Eigen/SparseExtra"
#include "igl/edge_topology.h"
#include "igl/edge_lengths.h"
#include "igl/slice_mask.h"
#include <rigid_inertia_com.h>

using namespace mfem;
using namespace Eigen;

void MixedSQPBending::step() {
  data_.clear();

  E_prev_ = 0;

  int i = 0;
  double grad_norm;

  do {
    data_.timer.start("step");
    update_system();
    substep(i, grad_norm);
    linesearch(x_, dx_, s_, ds_);

    // x_ += dx_;
    // s_ += ds_;
     std::cout << "ds norm: " << ds_.norm() << " la norm: " << la_.norm() << std::endl;

    double E = energy(x_, s_, a_, la_, ga_);
    double res = std::abs((E - E_prev_) / E);
    data_.egrad_.push_back(rhs_.norm());
    data_.energies_.push_back(E);
    data_.energy_residuals_.push_back(res);
    E_prev_ = E;
    data_.timer.stop("step");


    ++i;
  } while (i < config_->outer_steps && grad_norm > config_->newton_tol);

  if (config_->show_data) {
    data_.print_data(config_->show_timing);
  }
  update_configuration();
}

void MixedSQPBending::build_lhs() {
  data_.timer.start("LHS");

  double ih2 = 1. / (wdt_*wdt_*config_->h * config_->h);
  double h2 = (wdt_*wdt_*config_->h * config_->h);

  Ha_.resize(nedges_);
  Ha_inv_.resize(nedges_);
  grad_a_.resize(nedges_);
  // std::cout << "nedges_: " << nedges_ << std::endl;
  // std::cout << "a_ : "<< a_.size() << " a0_: " << a0_.size() << " l size: " << l_.size() << std::endl;
  std::vector<Eigen::Triplet<double>> trips(nedges_);
  #pragma omp parallel for
  for (int i = 0; i < nedges_; ++i) {
    Ha_inv_[i] = 1 / h2 / config_->kappa;
    grad_a_[i] = h2 * config_->kappa * (a_(i) - a0_(i));
    Ha_[i] = (1.0 / l_(i)) * h2 * config_->kappa;
    trips[i] = Triplet<double>(i,i,Ha_[i]);
  }
  SparseMatrixd Ha(PDW_.cols(), PDW_.cols());
  Ha.setFromTriplets(trips.begin(), trips.end());
  SparseMatrixdRowMajor Ha2 = PDW_ * Ha * PDW_.transpose();
  // std::cout << "PDW: " << PDW_.rows() << " PDW_.cols: " << PDW_.cols() << std::endl
  // std::cout << "Ha2 .rows() : " << Ha2.rows() << M_.rows() << std::endl;

  data_.timer.start("Hinv");
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector3d& si = s_.segment<3>(3*i,3);
    Matrix3d H = h2 * mesh_->material_->hessian(si);
    Hinv_[i] = H.inverse();
    g_[i] = h2 * mesh_->material_->gradient(si);
    H_[i] = (1.0 / vols_[i]) * (Sym3inv * H * Sym3inv);
    
    // Vector6d si2;
    // si2 << si(0), 1, si(1), 0, si(2), 0;
    // std::cout << "g1: \n" << object_->material_->gradient(si) <<
      // " \n g2: \n" << object_->material_->gradient(si2) << "\n"<< std::endl;
  }
  data_.timer.stop("Hinv");
  
  data_.timer.start("Local H");

  const std::vector<MatrixXd>& Jloc = mesh_->local_jacobians();
  std::vector<MatrixXd> Hloc(nelem_); 
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    Hloc[i] = (Jloc[i].transpose() * (dS_[i] * H_[i]
        * dS_[i].transpose()) * Jloc[i]) * (vols_[i] * vols_[i]);
  }
  data_.timer.stop("Local H");
  data_.timer.start("Update LHS");
  assembler_->update_matrix(Hloc);
  data_.timer.stop("Update LHS");

  lhs_ = M_ + assembler_->A + Ha2;
  data_.timer.stop("LHS");
  //saveMarket(lhs_, "lhs_full.mkt");
}

void MixedSQPBending::build_rhs() {
  data_.timer.start("RHS");

  rhs_.resize(x_.size());
  rhs_.setZero();
  gl_.resize(3*nelem_);
  gg_.resize(nedges_);
  
  double h = wdt_*config_->h;
  double h2 = h*h;

  VectorXd xt = P_.transpose()*x_ + b_;

  VectorXd ax;
  normals(xt, n_);
  angles(xt, n_, ax);
  #pragma omp parallel for
  for (int i = 0; i < nedges_; ++i) {
    gg_(i) = l_(i) * Ha_[i] * (ax(i) - a_(i)) + grad_a_[i];
  }

  VectorXd tmp(9*nelem_);
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const Vector3d& si = s_.segment<3>(3*i);
    gl_.segment<3>(3*i) = vols_[i] * H_[i] * Sym3 * (S_[i] - si) + Sym3inv*g_[i];
    tmp.segment<9>(9*i) = dS_[i]*gl_.segment<3>(3*i);
  }

  rhs_ = -mesh_->jacobian() * tmp - PDW_ * gg_ - PM_*(wx_*xt + wx0_*x0_
      + wx1_*x1_ + wx2_*x2_ - h2*f_ext_);
  data_.timer.stop("RHS");

  grad_.resize(x_.size() + 3*nelem_);
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    tmp.segment<9>(9*i) = dS_[i]*la_.segment<3>(3*i);
    grad_.segment<3>(x_.size() + 3*i) = vols_[i] * (g_[i] + Sym3*la_.segment<3>(3*i));
  }
  grad_.segment(0,x_.size()) = PM_*(wx_*xt + wx0_*x0_ + wx1_*x1_ + wx2_*x2_
      - h2*f_ext_) - mesh_->jacobian()  * tmp;
}

void MixedSQPBending::substep(int step, double& decrement) {
  int niter = 0;

  data_.timer.start("global");
  solver_.compute(lhs_);
  if(solver_.info()!=Success) {
   std::cerr << "prefactor failed! " << std::endl;
   exit(1);
  }
  dx_ = solver_.solve(rhs_);
  data_.timer.stop("global");


  data_.timer.start("local");
  Jdx_ = -mesh_->jacobian().transpose() * dx_;
  la_ = -gl_;

  double h2 = config_->h*config_->h;
  ArrayXd Ha = ((config_->kappa * h2)/l_.array());
  ArrayXd ga = (h2*config_->kappa)*(a_-a0_).array();

  ga_ = gg_.array() - (PDW_.transpose() * dx_).array() * Ha;
  
  // Update per-element R & S matrices
  ds_.resize(3*nelem_);

  da_.resize(nedges_);
  da_ = -(l_.array()*(ga_.array() + ga)) / Ha;
  std::cout << "da: " << da_.norm() << std::endl;
  a_ += da_;
  #pragma omp parallel for 
  for (int i = 0; i < nelem_; ++i) {
    la_.segment<3>(3*i) += H_[i] * (dS_[i].transpose() * Jdx_.segment<9>(9*i));
    ds_.segment<3>(3*i) = -Hinv_[i] * (Sym3 * la_.segment<3>(3*i)+ g_[i]);
  }
  data_.timer.stop("local");

  decrement = std::max(dx_.lpNorm<Infinity>(), ds_.lpNorm<Infinity>());
}

void MixedSQPBending::update_system() {

  if (!mesh_->fixed_jacobian()) {
    VectorXd x = P_.transpose()*x_ + b_;
    mesh_->update_jacobian(x);
  }

  VectorXd x = P_.transpose()*x_ + b_;

  normals(x, n_);
  std::vector<Eigen::VectorXd> tmp;
  grad_angles(x, n_, tmp); 

  // Compute rotations and rotation derivatives
  update_rotations();

  // Assemble blocks for left and right hand side
  build_lhs();
  build_rhs();
}


void MixedSQPBending::update_rotations() {
  data_.timer.start("Rot Update");
  dS_.resize(nelem_);

  VectorXd def_grad;
  mesh_->deformation_gradient(P_.transpose()*x_+b_, def_grad);

  #pragma omp parallel for 
  for (int i = 0; i < nelem_; ++i) {

    Matrix<double, 9, 9> J;
    
    //polar decomp code
    Eigen::Matrix<double, 9,9> dRdF;

    newton_procrustes(R_[i], Eigen::Matrix3d::Identity(), 
        sim::unflatten<3,3>(def_grad.segment(9*i,9)), true, dRdF, 1e-6, 100);
    
    Eigen::Matrix3d Sf = R_[i].transpose()
        * sim::unflatten<3,3>(def_grad.segment(9*i,9));

    Sf = 0.5*(Sf+Sf.transpose());
    S_[i] << Sf(0,0), Sf(2,2), Sf(2,0);

    J = sim::flatten_multiply<Eigen::Matrix3d>(R_[i].transpose())
        * (Matrix9d::Identity() 
           - sim::flatten_multiply_right<Eigen::Matrix3d>(Sf)*dRdF);

    Matrix<double, 3, 9> Js;
    Js.row(0) = J.row(0);
    Js.row(1) = J.row(8);
    Js.row(2) = 0.5*(J.row(2) + J.row(6));
    //Js.row(2) = 0.5*(J.row(5) + J.row(7));
    dS_[i] = Js.transpose()*Sym3;
  }
  data_.timer.stop("Rot Update");
}

double MixedSQPBending::energy(const VectorXd& x, const VectorXd& s,
    const VectorXd& a, const VectorXd& la, const Eigen::VectorXd& ga) {
  double h = wdt_*config_->h;
  double h2 = h*h;
  VectorXd xt = P_.transpose()*x + b_;
  VectorXd xdiff = P_ * (wx_*xt + wx0_*x0_ + wx1_*x1_ + wx2_*x2_ - h*h*f_ext_);
  double Em = 0.5*xdiff.transpose()*M_*xdiff;

  VectorXd def_grad;
  mesh_->deformation_gradient(xt, def_grad);

  VectorXd ax;
  normals(xt, n_);
  angles(xt, n_, ax);
  double el = 0;
  // #pragma omp parallel for reduction( + : el)
  // for (int i = 0; i < nedges_; ++i) {
  //   el += l_(i) * (h2 * 0.5 * config_->kappa * std::pow(a(i) - a0_(i), 2)
  //       - ga(i)*(ax(i) - a(i)));
  // }

  double e = 0;
  #pragma omp parallel for reduction( + : e )
  for (int i = 0; i < nelem_; ++i) {
  
    Matrix3d R = R_[i];
    newton_procrustes(R, Matrix3d::Identity(), Map<Matrix3d>(def_grad.segment<9>(9*i).data()));
    Matrix3d S = R.transpose()*Eigen::Map<Matrix3d>(def_grad.segment<9>(9*i).data());

  //std::cout << "Si: \n" << S << std::endl;

    Vector3d stmp; 
    stmp << S(0,0), S(2,2), 0.5*(S(2,0) + S(0,2));
    if ( (S(1,1) - 1.0) > 1e-12) {
      std::cout << "S: " << S << std::endl;
    }
    
    const Vector3d& si = s.segment<3>(3*i);
    Vector3d diff = Sym3 * (stmp - si);
    e += h2 * mesh_->material_->energy(si) * vols_[i]
        - la.segment<3>(3*i).dot(diff) * vols_[i];
  }
  e += (Em + el);
  return e;
}

void MixedSQPBending::normals(const VectorXd& x, MatrixXd& n) {
  const MatrixXi& T = mesh_->T_;
  n.resize(T.rows(), T.cols());

  #pragma omp parallel for
  for(int i = 0; i < T.rows(); ++i) {
    Matrix<double, 9, 3> N;
    Vector3d v1 = x.segment<3>(3*T(i,1)) - x.segment<3>(3*T(i,0));
    Vector3d v2 = x.segment<3>(3*T(i,2)) - x.segment<3>(3*T(i,0));
    Vector3d ni = v1.cross(v2);
    ni.normalize();
    n.row(i) = ni.transpose();
  }
}
void MixedSQPBending::angles(const VectorXd& x, const MatrixXd& n,
    VectorXd& a) {
  // std::cout << "EF: " << EF_ << std::endl;
  a.resize(nedges_);
  #pragma omp parallel for
  for (int i = 0; i < nedges_; ++i) {
    const RowVector3d& n1 = n.row(EF_(i,0));
    const RowVector3d& n2 = n.row(EF_(i,1));
    a(i) = n1.dot(n2);
  }
}

void MixedSQPBending::grad_angles(const VectorXd& x, const MatrixXd& N,
    std::vector<VectorXd> da) {
  da.resize(nedges_);
  Dx_.resize(nedges_, J_.cols());
  auto cross_product_mat = [](const RowVector3d& v)-> Matrix3d {
    Matrix3d mat;
    mat <<     0, -v(2),  v(1),
            v(2),     0, -v(0),
           -v(1),  v(0),     0;
    return mat;
  };

// std::cout << "grad angles: " << std::endl; return;
  // a(i) = n1.dot(n2)
  // da/dx = dn1/dx * n2 + dn2/dx * n1
  std::vector<Triplet<double>> trips(nedges_*18);

  #pragma omp parallel for
  for(int i = 0; i < nedges_; ++i) {
    for (int j = 0; j < 2; ++j) {
      const RowVector3i& T = mesh_->T_.row(EF_(i,j));
      Vector3d v1 = x.segment<3>(3*T(1)) - x.segment<3>(3*T(0));
      Vector3d v2 = x.segment<3>(3*T(2)) - x.segment<3>(3*T(0));
      Vector3d n = v1.cross(v2);
      double l = n.norm();
      n /= l;

      Matrix3d dx1 = cross_product_mat(v1);
      Matrix3d dx2 = cross_product_mat(v2);

      Matrix<double, 3, 9> dn_dq;
      dn_dq.setZero();
      dn_dq.block<3,3>(0,0) = dx2 - dx1;
      dn_dq.block<3,3>(0,3) = -dx2;
      dn_dq.block<3,3>(0,6) = dx1;

      const RowVector3d& other = N.row(EF_(i,(j+1)%2));
      Vector9d da_dx = (other * ((Matrix3d::Identity() - n*n.transpose()) 
          * dn_dq / l)).transpose();
      for (int k =0; k < 3; ++k) {
        trips[18*i + 9*j + 3*k + 0] = Triplet<double>(i,3*T(k)+0,da_dx(3*k+0));
        trips[18*i + 9*j + 3*k + 1] = Triplet<double>(i,3*T(k)+1,da_dx(3*k+1));
        trips[18*i + 9*j + 3*k + 2] = Triplet<double>(i,3*T(k)+2,da_dx(3*k+2));
      }

    }
  }
  Dx_.setFromTriplets(trips.begin(),trips.end());
  PDW_ = P_ * Dx_.transpose() * L_;

}

bool MixedSQPBending::linesearch(Eigen::VectorXd& x, const Eigen::VectorXd& dx,
        Eigen::VectorXd& s, const Eigen::VectorXd& ds) {
  data_.timer.start("linesearch");

  auto value = [&](const VectorXd& xs)->double {
    return energy(xs.segment(0,x.size()), xs.segment(x.size(),s.size()), a_, la_, ga_);
  };

  VectorXd f(x.size() + s.size());
  f.segment(0,x.size()) = x;
  f.segment(x.size(),s.size()) = s;

  VectorXd g(x.size() + s.size());
  g.segment(0,x.size()) = dx;
  g.segment(x.size(),s.size()) = ds;

  double alpha = 1.0;
  VectorXd grad;

  // std::cout << "grad_ norm: " << grad_.norm() << std::endl;
  // SolverExitStatus status = linesearch_backtracking_bisection(f, g, value,
  //     grad, alpha, config_->ls_iters, 0.1, 0.5, E_prev_);
  SolverExitStatus status = linesearch_backtracking_cubic(f, g, value,
      grad_, alpha, config_->ls_iters, 1e-4, 0.5, E_prev_);    
  // std::cout << "ALPHA: " << alpha << std::endl;
  bool done = (status == MAX_ITERATIONS_REACHED);
  x = f.segment(0, x.size());
  s = f.segment(x.size(), s.size());
  data_.timer.stop("linesearch");
  return done;
}

void MixedSQPBending::reset() {

  MixedOptimizer::reset();

  igl::edge_topology(mesh_->V0_, mesh_->T_, EV_, FE_, EF_);

  ArrayXX<bool> valid = (EF_.col(0).array() != -1 && EF_.col(1).array() != -1);
  MatrixXi tmp1,tmp2;
  igl::slice_mask(EV_, valid, Array<bool,2,1>::Ones(), tmp1);
  igl::slice_mask(EF_, valid, Array<bool,2,1>::Ones(), tmp2);

  EV_ = tmp1;
  //FE_ = tmp2;
  EF_ = tmp2;
  igl::edge_lengths(mesh_->V0_, EV_, l_);

  nedges_ = l_.size(); 
  std::cout << "nedges: " << nedges_ << std::endl;

  // Initialize volume sparse matrix
  L_.resize(nedges_, nedges_);
  std::vector<Triplet<double>> trips;
  for (int i = 0; i < nedges_; ++i) {
    trips.push_back(Triplet<double>(i, i, l_(i)));
  }
  L_.setFromTriplets(trips.begin(),trips.end());

  n_.resize(mesh_->T_.rows(), 3);
  a_.resize(nedges_);
  a0_.resize(nedges_);
  ga_.resize(nedges_);
  s_.resize(3 * nelem_);
  la_.resize(3 * nelem_);
  la_.setZero();
  S_.resize(nelem_);
  H_.resize(nelem_);
  Hinv_.resize(nelem_);
  g_.resize(nelem_);
    // Make sure matrices are initially zero

  Vector3d I3(1,1,0);
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    S_[i] << I3;
    s_.segment<3>(3*i) = I3;
  }

  int curr = 0;
  std::vector<int> free_map(mesh_->is_fixed_.size(), -1);  
  for (int i = 0; i < mesh_->is_fixed_.size(); ++i) {
    if (mesh_->is_fixed_(i) == 0) {
      free_map[i] = curr++;
    }
  }
  assembler_ = std::make_shared<Assembler<double,3>>(mesh_->T_, free_map);
  vec_assembler_ = std::make_shared<VecAssembler<double,3>>(mesh_->T_,
      free_map);


  // SQP PD //
  SparseMatrixdRowMajor A;
  A.resize(nelem_*9, nelem_*9);
  trips.clear();
  for (int i = 0; i < nelem_; ++i) {
    for (int j = 0; j < 9; ++j) {
      trips.push_back(Triplet<double>(9*i+j, 9*i+j, mesh_->config_->mu / vols_[i]));
    }
  }
  A.setFromTriplets(trips.begin(),trips.end());

  double h2 = wdt_*wdt_*config_->h * config_->h;
  mesh_->jacobian(Jw_, vols_, true);
  mesh_->jacobian(Jloc_);
  PJ_ = P_ * Jw_.transpose();
  PM_ = P_ * Mfull_;

  SparseMatrixdRowMajor L = PJ_ * A * PJ_.transpose();
  SparseMatrixdRowMajor lhs = M_ + h2*L;
  solver_arap_.compute(lhs);

  //build up reduced space
  T0_.resize(3*mesh_->V0_.rows(), 12);

  //compute center of mass
  Eigen::Matrix3d I;
  Eigen::Vector3d c;
  double mass = 0;

  //std::cout<<"HERE 1 \n";
  // TODO wrong? should be F_ not T_ for tetrahedra
  sim::rigid_inertia_com(I, c, mass, mesh_->V0_, mesh_->T_, 1.0);

  for(unsigned int ii=0; ii<mesh_->V0_.rows(); ii++ ) {

    //std::cout<<"HERE 2 "<<ii<<"\n";
    T0_.block<3,3>(3*ii, 0) = Eigen::Matrix3d::Identity()*(mesh_->V0_(ii,0) - c(0));
    T0_.block<3,3>(3*ii, 3) = Eigen::Matrix3d::Identity()*(mesh_->V0_(ii,1) - c(1));
    T0_.block<3,3>(3*ii, 6) = Eigen::Matrix3d::Identity()*(mesh_->V0_(ii,2) - c(2));
    T0_.block<3,3>(3*ii, 9) = Eigen::Matrix3d::Identity();

  }

  T0_ = P_*T0_;
  //std::cout<<"c: "<<c.transpose()<<"\n";
  //std::cout<<"T0: \n"<<T0_<<"\n";

  Matrix<double, 12,12> tmp_pre_affine = T0_.transpose()*lhs*T0_; 
  pre_affine_ = tmp_pre_affine.inverse();
  //

  VectorXd xt = P_.transpose()*x_+b_;
  normals(xt, n_);
  angles(xt, n_, a0_);
  a_ = a0_;
  ga_.setZero();
  std::cout << "nedges: " << nedges_ << std::endl;
}
