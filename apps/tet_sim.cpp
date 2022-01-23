#include "polyscope/polyscope.h"

// libigl
#include <igl/boundary_facets.h>
#include <igl/invert_diag.h>
#include <igl/readMESH.h>
#include <igl/volume.h>
#include <igl/svd3x3.h>
#include "svd3x3_sse.h"

// Polyscope
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/volume_mesh.h"
#include "polyscope/surface_mesh.h"
#include "args/args.hxx"
#include "json/json.hpp"

// Bartels
#include <EigenTypes.h>
#include "linear_tetmesh_mass_matrix.h"
#include "linear_tet_mass_matrix.h"
#include "linear_tetmesh_dphi_dX.h"


#include "pinning_matrix.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <unordered_set>
#include <utility>

#include <chrono>

#if defined(SIM_USE_OPENMP)
#include <omp.h>
#endif

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

using namespace std::chrono;
using namespace Eigen;
using SparseMatrixdRowMajor = Eigen::SparseMatrix<double,RowMajor>;

// The mesh, Eigen representation
MatrixXd meshV;
MatrixXi meshF;
MatrixXi meshT; // tetrahedra

VectorXd dq_la;

// F = RS
std::vector<Matrix3d> R; // Per-element rotations
std::vector<Vector6d> S; // Per-element symmetric deformation

SparseMatrixd M;          // mass matrix
SparseMatrixd P;          // pinning constraint (for vertices)
SparseMatrixd P_kkt;      // pinning constraint (for kkt matrix)
SparseMatrixdRowMajor Jw; // integrated (weighted) jacobian
SparseMatrixdRowMajor J;  // jacobian
MatrixXd dphidX; 
VectorXi pinnedV;

// Simulation params
double h = 0.02;//0.1;
double density = 1000.0;
double ym = 2e5;
double pr = 0.45;
double mu = ym/(2.0*(1.0+pr));
double alpha = mu;
double ih2 = 1.0/h/h;
double grav = -9.8;
double plane_d;

bool warm_start = true;
bool floor_collision = true;

TensorFixedSize<double, Sizes<3, 3, 9>> B_33;  // 9x1 -> 3x3
TensorFixedSize<double, Sizes<3, 3, 6>> B_33s; // 6x1 -> 3x3 (symmetric)
TensorFixedSize<double, Sizes<3, 9, 3, 6>> Te;

Matrix<double, 6,1> I_vec;


// Configuration vectors & body forces
VectorXd qt;    // current positions
VectorXd q0;    // previous positions
VectorXd q1;    // previous^2 positions
VectorXd f_ext; // per-node external forces
VectorXd f_ext0;// per-node external forces (not integrated)
VectorXd la;    // lambdas
VectorXd b;     // coordinates projected out
VectorXd vols;  // per element volume

// KKT system
SparseMatrixd lhs;
#if defined(SIM_USE_CHOLMOD)
CholmodSimplicialLDLT<SparseMatrixd> solver;
#else
SimplicialLDLT<SparseMatrixd> solver;
#endif
VectorXd rhs;

// ------------------------------------ //

VectorXd collision_force() {

  //Vector3d N(plane(0),plane(1),plane(2));
  //Vector3d N(.4,.7,0);
  Vector3d N(0.,1.,0.);
  N = N / N.norm();
  double d = plane_d;

  int n = qt.size() / 3;
  VectorXd ret(qt.size());
  ret.setZero();

  double k = 80; //20 for octopus ssliding

  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    Vector3d xi(qt(3*i)+dq_la(3*i),
        qt(3*i+1)+dq_la(3*i+1),
        qt(3*i+2)+dq_la(3*i+2));
    double dist = xi.dot(N);
    if (dist < plane_d) {
      ret.segment(3*i,3) = k*(plane_d-dist)*N;
    }
  }
  return M*ret;
}

void build_kkt_lhs() {

  VectorXd densities = VectorXd::Constant(meshT.rows(), density);
  sim::linear_tetmesh_dphi_dX(dphidX, meshV, meshT);

  std::vector<Triplet<double>> trips;

  for (int i = 0; i < meshT.rows(); ++i) {
    // 1. Mass matrix terms
    Matrix12d Me;
    sim::linear_tet_mass_matrix(Me, meshT.row(i), densities(i), vols(i)); 

    // mass matrix assembly
    for (int j = 0; j < 4; ++j) {
      int vid1 = meshT(i,j);
      for (int k = 0; k < 4; ++k) {
        int vid2 = meshT(i,k);
        for (int l = 0; l < 3; ++l) {
          for (int m = 0; m < 3; ++m) {
            double val = ih2 * Me(3*j+l, 3*k+m);
            trips.push_back(Triplet<double>(3*vid1+l, 3*vid2+m, val));
          }
        }
      }
    }

    // 2. J matrix (and transpose) terms
    // Local jacobian
    Matrix<double, 4,3> dX = sim::unflatten<4,3>(dphidX.row(i));

    // Local block
    Matrix<double,9,12> B;
    B  << dX(0,0), 0      , 0      , dX(1,0), 0      , 0      , dX(2,0), 0      , 0      , dX(3,0),       0,       0,
          dX(0,1), 0      , 0      , dX(1,1), 0      , 0      , dX(2,1), 0      , 0      , dX(3,1),       0,       0, 
          dX(0,2), 0      , 0      , dX(1,2), 0      , 0      , dX(2,2), 0      , 0      , dX(3,2),       0,       0, 
          0      , dX(0,0), 0      , 0      , dX(1,0), 0      , 0      , dX(2,0), 0      , 0      , dX(3,0),       0,
          0      , dX(0,1), 0      , 0      , dX(1,1), 0      , 0      , dX(2,1), 0      , 0      , dX(3,1),       0,
          0      , dX(0,2), 0      , 0      , dX(1,2), 0      , 0      , dX(2,2), 0      , 0      , dX(3,2),       0,
          0      , 0      , dX(0,0), 0      , 0      , dX(1,0), 0      , 0      , dX(2,0), 0      , 0      , dX(3,0),
          0      , 0      , dX(0,1), 0      , 0      , dX(1,1), 0      , 0      , dX(2,1), 0      , 0      , dX(3,1),
          0      , 0      , dX(0,2), 0      , 0      , dX(1,2), 0      , 0      , dX(2,2), 0      , 0      , dX(3,2); 
    int offset = meshV.size(); // offset for off diagonal blocks

    // Assembly for the i-th lagrange multiplier matrix which
    // is associated with 4 vertices (for tetrahedra)
    for (int j = 0; j < 9; ++j) {

      // k-th vertex of the tetrahedra
      for (int k = 0; k < 4; ++k) {
        int vid = meshT(i,k); // vertex index

        // x,y,z index for the k-th vertex
        for (int l = 0; l < 3; ++l) {
          double val = B(j,3*k+l) * vols(i); 
          // subdiagonal term
          trips.push_back(Triplet<double>(offset+9*i+j, 3*vid+l, val));

          // superdiagonal term
          trips.push_back(Triplet<double>(3*vid+l, offset+9*i+j, val));
        }
      }
    }

    // 3. Compliance matrix terms
    // Each local term is equivalent and is a scaled diagonal matrix
    double He = -vols(i)/alpha;
    for (int j = 0; j < 9; ++j) {
      trips.push_back(Triplet<double>(offset+9*i+j,offset+9*i+j, He));
    }
  }

  int sz = meshV.size() + meshT.rows()*9;
  lhs.resize(sz,sz);
  lhs.setFromTriplets(trips.begin(), trips.end());
  lhs = P_kkt * lhs * P_kkt.transpose();

  #if defined(SIM_USE_CHOLMOD)
  std::cout << "Using CHOLDMOD solver" << std::endl;
  #endif
  solver.compute(lhs);
  if(solver.info()!=Success) {
    std::cerr << " KKT prefactor failed! " << std::endl;
  }
}

void build_kkt_rhs() {
  int sz = qt.size() + meshT.rows()*9;
  rhs.resize(sz);
  rhs.setZero();

  // Positional forces 
  rhs.segment(0, qt.size()) = f_ext + ih2*M*(q0 - q1);

  // Lagrange multiplier forces
  array<IndexPair<int>, 2> dims = { 
    IndexPair<int>(0, 0), IndexPair<int>(2, 1)
  };

  #pragma omp parallel for
  for (int i = 0; i < meshT.rows(); ++i) {
    // 1. W * st term +  - W * Hinv * g term
    // Becomes W * I_vec = flatten(R[i]^t)
    Vector9d Ws = vols(i) * sim::flatten(R[i].transpose()); 
    rhs.segment(qt.size() + 9*i, 9) = Ws;
  }

  // 3. Jacobian term
  rhs.segment(qt.size(), 9*meshT.rows()) -= Jw*(P.transpose()*qt+b);
}

void update_SR_fast() {

  // Local hessian (constant for all elements)
  Vector6d He_inv_vec;
  He_inv_vec << 1,1,1,0.5,0.5,0.5;
  He_inv_vec /= alpha;
  DiagonalMatrix<double,6> He_inv = He_inv_vec.asDiagonal();

  array<IndexPair<int>, 2> dims = { 
    IndexPair<int>(0, 0), IndexPair<int>(2, 1)
  };

  double t_1=0,t_2=0,t_3=0,t_4=0,t_5=0; 
  auto start = high_resolution_clock::now();
  VectorXd def_grad = J*(P.transpose()*(qt+dq_la.segment(0, qt.rows()))+b);
  auto end = high_resolution_clock::now();
  t_1 = duration_cast<nanoseconds>(end-start).count()/1e6;

  int N = (meshT.rows() / 4) + int(meshT.rows() % 4 != 0);

  #pragma omp parallel for 
  for (int ii = 0; ii < N; ++ii) {

    Matrix<float,12,3> Y4,R4;

    for (int jj = 0; jj < 4; ++jj) {
      int i = ii*4 +jj;
      if (i >= meshT.rows())
        break;


      Vector9d li = la.segment(9*i,9)/alpha + def_grad.segment(9*i,9);

      // 1. Update S[i] using new lambdas
      start = high_resolution_clock::now();
      // (la    B ) : (R    C    s) 
      //  1x9 9x3x3   3x3 3x3x6 6x1
      //  we want (la B) : (R C)
      Matrix<double,9,6> W;
      W <<
        R[i](0,0), 0,         0,         0,         R[i](0,2), R[i](0,1),
        0,         R[i](0,1), 0,         R[i](0,2), 0,         R[i](0,0),
        0,         0,         R[i](0,2), R[i](0,1), R[i](0,0), 0, 
        R[i](1,0), 0,         0,         0,         R[i](1,2), R[i](1,1),
        0,         R[i](1,1), 0,         R[i](1,2), 0,         R[i](1,0),
        0,         0,         R[i](1,2), R[i](1,1), R[i](1,0), 0,  
        R[i](2,0), 0,         0,         0,         R[i](2,2), R[i](2,1),
        0,         R[i](2,1), 0,         R[i](2,2), 0        , R[i](2,0),
        0,         0,         R[i](2,2), R[i](2,1), R[i](2,0), 0;
      //TensorMap<Tensor<double, 2>> Ri(R[i].data(), 3, 3);
      //TensorFixedSize<double, Sizes<9,6>> TR = Te.contract(Ri, dims);
      //Matrix<double,9,6> W = Map<Matrix<double,9,6>>(TR.data(),
      //    TR.dimension(0), TR.dimension(1));
      end = high_resolution_clock::now();
      t_2 += duration_cast<nanoseconds>(end-start).count()/1e6;

      // H^-1 * g = s^i - I
      start = high_resolution_clock::now();
      Vector6d ds = He_inv*W.transpose()*la.segment(9*i,9) -(S[i]-I_vec); 
      S[i] += ds;
      end = high_resolution_clock::now();
      t_3 += duration_cast<nanoseconds>(end-start).count()/1e6;

      // 2. Solve rotation matrices

      start = high_resolution_clock::now();
      //array<IndexPair<int>, 1> dims_l = {IndexPair<int>(1, 0)};
      //array<IndexPair<int>, 1> dims_s = {IndexPair<int>(2, 0)};
      //TensorMap<Tensor<double, 1>> l(li.data(), 9);
      //TensorMap<Tensor<double, 1>> Si(S[i].data(), 6);
      //TensorFixedSize<double, Sizes<3,3,6>> Tel = Te.contract(l, dims_l);
      //TensorFixedSize<double, Sizes<3,3>> Tels = Tel.contract(Si, dims_s);
      //Matrix3d Y = Map<Matrix3d>(Tels.data(), 3, 3);
      //Y4.block(3*jj, 0, 3, 3) = Y.cast<float>();

      //  la    B     C    R   s
      //  1x9 9x3x3 3x3x6 3x3 6x1 
      //  we want [la   B ]  [ C    s]
      //          1x9 9x3x3  3x3x6 6x1
      Matrix3d Cs;
      Cs << S[i](0), S[i](5), S[i](4), 
            S[i](5), S[i](1), S[i](3), 
            S[i](4), S[i](3), S[i](2); 
      Matrix3d y4 = Map<Matrix3d>(li.data()).transpose()*Cs;
      Y4.block(3*jj, 0, 3, 3) = y4.cast<float>();

      //std::cout << (y4-Y).norm() << std::endl;
      //if (i == 0) {
      //  std::cout << "Y3: \n " << Y << std::endl;
      //  std::cout << "Y3new: \n " << y4 << std::endl;
      //}
      end = high_resolution_clock::now();
      t_4 += duration_cast<nanoseconds>(end-start).count()/1e6;
    }
    // Solve rotations
    auto start = high_resolution_clock::now();
    polar_svd3x3_sse(Y4,R4);
    for (int jj = 0; jj < 4; jj++) {
      int i = ii*4 +jj;
      if (i >= meshT.rows())
        break;
      R[i] = R4.block(3*jj,0,3,3).cast<double>();
    }
    auto end = high_resolution_clock::now();
    t_5 += duration_cast<nanoseconds>(end-start).count()/1e6;
  }
  //std::cout << "Def grad: " << t_1 << "ms S[i]: " << t_2 << "ms ds: " << t_3
  //  << "ms Yf: " << t_4 << "ms SVD: " << t_5 << std::endl;
}

void update_SR() {

  // Local hessian (constant for all elements)
  Vector6d He_inv_vec;
  He_inv_vec << 1,1,1,0.5,0.5,0.5;
  He_inv_vec /= alpha;
  DiagonalMatrix<double,6> He_inv = He_inv_vec.asDiagonal();

  array<IndexPair<int>, 2> dims = { 
    IndexPair<int>(0, 0), IndexPair<int>(2, 1)
  };

  double t_1=0,t_2=0,t_3=0,t_4=0,t_5=0; 
  auto start = high_resolution_clock::now();
  VectorXd def_grad = J*(P.transpose()*(qt+dq_la.segment(0, qt.rows()))+b);
  auto end = high_resolution_clock::now();
  t_1 = duration_cast<nanoseconds>(end-start).count()/1e6;

  #pragma omp parallel for 
  for (int i = 0; i < meshT.rows(); ++i) {
    Vector9d li = la.segment(9*i,9)/alpha + def_grad.segment(9*i,9);

    // 1. Update S[i] using new lambdas
    start = high_resolution_clock::now();
    TensorMap<Tensor<double, 2>> Ri(R[i].data(), 3, 3);
    TensorFixedSize<double, Sizes<9,6>> TR = Te.contract(Ri, dims);
    Matrix<double,9,6> W = Map<Matrix<double,9,6>>(TR.data(),
        TR.dimension(0), TR.dimension(1));
    end = high_resolution_clock::now();
    t_2 += duration_cast<nanoseconds>(end-start).count()/1e6;

    // H^-1 * g = s^i - I
    start = high_resolution_clock::now();
    VectorXd ds = He_inv*W.transpose()*la.segment(9*i,9) -(S[i]-I_vec); 
    S[i] += ds;
    end = high_resolution_clock::now();
    t_3 += duration_cast<nanoseconds>(end-start).count()/1e6;

    // 2. Solve rotation matrices
    Matrix3f U,V; 
    Vector3f s; 
    Matrix3d Y;

    start = high_resolution_clock::now();
    array<IndexPair<int>, 1> dims_l = {IndexPair<int>(1, 0)};
    array<IndexPair<int>, 1> dims_s = {IndexPair<int>(2, 0)};
    TensorMap<Tensor<double, 1>> l(li.data(), 9);
    TensorMap<Tensor<double, 1>> Si(S[i].data(), 6);
    TensorFixedSize<double, Sizes<3,3,6>> Tel = Te.contract(l, dims_l);
    TensorFixedSize<double, Sizes<3,3>> Tels = Tel.contract(Si, dims_s);
    Y = Map<Matrix3d>(Tels.data(), 3, 3);
    Matrix3f Yf = Y.cast<float>();
    end = high_resolution_clock::now();
    t_4 += duration_cast<nanoseconds>(end-start).count()/1e6;
    start = high_resolution_clock::now();
    //std::cout << "Yf : \n " << Yf << std::endl;
    //std::cout << "l: \n" << l << std::endl;
    //std::cout << "Si: \n" << Si << std::endl;
    //std::cout << "Tels: \n" << Tels << std::endl;
    //
    //IGL_INLINE void igl::polar_svd3x3_sse(const Eigen::Matrix<T, 3*4, 3>& A, Eigen::Matrix<T, 3*4, 3> &R)
    Matrix<float,12,3> Y4,R4;
    polar_svd3x3_sse(Y4,R4);

    igl::svd3x3(Yf, U, s, V);

    R[i] = (U*V.transpose()).cast<double>();
    //R[i].transposeInPlace();
    end = high_resolution_clock::now();
    t_5 += duration_cast<nanoseconds>(end-start).count()/1e6;
  }
  //std::cout << "Def grad: " << t_1 << "ms S[i]: " << t_2 << "ms ds: " << t_3
  //  << "ms Yf: " << t_4 << "ms SVD: " << t_5 << std::endl;
}


void init_sim() {

  //rotate the mesh
  /*Matrix3d R_test;
  R_test << 0.707, -0.707, 0,
            0.707, 0.707, 0,
            0, 0, 1;*/

  I_vec << 1, 1, 1, 0, 0, 0; // Identity in symmetric format

  // Initialize rotation matrices to identity
  R.resize(meshT.rows());
  S.resize(meshT.rows());
  for (int i = 0; i < meshT.rows(); ++i) {
    R[i].setIdentity();
    //R[i] = R_test;
    S[i] = I_vec;
  }

  // Initial lambdas
  la.resize(9 * meshT.rows());
  la.setZero();

  // Mass matrix
  VectorXd densities = VectorXd::Constant(meshT.rows(), density);
  igl::volume(meshV, meshT, vols);
  //vols.array() /= vols.maxCoeff(); 
  //std::cout << "VOLS: " << vols << std::endl;
  //std::cout << "warning settings vols to 1" << std::endl;
  //vols.setOnes();


  sim::linear_tetmesh_mass_matrix(M, meshV, meshT, densities, vols);

  // J matrix (big jacobian guy)
  std::vector<Triplet<double>> trips;
  std::vector<Triplet<double>> trips2;
  sim::linear_tetmesh_dphi_dX(dphidX, meshV, meshT);
  for (int i = 0; i < meshT.rows(); ++i) { 

    Matrix<double, 4,3> dX = sim::unflatten<4,3>(dphidX.row(i));

    // Local block
    Matrix<double,9,12> B;
    B  << dX(0,0), 0      , 0      , dX(1,0), 0      , 0      , dX(2,0), 0      , 0      , dX(3,0),       0,       0,
          dX(0,1), 0      , 0      , dX(1,1), 0      , 0      , dX(2,1), 0      , 0      , dX(3,1),       0,       0, 
          dX(0,2), 0      , 0      , dX(1,2), 0      , 0      , dX(2,2), 0      , 0      , dX(3,2),       0,       0, 
          0      , dX(0,0), 0      , 0      , dX(1,0), 0      , 0      , dX(2,0), 0      , 0      , dX(3,0),       0,
          0      , dX(0,1), 0      , 0      , dX(1,1), 0      , 0      , dX(2,1), 0      , 0      , dX(3,1),       0,
          0      , dX(0,2), 0      , 0      , dX(1,2), 0      , 0      , dX(2,2), 0      , 0      , dX(3,2),       0,
          0      , 0      , dX(0,0), 0      , 0      , dX(1,0), 0      , 0      , dX(2,0), 0      , 0      , dX(3,0),
          0      , 0      , dX(0,1), 0      , 0      , dX(1,1), 0      , 0      , dX(2,1), 0      , 0      , dX(3,1),
          0      , 0      , dX(0,2), 0      , 0      , dX(1,2), 0      , 0      , dX(2,2), 0      , 0      , dX(3,2);         

    // Assembly for the i-th lagrange multiplier matrix which
    // is associated with 4 vertices (for tetrahedra)
    for (int j = 0; j < 9; ++j) {

      // k-th vertex of the tetrahedra
      for (int k = 0; k < 4; ++k) {
        int vid = meshT(i,k); // vertex index

        // x,y,z index for the k-th vertex
        for (int l = 0; l < 3; ++l) {
          double val = B(j,3*k+l) ;//* vols(i); 
          trips.push_back(Triplet<double>(9*i+j, 3*vid+l, val));
          trips2.push_back(Triplet<double>(9*i+j, 3*vid+l, val*vols(i)));
        }
      }
    }
  }
  J.resize(9*meshT.rows(), meshV.size());
  J.setFromTriplets(trips.begin(),trips.end());
  Jw.resize(9*meshT.rows(), meshV.size());
  Jw.setFromTriplets(trips2.begin(),trips2.end());

  // Create local tensors used for reshaping vectors to matrices
  B_33.setValues({
      {
        {1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0, 0},
      },
      {
        {0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 0, 0},
      },
      {
        {0, 0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 1},
      }
  });
  //TODO confirm ordering is okay
  B_33s.setValues({
      {
        {1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1, 0} 
      },
      {
        {0, 0, 0, 0, 0, 1},
        {0, 1, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0} 
      },  
      {
        {0, 0, 0, 0, 1, 0},
        {0, 0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0, 0} 
      }
  }); 
  array<IndexPair<int>, 1> dims = {IndexPair<int>(1, 1)
  };
  Te = B_33.contract(B_33s, dims);

  // Pinning matrices
  double min_x = meshV.col(0).minCoeff();
  double max_x = meshV.col(0).maxCoeff();
  double pin_x = min_x + (max_x-min_x)*0.3;
  double min_y = meshV.col(1).minCoeff();
  double max_y = meshV.col(1).maxCoeff();
  double pin_y = max_y - (max_y-min_y)*0.1;
  //double pin_y = min_y + (max_y-min_y)*0.1;
  //pinnedV = (meshV.col(0).array() < pin_x).cast<int>(); 
  //pinnedV = (meshV.col(1).array() > pin_y).cast<int>(); 
  //pinnedV(0) = 1;
  //polyscope::getVolumeMesh("input mesh")
  polyscope::getSurfaceMesh("input mesh")
    ->addVertexScalarQuantity("pinned", pinnedV);
  P = pinning_matrix(meshV,meshT,pinnedV,false);
  P_kkt = pinning_matrix(meshV,meshT,pinnedV,true);

  MatrixXd tmp = meshV.transpose();
  //MatrixXd tmp = (R_test*meshV.transpose());
  //MatrixXd tmp = Rot*meshV.transpose();

  qt = Map<VectorXd>(tmp.data(), meshV.size());

  b = qt - P.transpose()*P*qt;
  qt = P * qt;
  q0 = qt;
  q1 = qt;
  dq_la = 0*qt;

  // Project out mass matrix pinned point
  M = P * M * P.transpose();

  // External gravity force
  f_ext = M * P *Vector3d(0,grav,0).replicate(meshV.rows(),1);
  f_ext0 = P *Vector3d(0,grav,0).replicate(meshV.rows(),1);


  build_kkt_lhs();

  //EigenSolver<MatrixXd> eigensolver;
  //eigensolver.compute(MatrixXd(lhs));
  //std::cout << "Evals: \n" << eigensolver.eigenvalues().real() << std::endl;
  //std::cout << "LHS norm: " << lhs.norm() << std::endl;

}

void simulation_step() {
  // Does each iteration use dq anywhere besides being a solution of
  // the kkt solve?
  // ds & lambda are used in the projection, but to confirm delta q
  // is just one of the "end products"
  //
  int steps=10;
  dq_la.setZero();

  // Warm start solver
  if (warm_start) {
    dq_la.segment(0,qt.size()) = (qt-q0) + h*h*f_ext0;
    update_SR_fast();
  }
  
  double t_coll=0, t_rhs = 0, t_solve = 0, t_SR = 0; 
  for (int i = 0; i < steps; ++i) {
    // TODO partially inefficient right?
    //  dq rows are fixed throughout the optimization, only lambda rows
    //  change correct?
    auto start = high_resolution_clock::now();
    build_kkt_rhs();
    auto end = high_resolution_clock::now();
    t_rhs += duration_cast<nanoseconds>(end-start).count()/1e6;
    start = end;

    if (floor_collision) {
      VectorXd f_coll = collision_force();
      t_coll += duration_cast<nanoseconds>(end-start).count()/1e6;
      rhs.segment(0,qt.size()) += f_coll;
      end = high_resolution_clock::now();
      t_coll += duration_cast<nanoseconds>(end-start).count()/1e6;
      start = end;
    }

    dq_la = solver.solve(rhs);

    end = high_resolution_clock::now();
    t_solve += duration_cast<nanoseconds>(end-start).count()/1e6;
    start=end;
    la = dq_la.segment(qt.size(),9*meshT.rows());
    
    // Update per-element R & S matrices
    update_SR_fast();
    end = high_resolution_clock::now();
    t_SR += duration_cast<nanoseconds>(end-start).count()/1e6;

  }
  t_coll/=steps; t_rhs /=steps; t_solve/=steps; t_SR/=steps;
  std::cout << "[Avg Time ms] collision: " << t_coll << 
    " rhs: " << t_rhs << " solver: " << t_solve <<
    " update S & R: " << t_SR << std::endl;

  q1 = q0;
  q0 = qt;
  qt += dq_la.segment(0, qt.size());

  // Initial configuration vectors (assuming 0 initial velocity)
  VectorXd q = P.transpose()*qt + b;
  MatrixXd tmp = Map<MatrixXd>(q.data(), meshV.cols(), meshV.rows());
  meshV = tmp.transpose();
}

void callback() {

  static bool simulating = false;
  static bool show_pinned = false;

  ImGui::PushItemWidth(100);

  ImGui::Checkbox("floor collision",&floor_collision);
  ImGui::Checkbox("warm start",&warm_start);
  ImGui::Checkbox("simulate",&simulating);
  //if(ImGui::Button("show pinned")) {
  //} 

  if(ImGui::Button("sim step") || simulating) {
    simulation_step();
    //polyscope::getVolumeMesh("input mesh")
    polyscope::getSurfaceMesh("input mesh")
      ->updateVertexPositions(meshV);
  }
  //ImGui::SameLine();
  //ImGui::InputInt("source vertex", &iVertexSource);

  ImGui::PopItemWidth();
}

int main(int argc, char **argv) {
  // Configure the argument parser
  args::ArgumentParser parser("Mixed FEM");
  args::Positional<std::string> inFile(parser, "mesh", "input mesh");

  // Parse args
  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError e) {
    std::cerr << e.what() << std::endl;

    std::cerr << parser;
    return 1;
  }

  // Options
  polyscope::options::autocenterStructures = true;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;

  // Initialize polyscope
  polyscope::init();

  std::string filename = args::get(inFile);
  std::cout << "loading: " << filename << std::endl;

  // Read the mesh
  igl::readMESH(filename, meshV, meshT, meshF);
  meshV.array() /= meshV.maxCoeff();

  // Register the mesh with Polyscope
  polyscope::options::autocenterStructures = false;
  //polyscope::registerTetMesh("input mesh", meshV, meshT);
  if (meshF.size() == 0){ 
    igl::boundary_facets(meshT, meshF);
  }
  polyscope::registerSurfaceMesh("input mesh", meshV, meshF);

  pinnedV.resize(meshV.rows());
  pinnedV.setZero();
  //polyscope::getVolumeMesh("input mesh")
  polyscope::getSurfaceMesh("input mesh")
    ->addVertexScalarQuantity("pinned", pinnedV);

  // Add the callback
  polyscope::state::userCallback = callback;

  // Adjust ground plane
  VectorXd a = meshV.colwise().minCoeff();
  VectorXd b = meshV.colwise().maxCoeff();
  a(1) -= (b(1)-a(1))*0.5;
  plane_d = a(1);
  polyscope::options::automaticallyComputeSceneExtents = false;
  polyscope::state::lengthScale = 1.;
  polyscope::state::boundingBox = std::tuple<glm::vec3, glm::vec3>{
    {a(0),a(1),a(2)},{b(0),b(1),b(2)}};

  // Initial simulation setup
  init_sim();

  // Show the gui
  polyscope::show();

  return 0;
}

