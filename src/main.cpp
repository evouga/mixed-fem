#include "polyscope/polyscope.h"

// libigl
#include <igl/invert_diag.h>
#include <igl/readMESH.h>
#include <igl/volume.h>
#include <igl/svd3x3.h>

// Polyscope
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/volume_mesh.h"
#include "args/args.hxx"
#include "json/json.hpp"

// Bartels
#include <EigenTypes.h>
#include "linear_tetmesh_B.h"
#include "fixed_point_constraint_matrix.h"
#include "linear_tetmesh_mass_matrix.h"
#include "linear_tet_mass_matrix.h"


#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <unordered_set>
#include <utility>

#include <igl/rotation_matrix_from_directions.h>
#include <chrono>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif


using namespace std::chrono;
using namespace Eigen;

// The mesh, Eigen representation
MatrixXd meshV;
MatrixXi meshF;
MatrixXi meshT; // tetrahedra

VectorXd dq_la;

// F = RS
std::vector<Matrix3d> R; // Per-element rotations
std::vector<Vector6d> S; // Per-element symmetric deformation

SparseMatrix<double> M;     // mass matrix
SparseMatrix<double> P;     // pinning constraint (for vertices)
SparseMatrix<double> P_kkt; // pinning constraint (for kkt matrix)
SparseMatrix<double> J;     // jacobian
MatrixXd dphidX; 
VectorXi pinnedV;

// Simulation params
double h = 0.1;//0.1;
double density = 1000.0;
double ym = 1e6;
double mu = 0.45;
double alpha = 2e7; // 1e4;
double ih2 = 1.0/h/h;
double grav = -9.8;

TensorFixedSize<double, Sizes<3, 3, 9>> B_33;  // 9x1 -> 3x3
TensorFixedSize<double, Sizes<3, 3, 6>> B_33s; // 6x1 -> 3x3 (symmetric)
TensorFixedSize<double, Sizes<3, 9, 3, 6>> Te;

Matrix<double, 6,1> I_vec;


// Configuration vectors & body forces
VectorXd qt;    // current positions
VectorXd q0;    // previous positions
VectorXd q1;    // previous^2 positions
VectorXd f_ext; // per-node external forces
VectorXd la;    // lambdas
VectorXd b;     // coordinates projected out
VectorXd vols;  // per element volume

// KKT system
SparseMatrixd lhs;
#if defined(SIM_USE_CHOLMOD)
CholmodSimplicialLDLT<SparseMatrixd> kkt_solver;
#else
SimplicialLDLT<SparseMatrixd> kkt_solver;
#endif
VectorXd rhs;

// ------------ TODO ------------------ //
// 1. Not missing volume anywhere
// ------------------------------------ //


Eigen::SparseMatrix<double> pinning_matrix(VectorXi to_keep, bool kkt) {

  typedef Eigen::Triplet<double> T;
  std::vector<T> trips;

  int d = 3;
  int row =0;
  for (int i = 0; i < meshV.rows(); ++i) {
    if (to_keep(i)) {
      for (int j = 0; j < d; ++j) {
        trips.push_back(T(row++, d*i + j, 1));
      }
    }
  }

  int n = meshV.size();
  if (kkt) {
    for (int i = 0; i < meshT.rows(); ++i) {
      for (int j = 0; j < d*d; ++j) {
        trips.push_back(T(row++, n + d*d*i+j, 1));
      }
    }
    n += d*d*meshT.rows();
  }

  Eigen::SparseMatrix<double> A(row, n);
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
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
          double val = B(j,3*k+l);// TODO * vols(i) ; 
          // subdiagonal term
          trips.push_back(Triplet<double>(offset+9*i+j, 3*vid+l, val));

          // superdiagonal term
          trips.push_back(Triplet<double>(3*vid+l, offset+9*i+j, val));
        }
      }
    }

    // 3. Compliance matrix terms
    // Each local term is equivalent and is a scaled diagonal matrix
    double He = -1.0/alpha;
    for (int j = 0; j < 9; ++j) {
      trips.push_back(Triplet<double>(offset+9*i+j,offset+9*i+j, He));
    }
  }

  int sz = meshV.size() + meshT.rows()*9;
  lhs.resize(sz,sz);
  lhs.setFromTriplets(trips.begin(), trips.end());

#if defined(SIM_USE_CHOLMOD)
  std::cout << "using choldmod" << std::endl;
#endif
  lhs = P_kkt * lhs * P_kkt.transpose();
  kkt_solver.compute(lhs);
  if(kkt_solver.info()!=Success) {
    std::cerr << " KKT prefactor failed! " << std::endl;
  }
  //Eigen::IOFormat CleanFmt(2, 0, ",", "\n", "[", "]");
  //std::cout << "M: \n" << MatrixXd(ih2*M).format(CleanFmt) << std::endl;
  //std::cout << "J: \n" << MatrixXd(J).format(CleanFmt) << std::endl;
  //std::cout << "L: \n" << MatrixXd(lhs).format(CleanFmt) << std::endl;
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
    // 1. W * st term
    TensorMap<Tensor<double, 2>> Ri(R[i].data(), 3, 3);
    TensorFixedSize<double, Sizes<9,6>> TR = Te.contract(Ri, dims);
    Matrix<double,9,6> W = Map<Matrix<double,9,6>>(TR.data(),
        TR.dimension(0), TR.dimension(1));

    Vector9d Ws = W * (I_vec); 

    // 2. - W * Hinv * g term
    rhs.segment(qt.size() + 9*i, 9) = Ws;

    // Safe for 1 tet lolw
    //std::cout << "Ws - Jq (1 TET Only)" << std::endl; 
    //std::cout << J*(P.transpose()*qt+b)<< std::endl;;
    //std::cout << (W*S[i]-(J*(P.transpose()*qt+b))).norm() << std::endl;;

  }

  // 3. Jacobian term
  rhs.segment(qt.size(), 9*meshT.rows()) -= J*(P.transpose()*qt+b);
  //std::cout << "Jq size: " << (J*(P.transpose()*qt+b)).size() << std::endl;
}

//#define VERBOSE

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


  std::cout << "warning settings vols to 1" << std::endl;
  vols.setOnes();


  sim::linear_tetmesh_mass_matrix(M, meshV, meshT, densities, vols);

  // J matrix (big jacobian guy)
  std::vector<Triplet<double>> trips;
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
          double val = B(j,3*k+l);// TODO * vols(i); 
          trips.push_back(Triplet<double>(9*i+j, 3*vid+l, val));
        }
      }
    }
  }
  J.resize(9*meshT.rows(), meshV.size());
  J.setFromTriplets(trips.begin(),trips.end());

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
  pinnedV = (meshV.col(0).array() > pin_x).cast<int>(); 
  //pinnedV.setOnes();
  //pinnedV(1) = 0;
  polyscope::getVolumeMesh("input mesh")
    ->addVertexScalarQuantity("pinned", pinnedV);
  P = pinning_matrix(pinnedV, false);
  P_kkt = pinning_matrix(pinnedV, true);

  // Initial configuration vectors (assuming 0 initial velocity)
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


  build_kkt_lhs();
 // build_kkt_rhs();
 // update_SR();
}

void simulation_step() {
  // Does each iteration use dq anywhere besides being a solution of
  // the kkt solve?
  // ds & lambda are used in the projection, but to confirm delta q
  // is just one of the "end products"
  //
  int steps=10;//10;
  
  for (int i = 0; i < steps; ++i) {
    // TODO partially inefficient right?
    //  dq rows are fixed throughout the optimization, only lambda rows
    //  change correct?

    double t_rhs = 0, t_solve = 0, t_SR = 0; 
    auto start = high_resolution_clock::now();
    build_kkt_rhs();
    auto end = high_resolution_clock::now();
    t_rhs = duration_cast<nanoseconds>(end-start).count()/1e6;
    start = end;
    dq_la = kkt_solver.solve(rhs);
    end = high_resolution_clock::now();
    t_solve = duration_cast<nanoseconds>(end-start).count()/1e6;
    start=end;
    la = dq_la.segment(qt.size(),9*meshT.rows());
    
    // Update per-element R & S matrices
    update_SR();
    end = high_resolution_clock::now();
    t_SR = duration_cast<nanoseconds>(end-start).count()/1e6;

    std::cout << "Time kkt rhs: " << t_rhs << " solver: " << t_solve 
      << " update S & R: " << t_SR << std::endl;
  }

  q1 = q0;
  q0 = qt;
  qt += dq_la.segment(0, qt.size());

  // Initial configuration vectors (assuming 0 initial velocity)
  VectorXd q = P.transpose()*qt + b;
  MatrixXd tmp = Map<MatrixXd>(q.data(), meshV.cols(), meshV.rows());
  meshV = tmp.transpose();
}

void callback() {

  static int numPoints = 2000;
  static float param = 3.14;
  static bool simulating = false;
  static bool show_pinned = false;

  ImGui::PushItemWidth(100);

  ImGui::Checkbox("simulate",&simulating);
  if(ImGui::Button("show pinned")) {

    double min_x = meshV.col(0).minCoeff();
    double max_x = meshV.col(0).maxCoeff();
    double pin_x = min_x + (max_x-min_x)*0.1;
    pinnedV = (meshV.col(0).array() > pin_x).cast<int>(); 
    polyscope::getVolumeMesh("input mesh")
      ->addVertexScalarQuantity("pinned", pinnedV);
    P = pinning_matrix(pinnedV, false);
    P_kkt = pinning_matrix(pinnedV, true);
    //std::cout << "P: \n" << P << " Ptmp: \n" << Ptmp << std::endl;

  } 

  if(ImGui::Button("sim step") || simulating) {
    //for (int i = 0; i < R.size(); ++i) {
    //  std::cout << "R0["<<i<<"]: \n" << R[i] << std::endl;
    //}
    //for (int i = 0; i < R.size(); ++i) {
    //  std::cout << "S0["<<i<<"]: \n" << S[i] << std::endl;
    //}
    simulation_step();
    polyscope::getVolumeMesh("input mesh")
      ->updateVertexPositions(meshV);
    //std::cout << "Jq: \n" << J*(P.transpose()*qt+b) << std::endl;;
    //std::cout << "la: \n" << la << std::endl;
    //for (int i = 0; i < R.size(); ++i) {
    //  std::cout << "R["<<i<<"]: \n" << R[i] << std::endl;
    //}
    //for (int i = 0; i < R.size(); ++i) {
    //  std::cout << "S["<<i<<"]: \n" << S[i] << std::endl;
    //}
  }
    
  //ImGui::SameLine();
  //ImGui::InputInt("source vertex", &iVertexSource);

  ImGui::PopItemWidth();
}

int main(int argc, char **argv) {
  // Configure the argument parser
  args::ArgumentParser parser("A simple demo of Polyscope with libIGL.\nBy "
                              "Nick Sharp (nsharp@cs.cmu.edu)",
                              "");
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

  // Register the mesh with Polyscope
  polyscope::registerTetMesh("input mesh", meshV, meshT);

  pinnedV.resize(meshV.rows());
  pinnedV.setZero();
  polyscope::getVolumeMesh("input mesh")
    ->addVertexScalarQuantity("pinned", pinnedV);

  // Add the callback
  polyscope::state::userCallback = callback;

  // Initial simulation setup
  init_sim();

  // Show the gui
  polyscope::show();

  return 0;
}

