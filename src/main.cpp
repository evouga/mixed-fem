#include "polyscope/polyscope.h"

// libigl
#include <igl/invert_diag.h>
#include <igl/readMESH.h>
#include <igl/volume.h>

// Polyscope
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/volume_mesh.h"
#include "args/args.hxx"
#include "json/json.hpp"

// Bartels
#include "EigenTypes.h"
#include "linear_tetmesh_B.h"
#include "fixed_point_constraint_matrix.h"
#include "linear_tetmesh_mass_matrix.h"
#include "linear_tet_mass_matrix.h"


#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <unordered_set>
#include <utility>

using namespace Eigen; // fuck it

// The mesh, Eigen representation
MatrixXd meshV;
MatrixXi meshF;
MatrixXi meshT; // tetrahedra

// F = RS
std::vector<Matrix3d> R; // Per-element rotations
std::vector<Vector6d> S; // Per-element symmetric deformation


SparseMatrix<double> M; // mass matrix
SparseMatrix<double> P; // pinning constraint
SparseMatrix<double> J; // jacobian
MatrixXd dphidX; 
VectorXi pinnedV;

// Simulation params
double h = 0.1;
double density = 1000.0;
double ym = 1e6;
double mu = 0.45;
double alpha = 100;
double ih2 = 1/h/h;
double grav = -9.8;

DiagonalMatrix<double, 9> WHiW; // local kkt hessian sandwhich (-W(H^-1)W^T)

TensorFixedSize<double, Sizes<3, 3, 9>> B_33;  // 9x1 -> 3x3
TensorFixedSize<double, Sizes<3, 3, 6>> B_33s; // 6x1 -> 3x3 (symmetric)
TensorFixedSize<double, Sizes<3, 9, 3, 6>> Te;

Vector6d I_vec = {1, 1, 1, 0, 0, 0}; // Identity is symmetric format

// Configuration vectors & body forces
VectorXd qt;    // current positions
VectorXd q0;    // previous positions
VectorXd q1;    // previous^2 positions
VectorXd f_ext; // per-node external forces

// KKT system
SparseMatrixd lhs;
SimplicialLDLT<SparseMatrixd> kkt_solver;
VectorXd rhs;

// ------------ TODO ------------------ //
// 1. Not missing volume anywhere
// 2. Tetrahedra pinning (for lambda)
// ------------------------------------ //

void build_kkt_lhs() {

  VectorXd densities = VectorXd::Constant(meshT.rows(), density);
  VectorXd vols;
  igl::volume(meshV, meshT, vols);
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
    Matrix<double,9,12> B = sim::flatten_multiply_right<
      Matrix<double, 3,4>>(sim::unflatten<4,3>(dphidX.row(i)));

    int offset = meshV.size(); // offset for off diagonal blocks

    // Assembly for the i-th lagrange multiplier matrix which
    // is associated with 4 vertices (for tetrahedra)
    for (int j = 0; j < 9; ++j) {

      // k-th vertex of the tetrahedra
      for (int k = 0; k < 4; ++k) {
        int vid = meshT(i,k); // vertex index

        // x,y,z index for the k-th vertex
        for (int l = 0; l < 3; ++l) {
          double val = B(j,3*k+l); 
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
  int sz = meshV.size() + meshT.rows()*9;
  rhs.resize(sz);
  rhs.setZero();

  // positional forces 
  rhs.segment(0, qt.size()) = f_ext - ih2*M*(qt - 2.0*q0 + q1);

  // lagrange multiplier forces
  
  VectorXd Jq = J*qt;
  array<IndexPair<int>, 2> dims = { 
    IndexPair<int>(0, 0), IndexPair<int>(2, 1)
  };
  //Te = B_33.contract(B_33s, dims);
  //TensorFixedSize<double, Sizes<3, 9, 3, 6>> Te;

  for (int i = 0; i < meshT.rows(); ++i) {
    // 1. W * st term
    TensorMap<Tensor<double, 2>> Ri(R[i].data(), 3, 3);
    TensorFixedSize<double, Sizes<9,6>> TR = Te.contract(Ri, dims);
    Matrix<double,9,6> W = Map<Matrix<double,9,6>>(TR.data(),
        TR.dimension(0), TR.dimension(1));

    Vector9d Ws = W * (I_vec); 
    //auto rhs_le = Je_ql*qe -Te * se - Te*(se-i_vec);
    // 3. - W * Hinv * g term
    rhs.segment(meshV.size() + 9*i, 9) = Ws;
  }
  rhs.segment(meshV.size(), 9*meshT.rows()) -= Jq;

  std::cout << "Jq size: " << Jq.size() << std::endl;
}

void init_sim() {

  // Initial configuration vectors (assuming 0 initial velocity)
  MatrixXd tmp = meshV.transpose();
  qt = Map<VectorXd>(tmp.data(), meshV.size());
  q0 = qt;
  q1 = qt;

  // Initialize rotation matrices to identity
  R.resize(meshT.rows());
  S.resize(meshT.rows());
  for (int i = 0; i < meshT.rows(); ++i) {
    R[i].setIdentity();
    S[i] = I_vec;
  }

  // Mass matrix
  VectorXd densities = VectorXd::Constant(meshT.rows(), density);
  VectorXd vols;
  igl::volume(meshV, meshT, vols);
  sim::linear_tetmesh_mass_matrix(M, meshV, meshT, densities, vols);

  // External gravity force
  f_ext = M * Vector3d(0,grav,0).replicate(meshV.rows(),1);

  // Local arap inverse hessian
  Vector9d Htmp = Vector9d::Ones() * -(1.0/alpha);
  WHiW = Htmp.asDiagonal();
  
  // J matrix (big jacobian guy)
  std::vector<Triplet<double>> trips;
  sim::linear_tetmesh_dphi_dX(dphidX, meshV, meshT);
  for (int i = 0; i < meshT.rows(); ++i) { 
    // Local block
    Matrix<double,9,12> B = sim::flatten_multiply_right<
      Matrix<double, 3,4>>(sim::unflatten<4,3>(dphidX.row(i)));

    // Assembly for the i-th lagrange multiplier matrix which
    // is associated with 4 vertices (for tetrahedra)
    for (int j = 0; j < 9; ++j) {

      // k-th vertex of the tetrahedra
      for (int k = 0; k < 4; ++k) {
        int vid = meshT(i,k); // vertex index

        // x,y,z index for the k-th vertex
        for (int l = 0; l < 3; ++l) {
          double val = B(j,3*k+l); 
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
  //array<IndexPair<int>, 2> double_dims = { 
  //  IndexPair<int>(0, 0), IndexPair<int>(1, 1)
  //};
  
  Te = B_33.contract(B_33s, dims);
  //std::cout << "T dims: " << T.dimension(0) << ", " << T.dimension(1) << ", " 
  // << T.dimension(2) << ", " << T.dimension(3) << std::endl;
  //Te = Map<Matrix<double,9,6>>(T.data(), T.dimension(0), T.dimension(1));
  //
  //Tensor<double, 2> He = B_33s.contract(B_33s, double_dims);
  //std::cout << "T: \n" << T << std::endl;
  //std::cout << "Te: \n" << Te << std::endl;
  //std::cout << "He: \n" << He << std::endl;

  build_kkt_lhs();
  build_kkt_rhs();

}

void callback() {

  static int numPoints = 2000;
  static float param = 3.14;
  static bool simulate = false;
  static bool show_pinned = false;

  ImGui::PushItemWidth(100);

  // Curvature
  if (ImGui::Button("add curvature")) {
    //addCurvatureScalar();
  }

  ImGui::Checkbox("simulate",&simulate);
  if(ImGui::Button("show pinned")) {

    double min_x = meshV.col(0).minCoeff();
    double max_x = meshV.col(0).maxCoeff();
    double pin_x = min_x + (max_x-min_x)*0.3;
    pinnedV = (meshV.col(0).array() < pin_x).cast<int>(); 
    polyscope::getVolumeMesh("input mesh")
      ->addVertexScalarQuantity("pinned", pinnedV);

    std::vector<int> indices;
    for (int i = 0; i < pinnedV.size(); ++i) {
      if (pinnedV(i) == 1)
        indices.push_back(i);
    }

    // TODO don't we need one for tetrahedra too?
    // If all 4 vertices are projected out, we should probably
    // projec those tetrahedra out of things too
    sim::fixed_point_constraint_matrix(P,meshV,
        Eigen::Map<Eigen::VectorXi>(indices.data(),indices.size()));

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
