#include "polyscope/polyscope.h"

// libigl
#include <igl/boundary_facets.h>
#include <igl/invert_diag.h>
#include <igl/readMESH.h>
#include <igl/readOBJ.h>
#include <igl/volume.h>
#include <igl/AABB.h>
#include <igl/in_element.h>
#include <igl/barycentric_coordinates.h>


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

#include "corotational.h"
#include "arap.h"
#include "svd3x3_sse.h"
#include "pinning_matrix.h"
#include "tet_kkt.h"
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
MatrixXd meshV, skinV;
MatrixXi meshF, skinF;
MatrixXi meshT; // tetrahedra
SparseMatrixd lbs; // linear blend skinning matrix

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
double lambda = (ym*pr)/((1.0+pr)*(1.0-2.0*pr));

double alpha = mu;
double ih2 = 1.0/h/h;
double grav = -9.8;
double plane_d;
double beta = 100;

bool warm_start = true;
bool floor_collision = true;

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
std::vector<Triplet<double>> lhs_trips;
SparseMatrixd lhs;
#if defined(SIM_USE_CHOLMOD)
CholmodSimplicialLDLT<SparseMatrixd> solver;
#else
SimplicialLDLT<SparseMatrixd> solver;
#endif
ConjugateGradient<SparseMatrixd> cg;
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

  std::vector<Triplet<double>> trips;

  int sz = meshV.size() + meshT.rows()*9;
  tet_kkt_lhs(M, Jw, ih2, trips); 

  lhs_trips = trips;

  arap_compliance(meshV, meshT, vols, alpha, trips);
  //corotational_compliance(meshV, meshT, R, vols, mu, lambda, trips);

  lhs.resize(sz,sz);
  lhs.setFromTriplets(trips.begin(), trips.end());
  //std::cout << "TRIPS: \n" << lhs << std::endl;
  lhs = P_kkt * lhs * P_kkt.transpose();

  #if defined(SIM_USE_CHOLMOD)
  std::cout << "Using CHOLDMOD solver" << std::endl;
  #endif
  solver.compute(lhs);
  if(solver.info()!=Success) {
    std::cerr << " KKT prefactor failed! " << std::endl;
  }

  cg.compute(lhs);
  if (cg.info() != Success) {
    std::cerr << " CG KKT prefactor failed! " << std::endl;
  }
}

void build_kkt_rhs() {
  int sz = qt.size() + meshT.rows()*9;
  rhs.resize(sz);
  rhs.setZero();

  // Positional forces 
  rhs.segment(0, qt.size()) = f_ext + ih2*M*(q0 - q1);

  // Lagrange multiplier forces
  #pragma omp parallel for
  for (int i = 0; i < meshT.rows(); ++i) {
    // 1. W * st term +  - W * Hinv * g term
    //rhs.segment(qt.size() + 9*i, 9) = vols(i) * arap_rhs(R[i]);
    rhs.segment(qt.size() + 9*i, 9) = vols(i) * corotational_rhs(R[i],
        S[i], mu, lambda);
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

  double t_1=0,t_2=0,t_3=0,t_4=0,t_5=0; 
  auto start = high_resolution_clock::now();
  VectorXd def_grad = J*(P.transpose()*(qt+dq_la.segment(0, qt.rows()))+b);
  auto end = high_resolution_clock::now();
  t_1 = duration_cast<nanoseconds>(end-start).count()/1e6;

  int N = (meshT.rows() / 4) + int(meshT.rows() % 4 != 0);

  double fac = beta * (la.maxCoeff() + 1e-8);

  #pragma omp parallel for 
  for (int ii = 0; ii < N; ++ii) {

    Matrix<float,12,3> Y4,R4;

    for (int jj = 0; jj < 4; ++jj) {
      int i = ii*4 +jj;
      if (i >= meshT.rows())
        break;
      Vector9d li = la.segment(9*i,9)/fac + def_grad.segment(9*i,9);

      // 1. Update S[i] using new lambdas
      start = high_resolution_clock::now();
      //S[i] += arap_ds(R[i], S[i], la.segment(9*i,9), mu, lambda);
      S[i] += corotational_ds(R[i], S[i], la.segment(9*i,9), mu, lambda);
      end = high_resolution_clock::now();
      t_3 += duration_cast<nanoseconds>(end-start).count()/1e6;

      // 2. Solve rotation matrices
      start = high_resolution_clock::now();

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
  sim::linear_tetmesh_mass_matrix(M, meshV, meshT, densities, vols);

  J = tet_jacobian(meshV,meshT,vols,false);
  Jw = tet_jacobian(meshV,meshT,vols,true);

  // Pinning matrices
  double min_x = meshV.col(0).minCoeff();
  double max_x = meshV.col(0).maxCoeff();
  double pin_x = min_x + (max_x-min_x)*0.3;
  double min_y = meshV.col(1).minCoeff();
  double max_y = meshV.col(1).maxCoeff();
  double pin_y = max_y - (max_y-min_y)*0.1;
  //double pin_y = min_y + (max_y-min_y)*0.1;
  //pinnedV = (meshV.col(0).array() < pin_x).cast<int>(); 
  pinnedV = (meshV.col(1).array() > pin_y).cast<int>(); 
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

  build_kkt_lhs();

  // Project out mass matrix pinned point
  M = P * M * P.transpose();

  // External gravity force
  f_ext = M * P *Vector3d(0,grav,0).replicate(meshV.rows(),1);
  f_ext0 = P *Vector3d(0,grav,0).replicate(meshV.rows(),1);
  //EigenSolver<MatrixXd> eigensolver;
  //eigensolver.compute(MatrixXd(lhs));
  //std::cout << "Evals: \n" << eigensolver.eigenvalues().real() << std::endl;
  //std::cout << "LHS norm: " << lhs.norm() << std::endl;
}

void simulation_step() {

  int steps=10;
  dq_la.setZero();

  // Warm start solver
  if (warm_start) {
    dq_la.segment(0,qt.size()) = (qt-q0) + h*h*f_ext0;
    update_SR_fast();
  }
  
  double t_coll=0, t_rhs = 0, t_solve = 0, t_SR = 0; 
  beta = 100;
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

    // New CG stuff
    //std::vector<Triplet<double>> trips;
    //int sz = meshV.size() + meshT.rows()*9;
    //trips = lhs_trips;
    ////arap_compliance(meshV, meshT, vols, alpha, trips);
    ////corotational_compliance(meshV, meshT, R, vols, mu, lambda, trips);
    //arap_compliance2(meshV, meshT, R, vols, mu, lambda, trips);
    //SparseMatrixd newlhs(sz,sz);
    //newlhs.setFromTriplets(trips.begin(), trips.end());
    //std::cout << " NEW LHS: " << newlhs << std::endl;
    //newlhs = P_kkt * newlhs * P_kkt.transpose();
    //cg.compute(newlhs);
    //dq_la = cg.solve(rhs);
    //std::cout << "#iterations:     " << cg.iterations() << std::endl;
    //std::cout << "estimated error: " << cg.error()      << std::endl;
    //////////////////

    end = high_resolution_clock::now();
    t_solve += duration_cast<nanoseconds>(end-start).count()/1e6;
    start=end;
    la = dq_la.segment(qt.size(),9*meshT.rows());
    
    // Update per-element R & S matrices
    update_SR_fast();
    end = high_resolution_clock::now();
    t_SR += duration_cast<nanoseconds>(end-start).count()/1e6;
    beta = std::min(mu, beta*1.5);

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

  // if skin enabled too
  if (skinV.rows() > 0) {
    skinV.col(0) = lbs * meshV.col(0);
    skinV.col(1) = lbs * meshV.col(1);
    skinV.col(2) = lbs * meshV.col(2);
  }

}

void callback() {

  static bool simulating = false;
  static bool show_pinned = false;

  ImGui::PushItemWidth(100);

  ImGui::Checkbox("floor collision",&floor_collision);
  ImGui::Checkbox("warm start",&warm_start);
  ImGui::Checkbox("simulate",&simulating);

  if(ImGui::Button("sim step") || simulating) {
    simulation_step();
    //polyscope::getVolumeMesh("input mesh")
    polyscope::getSurfaceMesh("input mesh")
      ->updateVertexPositions(meshV);

    polyscope::SurfaceMesh* skin_mesh;
    if ((skin_mesh = polyscope::getSurfaceMesh("skin mesh")) &&
        skin_mesh->isEnabled()) {
      skin_mesh->updateVertexPositions(skinV);
    }

  }
  //ImGui::SameLine();
  //ImGui::InputInt("source vertex", &iVertexSource);

  ImGui::PopItemWidth();
}

int main(int argc, char **argv) {
  // Configure the argument parser
  args::ArgumentParser parser("Mixed FEM");
  args::Positional<std::string> inFile(parser, "mesh", "input mesh");
  args::Positional<std::string> inSurf(parser, "hires mesh", "hires surface");

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
  double fac = meshV.maxCoeff();
  meshV.array() /= fac;

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

  // Check if skinning mesh is provided
  if (inSurf) {
    std::string hiresFn = args::get(inSurf);
    std::cout << "Reading in skinning mesh: " << hiresFn << std::endl;
    igl::readOBJ(hiresFn, skinV, skinF);
    skinV.array() /= fac;
    //polyscope::SurfaceMesh* skin_mesh;
    //skin_mesh = polyscope::registerSurfaceMesh("skin mesh", skinV, skinF);
    //skin_mesh->setEnabled(false);

    // Create skinning transforms
    igl::AABB<MatrixXd,3> aabb;
    aabb.init(meshV,meshT);
    VectorXi I;
    in_element(meshV,meshT,skinV,aabb,I);

    std::vector<Triplet<double>> trips;
    for (int i = 0; i < I.rows(); ++i) {
      if (I(i) >= 0) {
        RowVector4d out;
        igl::barycentric_coordinates(skinV.row(i),
            meshV.row(meshT(I(i),0)),
            meshV.row(meshT(I(i),1)),
            meshV.row(meshT(I(i),2)),
            meshV.row(meshT(I(i),3)), out);
        trips.push_back(Triplet<double>(i, meshT(I(i),0), out(0)));
        trips.push_back(Triplet<double>(i, meshT(I(i),1), out(1)));
        trips.push_back(Triplet<double>(i, meshT(I(i),2), out(2)));
        trips.push_back(Triplet<double>(i, meshT(I(i),3), out(3)));
      }
    }
    lbs.resize(skinV.rows(),meshV.rows());
    lbs.setFromTriplets(trips.begin(),trips.end());
  }
  polyscope::SurfaceMesh* skin_mesh;
  skin_mesh = polyscope::registerSurfaceMesh("skin mesh", skinV, skinF);
  skin_mesh->setEnabled(false);

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

