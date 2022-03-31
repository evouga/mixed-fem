#include "polyscope/polyscope.h"

// libigl
#include <igl/read_triangle_mesh.h>
#include <igl/boundary_facets.h>
#include <igl/remove_unreferenced.h>
#include <igl/invert_diag.h>
#include <igl/readOBJ.h>
#include <igl/readMESH.h>
#include <igl/writeOBJ.h>
#include <igl/doublearea.h>
#include <igl/svd3x3.h>
#include <igl/per_face_normals.h>
#include "svd/svd3x3_sse.h"

// Polyscope
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include "args/args.hxx"
#include "json/json.hpp"

// Bartels
#include <EigenTypes.h>
#include "linear_tri3dmesh_dphi_dX.h"

#include "arap.h"
#include "neohookean.h"
#include "corotational.h"
#include "preconditioner.h"
#include "tri_kkt.h"
#include "pinning_matrix.h"
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
MatrixXd meshV; // verts
MatrixXi meshF; // faces
MatrixXd meshN; // normals 

VectorXd dq_la;

// F = RS
std::vector<Matrix3d> R; // Per-element rotations
std::vector<Vector6d> S; // Per-element symmetric deformation
std::vector<Matrix6d> Hinv;

SparseMatrixd M;          // mass matrix
SparseMatrixd P;          // pinning constraint (for vertices)
SparseMatrixd P_kkt;      // pinning constraint (for kkt matrix)
SparseMatrixdRowMajor Jw; // integrated (weighted) jacobian
SparseMatrixdRowMajor J;  // jacobian
MatrixXd dphidX; 
VectorXi pinnedV;

// Simulation params
int solver_steps=5;
double h = 0.01;
double thickness = 1e-3;
double density = 100;
double ym = 1e5;
double pr = 0.40;
double mu = ym/(2.0*(1.0+pr));
double lambda = (ym*pr)/((1.0+pr)*(1.0-2.0*pr));
double alpha = mu;
double ih2 = 1.0/h/h;
double grav = -9.8;
double plane_d;
double beta = 100;

bool enable_slide = false;
bool enable_ext = true;
bool warm_start = true;
bool floor_collision = false;
bool export_sim = false;

Matrix<double, 6,1> I_vec;

// Configuration vectors & body forces
VectorXd qt;    // current positions
VectorXd q0;    // previous positions
VectorXd q1;    // previous^2 positions
VectorXd f_ext; // per-node external forces
VectorXd f_ext0;// per-node external forces (not integrated)
VectorXd la;    // lambdas
VectorXd b;     // coordinates projected out
VectorXd b0;     // coordinates projected out
VectorXd vols;  // per element volume
VectorXd dblA;  // per element volume

// KKT system
SparseMatrixd lhs;
SparseMatrixd lhs_sim;
#if defined(SIM_USE_CHOLMOD)
CholmodSimplicialLDLT<SparseMatrixd> solver;
//SimplicialLDLT<SparseMatrixd> solver;
#else
SimplicialLDLT<SparseMatrixd> solver;
#endif
ConjugateGradient<SparseMatrixd, Lower|Upper, FemPreconditioner<double>> cg;
VectorXd rhs;

double t_coll=0, t_asm = 0, t_precond=0, t_rhs = 0, t_solve = 0, t_SR = 0; 
// ------------------------------------ //
VectorXd collision_force() {

  //Vector3d N(plane(0),plane(1),plane(2));
  Vector3d N(.4,.2,.8);
  //Vector3d N(0.,1.,0.);
  N = N / N.norm();
  double d = plane_d;

  int n = qt.size() / 3;
  VectorXd ret(qt.size());
  ret.setZero();

  double k = 80; //20 for octopus ssliding

  //#pragma omp parallel for
  //for (int i = 0; i < n; ++i) {
  //  Vector3d xi(qt(3*i)+dq_la(3*i),
  //      qt(3*i+1)+dq_la(3*i+1),
  //      qt(3*i+2)+dq_la(3*i+2));
  //  double dist = xi.dot(N);
  //  if (dist < plane_d) {
  //    ret.segment(3*i,3) = k*(plane_d-dist)*N;
  //  }
  //}
  
  double min_y = meshV.col(1).minCoeff();
  double max_y = meshV.col(1).maxCoeff();
  double pin_y = min_y + (max_y-min_y)*0.05;
  VectorXi toforce = (meshV.col(1).array() < pin_y).cast<int>(); 
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    if (toforce(i)) {
      //ret.segment(3*i,3) = -1e2*N;
      ret.segment(3*i,3) = -3e2*N;
    }
  }

  return M*ret;
}

void build_kkt_lhs() {
 
  std::vector<Triplet<double>> trips, trips_sim;
  int sz = meshV.size() + meshF.rows()*9;
  tri_kkt_lhs(M, Jw, ih2, trips); 
  trips_sim = trips;

  // KKT preconditioner LHS
  diag_compliance(meshV, meshF, vols, alpha, trips);
  lhs.resize(sz,sz);
  lhs.setFromTriplets(trips.begin(), trips.end());
  lhs = P_kkt * lhs * P_kkt.transpose();

  corotational_compliance(meshV, meshF, R, vols, mu, lambda, trips_sim);
  lhs_sim.resize(sz,sz);
  lhs_sim.setFromTriplets(trips_sim.begin(), trips_sim.end());
  lhs_sim = P_kkt * lhs_sim * P_kkt.transpose();

  #if defined(SIM_USE_CHOLMOD)
  std::cout << "Using CHOLDMOD solver" << std::endl;
  #endif
  solver.compute(lhs);
  if(solver.info()!=Success) {
    std::cerr << " KKT prefactor failed! " << std::endl;
  } 
  cg.preconditioner().init(lhs);
  cg.setMaxIterations(10);
  cg.setTolerance(1e-7);
}

void build_kkt_rhs() {
  int sz = qt.size() + meshF.rows()*9;
  rhs.resize(sz);
  rhs.setZero();

  // Positional forces 
  rhs.segment(0, qt.size()) = ih2*M*(q0 - q1);
  if (enable_ext) {
    rhs.segment(0, qt.size()) += f_ext;
  }

  #pragma omp parallel for
  for (int i = 0; i < meshF.rows(); ++i) {
    // 1. W * st term +  - W * Hinv * g term
    Matrix3d NN = (meshN.row(i).transpose()) * meshN.row(i);
    Vector9d n = sim::flatten((R[i]*NN).transpose());
    //rhs.segment(qt.size() + 9*i, 9) = vols(i) * (arap_rhs(R[i]) - n);

    // For neohookean we need to compute Hinv
    Hinv[i] = neohookean_hinv(R[i],S[i],mu,lambda);
    rhs.segment(qt.size() + 9*i, 9) = vols(i) * (neohookean_rhs(R[i],
        S[i], Hinv[i], mu, lambda) - n);
  }

  // 3. Jacobian term
  rhs.segment(qt.size(), 9*meshF.rows()) -= Jw*(P.transpose()*qt+b);
}

void update_SR_fast() {
  VectorXd def_grad = J*(P.transpose()*(qt+dq_la.segment(0, qt.rows()))+b);
  int N = (meshF.rows() / 4) + int(meshF.rows() % 4 != 0);

  double fac = beta * (la.maxCoeff() + 1e-8);
  #pragma omp parallel for 
  for (int ii = 0; ii < N; ++ii) {

    Matrix<float,12,3> Y4,R4;

    for (int jj = 0; jj < 4; ++jj) {
      int i = ii*4 +jj;
      if (i >= meshF.rows())
        break;

      Vector9d li = la.segment(9*i,9)/fac + def_grad.segment(9*i,9);

      // 1. Update S[i] using new lambdas
      //S[i] += arap_ds(R[i], S[i], la.segment(9*i,9), mu, lambda);
      S[i] += neohookean_ds(R[i], S[i], la.segment(9*i,9), Hinv[i], mu, lambda);

      // 2. Solve rotation matrices
      //  la    B     C    R   s
      //  1x9 9x3x3 3x3x6 3x3 6x1 
      //  we want [la   B ]  [ C    s]
      //          1x9 9x3x3  3x3x6 6x1
      Matrix3d Cs;
      Cs << S[i](0), S[i](5), S[i](4), 
            S[i](5), S[i](1), S[i](3), 
            S[i](4), S[i](3), S[i](2); 
      Cs -= meshN.row(i).transpose()*meshN.row(i);
      Matrix3d y4 = Map<Matrix3d>(li.data()).transpose()*Cs;
      Y4.block(3*jj, 0, 3, 3) = y4.cast<float>();
    }
    // Solve rotations
    polar_svd3x3_sse(Y4,R4);
    for (int jj = 0; jj < 4; jj++) {
      int i = ii*4 +jj;
      if (i >= meshF.rows())
        break;
      R[i] = R4.block(3*jj,0,3,3).cast<double>();
    }
  }
}


void init_sim() {
  //rotate the mesh
  /*Matrix3d R_test;
  R_test << 0.707, -0.707, 0,
            0.707, 0.707, 0,
            0, 0, 1;*/

  I_vec << 1, 1, 1, 0, 0, 0; // Identity in symmetric format

  // Initialize rotation matrices to identity
  R.resize(meshF.rows());
  S.resize(meshF.rows());
  Hinv.resize(meshF.rows());
  for (int i = 0; i < meshF.rows(); ++i) {
    R[i].setIdentity();
    //R[i] = R_test;
    S[i] = I_vec;
    Hinv[i].setIdentity();
  }

  // Initial lambdas
  la.resize(9 * meshF.rows());
  la.setZero();

  // Mass matrix
  VectorXd densities = VectorXd::Constant(meshF.rows(), density);

  igl::doublearea(meshV, meshF, dblA);
  vols = dblA;
  vols.array() *= (thickness/2);

  M = trimesh_massmatrix(meshV,meshF,dblA);
  M = M*density*thickness; // note: assuming uniform density and thickness

  J = tri_jacobian(meshV,meshF,vols,false);
  Jw = tri_jacobian(meshV,meshF,vols,true);

  // Pinning matrices
  double min_x = meshV.col(0).minCoeff();
  double max_x = meshV.col(0).maxCoeff();
  double pin_x = min_x + (max_x-min_x)*0.3;
  double min_y = meshV.col(1).minCoeff();
  double max_y = meshV.col(1).maxCoeff();
  double pin_y = max_y - (max_y-min_y)*0.01;
  //double pin_y = min_y + (max_y-min_y)*0.1;
  //pinnedV = (meshV.col(0).array() < pin_x).cast<int>(); 
  pinnedV = (meshV.col(1).array() > pin_y).cast<int>(); 
  //pinnedV(0) = 1;
  polyscope::getSurfaceMesh("input mesh")
    ->addVertexScalarQuantity("pinned", pinnedV);
  P = pinning_matrix(meshV,meshF,pinnedV,false);
  P_kkt = pinning_matrix(meshV,meshF,pinnedV,true);
  
  MatrixXd tmp = meshV.transpose();
  //MatrixXd tmp = (R_test*meshV.transpose());
  //MatrixXd tmp = Rot*meshV.transpose();

  qt = Map<VectorXd>(tmp.data(), meshV.size());
  b = qt - P.transpose()*P*qt;
  b0 = b;
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
}

void simulation_step() {
  //
  dq_la.setZero();

  // Warm start solver
  if (warm_start) {
    dq_la.segment(0,qt.size()) = (qt-q0) + h*h*f_ext0;
    update_SR_fast();
  }
  
  //t_coll=0; t_asm = 0; t_precond=0; t_rhs = 0; t_solve = 0; t_SR = 0; 
  
  beta = 100;

  for (int i = 0; i < solver_steps; ++i) {
    auto start = high_resolution_clock::now();
    build_kkt_rhs();
    auto end = high_resolution_clock::now();
    t_rhs += duration_cast<nanoseconds>(end-start).count()/1e6;
    start = end;

    if (floor_collision) {
      VectorXd f_coll = collision_force();
      rhs.segment(0,qt.size()) += f_coll;
      end = high_resolution_clock::now();
      t_coll += duration_cast<nanoseconds>(end-start).count()/1e6;
      start = end;
    }

    // Temporary for benchmarking!
    VectorXd tmp(dq_la.size());
    start = high_resolution_clock::now();
    tmp = solver.solve(rhs);
    end = high_resolution_clock::now();
    t_precond += duration_cast<nanoseconds>(end-start).count()/1e6;
    start = end;

    if (i == 0) {
      dq_la = solver.solve(rhs);
    }
    //dq_la = tmp;

    start = high_resolution_clock::now();
    // CG solve
    update_neohookean_compliance(qt.size(), meshF.rows(), R, Hinv, vols,
        mu, lambda, lhs_sim);
    end = high_resolution_clock::now();
    t_asm += duration_cast<nanoseconds>(end-start).count()/1e6;
    start = end;

    cg.compute(lhs_sim);
    dq_la = cg.solveWithGuess(rhs, dq_la);
    end = high_resolution_clock::now();
    t_solve += duration_cast<nanoseconds>(end-start).count()/1e6;
    std::cout << "#iterations:     " << cg.iterations() << std::endl;
    std::cout << "estimated error: " << cg.error()      << std::endl;
    
    // Update per-element R & S matrices
    start = high_resolution_clock::now();
    la = dq_la.segment(qt.size(),9*meshF.rows());
    update_SR_fast();
    end = high_resolution_clock::now();
    t_SR += duration_cast<nanoseconds>(end-start).count()/1e6;
    beta *= std::min(mu, 1.5*beta);

  }
  //t_coll/=steps; t_rhs /=steps; t_solve/=steps; t_SR/=steps; t_asm/=steps;t_precond/=steps;
  //std::cout << "[Avg Time ms] collision: " << t_coll << 
  //  " rhs: " << t_rhs << " assembly: " << t_asm << " solver: " << t_solve <<
  //  " update S & R: " << t_SR << std::endl;

  q1 = q0;
  q0 = qt;
  qt += dq_la.segment(0, qt.size());

  static int total_steps = 0;
  double elapsed_time = total_steps*h;
  if (enable_slide) {
    double len = meshV.col(0).maxCoeff() - meshV.col(0).minCoeff();
    for (int i = 0; i < b.size()/3; ++i) {
      b(3*i+2) = b0(3*i+2) + 0.5*len*std::sin(elapsed_time*M_PI/4); 
    }
    total_steps++;
    //std::cout << "elapsed time: " << elapsed_time << std::endl;
  }

  // Initial configuration vectors (assuming 0 initial velocity)
  VectorXd q = P.transpose()*qt + b;
  MatrixXd tmp = Map<MatrixXd>(q.data(), meshV.cols(), meshV.rows());
  meshV = tmp.transpose();


  MatrixXd tmpN(meshN.rows(), 3);
  for (int i = 0; i < meshN.rows(); ++i) {
    Vector3d n = R[i]*meshN.row(i).transpose();
    tmpN.row(i) = n;
  }
  polyscope::getSurfaceMesh("input mesh")->
    addFaceVectorQuantity("normals", tmpN);
}

void callback() {

  static bool simulating = false;
  static bool show_pinned = false;

  ImGui::PushItemWidth(100);

  //ImGui::Checkbox("floor collision",&floor_collision);
  ImGui::Checkbox("force",&floor_collision);
  ImGui::Checkbox("warm start",&warm_start);
  ImGui::Checkbox("external forces",&enable_ext);
  ImGui::Checkbox("slide mesh",&enable_slide);
  ImGui::Checkbox("simulate",&simulating);
  ImGui::Checkbox("export",&export_sim);
  //if(ImGui::Button("show pinned")) {
  //} 
  static int step = 0;
  static int export_step = 0;

  if(ImGui::Button("sim step") || simulating) {
    simulation_step();
    ++step;
    polyscope::getSurfaceMesh("input mesh")
      ->updateVertexPositions(meshV);

    if (export_sim) {
      char buffer [50];
      int n = sprintf(buffer, "../data/cloth/sheet_soft2/%04d.png", export_step); 
      buffer[n] = 0;
      polyscope::screenshot(std::string(buffer), true);
      n = sprintf(buffer, "../data/cloth/sheet_soft2/%04d.obj", export_step++); 
      buffer[n] = 0;
      igl::writeOBJ(std::string(buffer),meshV,meshF);
    }

    std::cout << "STEP: " << step << std::endl;
    std::cout << "[Avg Time ms] " 
      << " collision: " << t_coll / solver_steps / step
      << " rhs: " << t_rhs / solver_steps / step
      << " preconditioner: " << t_precond / solver_steps / step
      << " KKT assembly: " << t_asm / solver_steps / step
      << " cg.solve(): " << t_solve / solver_steps / step
      << " update S & R: " << t_SR / solver_steps / step
      << std::endl;

  }
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
  //igl::readOBJ(filename, meshV, meshF);

  // Read tetmesh
  igl::read_triangle_mesh(filename,meshV,meshF);
  Eigen::MatrixXd NV;
  Eigen::MatrixXi NF;
  VectorXi VI,VJ;
  igl::remove_unreferenced(meshV,meshF,NV,NF,VI,VJ);
  meshV = NV;
  meshF = NF;

  meshV.array() /= meshV.maxCoeff();

  igl::per_face_normals(meshV,meshF,meshN);

  // Register the mesh with Polyscope
  polyscope::options::autocenterStructures = false;
  polyscope::registerSurfaceMesh("input mesh", meshV, meshF);

  meshN = -meshN;
  polyscope::getSurfaceMesh("input mesh")->
    addFaceVectorQuantity("normals", meshN);

  pinnedV.resize(meshV.rows());
  pinnedV.setZero();
  polyscope::getSurfaceMesh("input mesh")
    ->addVertexScalarQuantity("pinned", pinnedV);

  // Add the callback
  polyscope::state::userCallback = callback;

  // Initial simulation setup
  init_sim();

  // Show the gui
  polyscope::show();

  return 0;
}

