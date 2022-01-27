#include "polyscope/polyscope.h"

// libigl
#include <igl/unique_simplices.h>
#include <igl/cat.h>
#include <igl/remove_unreferenced.h>
#include <igl/boundary_facets.h>
#include <igl/invert_diag.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <igl/readMESH.h>
#include <igl/edges.h>
#include <igl/edge_lengths.h>
#include <igl/doublearea.h>
#include <igl/svd3x3.h>
#include <igl/per_face_normals.h>
#include "svd3x3_sse.h"

// Polyscope
#include "polyscope/messages.h"
#include "polyscope/curve_network.h"
#include "polyscope/surface_mesh.h"
#include "args/args.hxx"
#include "json/json.hpp"

// Bartels
#include <EigenTypes.h>
#include "linear_tri3dmesh_dphi_dX.h"

#include "arap.h"
#include "preconditioner.h"
#include "corotational.h"
#include "neohookean.h"
#include "tri_kkt.h"
#include "rod_kkt.h"
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
MatrixXd meshV;  // verts
MatrixXi meshE;  // edges
MatrixXi meshF;  // faces (for output only)
MatrixXd meshN;  // normals 
MatrixXd meshBN; // binormals 

VectorXd dq_la;

// F = RS
polyscope::CurveNetwork* curve = nullptr;
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
double h = 0.01;
double thickness = 1e-2;//1e-3;
double density = 10;
double ym = 1e7;
double pr = 0.45;
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
#else
SimplicialLDLT<SparseMatrixd> solver;
#endif
//ConjugateGradient<SparseMatrixd, Lower|Upper, FemPreconditioner<double>> cg;
BiCGSTAB<SparseMatrixd, FemPreconditioner<double>> cg;
VectorXd rhs;

int solver_steps=10;
double t_coll=0, t_asm = 0, t_precond=0, t_rhs = 0, t_solve = 0, t_SR = 0; 

// ------------------------------------ //
//
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
      ret.segment(3*i,3) = -1e2*N;
    }
  }

  return M*ret;
}

void build_kkt_lhs() {

  std::vector<Triplet<double>> trips, trips_sim;
  int sz = meshV.size() + meshE.rows()*9;
  tri_kkt_lhs(M, Jw, ih2, trips); 
  trips_sim = trips;

  diag_compliance(meshV, meshE, vols, alpha, trips);
  lhs.resize(sz,sz);
  lhs.setFromTriplets(trips.begin(), trips.end());
  lhs = P_kkt * lhs * P_kkt.transpose();

  corotational_compliance(meshV, meshE, R, vols, mu, lambda, trips_sim);
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
  int sz = qt.size() + meshE.rows()*9;
  rhs.resize(sz);
  rhs.setZero();

  // Positional forces 
  rhs.segment(0, qt.size()) = ih2*M*(q0 - q1);
  if (enable_ext) {
    rhs.segment(0, qt.size()) += f_ext;
  }

  #pragma omp parallel for
  for (int i = 0; i < meshE.rows(); ++i) {
    // 1. W * st term +  - W * Hinv * g term
    Matrix3d NN = (meshN.row(i).transpose()) * meshN.row(i);
    Matrix3d BNBN = (meshBN.row(i).transpose()) * meshBN.row(i);
    Vector9d n = sim::flatten((R[i]*NN).transpose());
    Vector9d bn = sim::flatten((R[i]*BNBN).transpose());
    //rhs.segment(qt.size() + 9*i, 9) = vols(i) * (arap_rhs(R[i]) - n - bn);

    // For neohookean we need to compute Hinv
    Hinv[i] = neohookean_hinv(R[i],S[i],mu,lambda);
    rhs.segment(qt.size() + 9*i, 9) = vols(i) * (neohookean_rhs(R[i],
        S[i], Hinv[i], mu, lambda) - n - bn);
  }

  // 3. Jacobian term
  rhs.segment(qt.size(), 9*meshE.rows()) -= Jw*(P.transpose()*qt+b);
}

void update_SR_fast() {

  VectorXd def_grad = J*(P.transpose()*(qt+dq_la.segment(0, qt.rows()))+b);

  int N = (meshE.rows() / 4) + int(meshE.rows() % 4 != 0);

  double fac = beta * (la.maxCoeff() + 1e-5);
  #pragma omp parallel for 
  for (int ii = 0; ii < N; ++ii) {

    Matrix<float,12,3> Y4,R4;
    Y4.setZero();
    R4.setZero();

    for (int jj = 0; jj < 4; ++jj) {
      int i = ii*4 +jj;
      if (i >= meshE.rows())
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
      Cs -= meshBN.row(i).transpose()*meshBN.row(i);
      Matrix3d y4 = Map<Matrix3d>(li.data()).transpose()*Cs;
      Y4.block(3*jj, 0, 3, 3) = y4.cast<float>();
    }
    // Solve rotations
    polar_svd3x3_sse(Y4,R4);
    for (int jj = 0; jj < 4; jj++) {
      int i = ii*4 +jj;
      if (i >= meshE.rows())
        break;
      R[i] = R4.block(3*jj,0,3,3).cast<double>();
    }
  }
}


void init_sim() {
  I_vec << 1, 1, 1, 0, 0, 0; // Identity in symmetric format

  // Initialize rotation matrices to identity
  R.resize(meshE.rows());
  S.resize(meshE.rows());
  Hinv.resize(meshE.rows());
  for (int i = 0; i < meshE.rows(); ++i) {
    R[i].setIdentity();
    //R[i] = R_test;
    S[i] = I_vec;
    Hinv[i].setIdentity();
  }

  // Initial lambdas
  la.resize(9 * meshE.rows());
  la.setZero();

  // Mass matrix
  // - assuming uniform density and thickness
  vols.resize(meshE.rows());
  VectorXd dvols(meshE.rows());
  VectorXd densities(meshE.rows());
  densities.array() = density;
  densities(densities.size()-1) = 1e8;
  for (int i = 0; i < meshE.rows(); ++i) {
    vols(i) = (meshV.row(meshE(i,0)) - meshV.row(meshE(i,1))).norm() * thickness;
    dvols(i) = vols(i) * thickness*densities(i) ;
  }
  M = rod_massmatrix(meshV, meshE, vols);
  M = M*density;

  J = rod_jacobian(meshV,meshE,vols,false);
  Jw = rod_jacobian(meshV,meshE,vols,true);

  // Pinning matrices
  double min_x = meshV.col(0).minCoeff();
  double max_x = meshV.col(0).maxCoeff();
  double pin_x = min_x + (max_x-min_x)*0.01;
  double min_y = meshV.col(1).minCoeff();
  double max_y = meshV.col(1).maxCoeff();
  double pin_y = max_y - (max_y-min_y)*0.1;
  //double pin_y = min_y + (max_y-min_y)*0.1;
  //pinnedV = (meshV.col(0).array() < pin_x).cast<int>(); 
  //pinnedV = (meshV.col(1).array() > pin_y).cast<int>(); 
  pinnedV(0) = 1;
  curve->addNodeScalarQuantity("pinned", pinnedV);
  P = pinning_matrix(meshV,meshE,pinnedV,false);
  P_kkt = pinning_matrix(meshV,meshE,pinnedV,true);
  
  MatrixXd tmp = meshV.transpose();
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
      t_coll += duration_cast<nanoseconds>(end-start).count()/1e6;
      rhs.segment(0,qt.size()) += f_coll;
      end = high_resolution_clock::now();
      t_coll += duration_cast<nanoseconds>(end-start).count()/1e6;
      start = end;
    }
    //dq_la = solver.solve(rhs);
    if (i == 0) {
      dq_la = solver.solve(rhs);
    }

    // Temporary for benchmarking!
    VectorXd tmp(dq_la.size());
    start = high_resolution_clock::now();
    tmp = solver.solve(rhs);
    end = high_resolution_clock::now();
    t_precond += duration_cast<nanoseconds>(end-start).count()/1e6;
    start = end;

    dq_la = tmp;
    // New CG stuff
    //start = high_resolution_clock::now();
    //// CG solve
    //update_neohookean_compliance(qt.size(), meshE.rows(), R, Hinv, vols,
    //    mu, lambda, lhs_sim);
    //end = high_resolution_clock::now();
    //t_asm += duration_cast<nanoseconds>(end-start).count()/1e6;
    //start = end;

    //cg.compute(lhs_sim);
    //dq_la = cg.solveWithGuess(rhs, dq_la);
    //end = high_resolution_clock::now();
    //t_solve += duration_cast<nanoseconds>(end-start).count()/1e6;
    //std::cout << "#iterations:     " << cg.iterations() << std::endl;
    //std::cout << "estimated error: " << cg.error()      << std::endl;
    
    // Update per-element R & S matrices
    start = high_resolution_clock::now();
    la = dq_la.segment(qt.size(),9*meshE.rows());
    
    update_SR_fast();
    end = high_resolution_clock::now();
    t_SR += duration_cast<nanoseconds>(end-start).count()/1e6;
    beta *= std::min(mu, 1.5*beta);

    //std::cout << "dq_la:\n" << dq_la << std::endl;
    //for (int i =0; i < R.size(); ++i ) {
    //  std::cout << "R" <<i << std::endl << R[i] << std::endl;
    //}
  }
  //t_coll/=steps; t_rhs /=steps; t_solve/=steps; t_SR/=steps;
  //std::cout << "[Avg Time ms] collision: " << t_coll << 
  //  " rhs: " << t_rhs << " solver: " << t_solve <<
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


  //MatrixXd tmpN(meshN.rows(), 3);
  //for (int i = 0; i < meshN.rows(); ++i) {
  //  Vector3d n = R[i]*meshN.row(i).transpose();
  //  tmpN.row(i) = n;
  //}
  //polyscope::getSurfaceMesh("input mesh")->
  //  addFaceVectorQuantity("normals", tmpN);
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
    curve->updateNodePositions(meshV);

    if (export_sim) {
      char buffer [50];
      int n = sprintf(buffer, "../data/rods/rod_%04d.png", export_step); 
      buffer[n] = 0;
      polyscope::screenshot(std::string(buffer), true);
      n = sprintf(buffer, "../data/rods/rod_%04d.obj", export_step++); 
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

  // Simple Rod
  //meshV.resize(2,3);
  //meshE.resize(1,2);
  //meshN.resize(2,3);
  //meshBN.resize(2,3);
  //meshV << 
  //  0.0, 0.5, 0.0,
  //  0.5, 0.5, 0.0;
  //meshE <<
  //  0, 1;

  //// Normals and binormals
  //meshN <<
  //  0, 1, 0,
  //  0, 1, 0;
  //meshBN <<
  //  0, 0, 1,
  //  0, 0, 1;

  // Double Rod
  meshV.resize(3,3);
  meshE.resize(2,2);
  meshN.resize(3,3);
  meshBN.resize(3,3);
  meshV << 
    0.0, 0.5, 0.0,
    0.25, 0.5, 0.0,
    0.5, 0.5, 0.0;
  meshE <<
    0, 1,
    1, 2;

  // Normals and binormals
  meshN <<
    0, 1, 0,
    0, 1, 0,
    0, 1, 0;
  meshBN <<
    0, 0, 1,
    0, 0, 1,
    0, 0, 1;
  int n=10;
  meshV.resize(n,3);
  meshE.resize(n-1,2);
  meshN.resize(n,3);
  meshBN.resize(n,3);
  for (int i = 0; i < n; ++i) {
    meshV.row(i) = Vector3d(i/double(n-1),0.5,0.0); 
    if (i < n-1)
      meshE.row(i) = Vector2i(i,i+1); 
    meshN.row(i) = Vector3d(0,1,0);
    meshBN.row(i) = Vector3d(0,0,1);
  }
  std::cout << "meshV: " << meshV << std::endl;

  if (filename != "") {
    std::cout << "loading: " << filename << std::endl;
    MatrixXi meshT;
    igl::readOBJ(filename,meshV,meshF);

    //igl::readMESH(filename, meshV, meshT, meshF);
    // Stupid shit so that i can export a OBJ
    //MatrixXi F1,F2,F3,F4;
    //F1.resize(meshT.rows(),3);
    //F2.resize(meshT.rows(),3);
    //F3.resize(meshT.rows(),3);
    //F4.resize(meshT.rows(),3);
    //F1.col(0) = meshT.col(0); F1.col(1) = meshT.col(1); F1.col(2) = meshT.col(2);
    //F2.col(0) = meshT.col(0); F2.col(1) = meshT.col(2); F2.col(2) = meshT.col(3);
    //F3.col(0) = meshT.col(0); F3.col(1) = meshT.col(1); F3.col(2) = meshT.col(3);
    //F4.col(0) = meshT.col(2); F4.col(1) = meshT.col(1); F4.col(2) = meshT.col(3);
    //igl::cat(1,F1,F2,meshF);
    //igl::cat(1,meshF,F3,F1);
    //igl::cat(1,F1,F4,meshF);
    //igl::unique_simplices(meshF,F1);
    //meshF = F1;
    //Eigen::MatrixXd NV;
    //Eigen::MatrixXi NF;
    //VectorXi VI,VJ;
    //igl::remove_unreferenced(meshV,meshF,NV,NF,VI,VJ);
    //meshV = NV;
    //meshF = NF;
    //igl::edges(meshF,meshE);

    
    // Using triangle surf edges as rods
    //igl::boundary_facets(meshT, meshF);
    //meshF = meshF.rowwise().reverse().eval();
    //Eigen::MatrixXd NV;
    //Eigen::MatrixXi NF;
    //VectorXi VI,VJ;
    //igl::remove_unreferenced(meshV,meshF,NV,NF,VI,VJ);
    //meshV = NV;
    //meshF = NF;
    igl::edges(meshF,meshE);

    // Using tet edges as rods
    //igl::edges(meshT,meshE);
    
    MatrixXd FN;
    meshN.resize(meshE.rows(),3);
    meshBN.resize(meshE.rows(),3);
    for (int i = 0; i < meshE.rows(); ++i ) {
      Vector3d diff = (meshV.row(meshE(i,0)) - meshV.row(meshE(i,1)));
      diff /= diff.norm();
      Vector3d tmp(diff(0)-1.0,diff(2),diff(1));
      tmp /= tmp.norm();
      Vector3d N = diff.cross(tmp).normalized();
      Vector3d BN = diff.cross(N).normalized();
      meshN.row(i) = N;
      meshBN.row(i) = BN;
    }
  }

  meshV.array() /= meshV.maxCoeff();

  // Register the mesh with Polyscope
  polyscope::options::autocenterStructures = false;
  curve = polyscope::registerCurveNetwork("input network", meshV, meshE);

  pinnedV.resize(meshV.rows());
  pinnedV.setZero();
  curve->addNodeScalarQuantity("pinned", pinnedV);

  // Add the callback
  polyscope::state::userCallback = callback;
  //
  // Adjust ground plane
  VectorXd a = meshV.colwise().minCoeff();
  VectorXd b = meshV.colwise().maxCoeff();
  a(1) -= 1.0;
  b(1) += 0.5;
  polyscope::options::automaticallyComputeSceneExtents = false;
  polyscope::state::lengthScale = 1.;
  polyscope::state::boundingBox = std::tuple<glm::vec3, glm::vec3>{
    {a(0),a(1),a(2)},{b(0),b(1),b(2)}};

  std::cout << "V : " << meshV.rows() << " E: " << meshE.rows() << std::endl;

  // Initial simulation setup
  init_sim();

  // Show the gui
  polyscope::show();

  return 0;
}

