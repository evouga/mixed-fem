#include "polyscope/polyscope.h"

// libigl
#include <igl/boundary_facets.h>
#include <igl/invert_diag.h>
#include <igl/readMESH.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/volume.h>
#include <igl/AABB.h>
#include <igl/in_element.h>
#include <igl/barycentric_coordinates.h>
//#include "eigen_svd.h"

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

//eigen unsupported I/O
#include <eigen3/unsupported/Eigen/src/SparseExtra/MarketIO.h>

// #include "preconditioner.h"
// #include "corotational.h"
//#include "neohookean.h"
// #include "arap.h"
#include "svd/svd3x3_sse.h"
#include "pinning_matrix.h"
#include "tet_kkt.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/IterativeSolvers>
#include <iostream>
#include <unordered_set>
#include <utility>
#include <pcg.h>
#include "kkt.h"

#include "objects/tet_object.h"
#include "materials/neohookean_model.h"
#include "config.h"

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
MatrixXd meshV, meshV0, skinV;
MatrixXi meshF, skinF;
MatrixXi meshT; // tetrahedra
SparseMatrixd lbs; // linear blend skinning matrix
VectorXi pinnedV;

// Simulation params
double h = 0.034;//0.1;
double density = 1000.0;
double ym = 1e5;
//double ym = 1e5;
double pr = 0.45;
double mu = ym/(2.0*(1.0+pr));
double lambda = (ym*pr)/((1.0+pr)*(1.0-2.0*pr));

double ih2 = 1.0/h/h;
double grav = -9.8;
double plane_d;
double beta = 5.;
double ibeta = 1./beta;

bool floor_collision = true;
bool export_sim = false;
bool warm_start = true;


double t_coll=0, t_asm = 0, t_precond=0, t_rhs = 0, t_solve = 0, t_SR = 0; 
int outer_steps = 2;
int inner_steps = 7;

using namespace mfem;
std::shared_ptr<MaterialModel> material;
std::shared_ptr<MaterialConfig> material_config;
std::shared_ptr<SimObject> tet_object;

// ------------------------------------ //

// void simulation_step() {

//   dq_la.setZero();

//   // Warm start solver
//   if (warm_start) {
//     dq_la.segment(0,qt.size()) = (qt-q0) + h*h*f_ext0;
//     update_SR();
//   }
  
//   for (int i = 0; i < outer_steps; ++i) {
//     ibeta = 1./beta;
//     energy_grad();
//     for (int j = 0; j < inner_steps; ++j) {

//       auto start = high_resolution_clock::now();
//       build_kkt_rhs();
//       auto end = high_resolution_clock::now();
//       t_rhs += duration_cast<nanoseconds>(end-start).count()/1e6;
//       start = end;

//       if (floor_collision) {
//         VectorXd f_coll = collision_force();
//         rhs.segment(0,qt.size()) += f_coll;
//         end = high_resolution_clock::now();
//         t_coll += duration_cast<nanoseconds>(end-start).count()/1e6;
//         start = end;
//       }

//       // Temporary for benchmarking!
//       VectorXd tmp(dq_la.size());
//       start = high_resolution_clock::now();
//       tmp = solver.solve(rhs);
//       end = high_resolution_clock::now();
//       t_precond += duration_cast<nanoseconds>(end-start).count()/1e6;
//       start = end;

//       if (i == 0 && j == 0) {
//         dq_la = solver.solve(rhs);
//       }
//       start=end;

//       start = high_resolution_clock::now();
//       // New CG stuff
//       //update_arap_compliance(qt.size(), meshT.rows(), R, vols,
//       //    mu, lambda, lhs_sim);
//       //update_corotational_compliance(qt.size(), meshT.rows(), R, vols,
//       //    mu, lambda, lhs_sim);
//       material->update_compliance(qt.size(), meshT.rows(), R, Hinv, vols,
//           lhs_sim);
//       end = high_resolution_clock::now();
//       t_asm += duration_cast<nanoseconds>(end-start).count()/1e6;
//       start = end;

//       pcg(dq_la, lhs_sim, rhs, tmp_r, tmp_z, tmp_p, tmp_Ap, solver);
//       end = high_resolution_clock::now();
//       t_solve += duration_cast<nanoseconds>(end-start).count()/1e6;
      
//       // Update per-element R & S matrices
//       start = high_resolution_clock::now();
//       dq = dq_la.segment(0,qt.size());
//       la = dq_la.segment(qt.size(),9*meshT.rows());
//       update_SR();

//       end = high_resolution_clock::now();
//       t_SR += duration_cast<nanoseconds>(end-start).count()/1e6;
//       ibeta = std::min(1e-8, 0.9*ibeta);
//     }
//   }

//   q1 = q0; q0 = qt;
//   qt += dq;

//   // Initial configuration vectors (assuming 0 initial velocity)
//   VectorXd q = P.transpose()*qt + b;
//   MatrixXd tmp = Map<MatrixXd>(q.data(), meshV.cols(), meshV.rows());
//   meshV = tmp.transpose();

//   // if skin enabled too
//   if (skinV.rows() > 0) {
//     skinV.col(0) = lbs * meshV.col(0);
//     skinV.col(1) = lbs * meshV.col(1);
//     skinV.col(2) = lbs * meshV.col(2);
//   }

// }

void callback() {

  static bool simulating = false;
  static bool show_pinned = false;

  ImGui::PushItemWidth(100);

  ImGui::Checkbox("floor collision",&floor_collision);
  ImGui::Checkbox("warm start",&warm_start);
  ImGui::Checkbox("simulate",&simulating);
  ImGui::Checkbox("export",&export_sim);
  static int step = 0;
  static int export_step = 0;

  if(ImGui::Button("sim step") || simulating) {
    for(unsigned int ii=0; ii<3; ++ii) {
      // simulation_step();
    }
    ++step;
    //polyscope::getVolumeMesh("input mesh")
    polyscope::getSurfaceMesh("input mesh")
      ->updateVertexPositions(meshV);

    polyscope::SurfaceMesh* skin_mesh;
    if ((skin_mesh = polyscope::getSurfaceMesh("skin mesh")) &&
        skin_mesh->isEnabled()) {
      skin_mesh->updateVertexPositions(skinV);
    }

    if (export_sim) {
      char buffer [50];
      int n = sprintf(buffer, "../data/tet_%04d.png", export_step); 
      buffer[n] = 0;
      polyscope::screenshot(std::string(buffer), true);
      n = sprintf(buffer, "../data/tet_%04d.obj", export_step++); 
      buffer[n] = 0;
      if (skinV.rows() > 0)
        igl::writeOBJ(std::string(buffer),skinV,skinF);
      else
        igl::writeOBJ(std::string(buffer),meshV,meshF);
    }

    std::cout << "STEP: " << step << std::endl;
    std::cout << "[Avg Time ms] " 
      << " collision: " << t_coll / outer_steps / step
      << " rhs: " << t_rhs / outer_steps / step
      << " preconditioner: " << t_precond / outer_steps / step
      << " KKT assembly: " << t_asm / outer_steps / step
      << " cg.solve(): " << t_solve / outer_steps / step
      << " update S & R: " << t_SR / outer_steps / step
      << std::endl;

  }
  if (step == 570) simulating=false;
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

  std::cout << "V : " << meshV.rows() << " T: " << meshT.rows() << std::endl;
  meshV0 = meshV;

  material_config = std::make_shared<MaterialConfig>();
  material_config->mu = mu;
  material_config->la = lambda;

  material = std::make_shared<StableNeohookean>(material_config);

  // Initial simulation setup
  // init_sim();

  tet_object = std::make_shared<TetrahedralObject>(meshV, meshT,
      material, material_config);
  tet_object->init();

  // Show the gui
  polyscope::show();

  return 0;
}

