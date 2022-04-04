#include "polyscope/polyscope.h"

// libigl
#include <igl/boundary_facets.h>
#include <igl/readMESH.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
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

#include "simulator.h"
#include "objects/tet_object.h"
#include "materials/material_model.h"

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

bool floor_collision = true;
bool export_sim = false;
bool warm_start = true;


double t_coll=0, t_asm = 0, t_precond=0, t_rhs = 0, t_solve = 0, t_SR = 0; 

using namespace mfem;
std::shared_ptr<SimConfig> config;
std::shared_ptr<MaterialModel> material;
std::shared_ptr<MaterialConfig> material_config;
std::shared_ptr<SimObject> tet_object;

// ------------------------------------ //

void simulation_step() {

  Simulator sim(tet_object, config);
  sim.step();
  meshV = tet_object->vertices();

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
  ImGui::Checkbox("export",&export_sim);
  static int step = 0;
  static int export_step = 0;

  if(ImGui::Button("sim step") || simulating) {
    for(unsigned int ii=0; ii<3; ++ii) {
      simulation_step();
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

    // std::cout << "STEP: " << step << std::endl;
    // std::cout << "[Avg Time ms] " 
    //   << " collision: " << t_coll / outer_steps / step
    //   << " rhs: " << t_rhs / outer_steps / step
    //   << " preconditioner: " << t_precond / outer_steps / step
    //   << " KKT assembly: " << t_asm / outer_steps / step
    //   << " cg.solve(): " << t_solve / outer_steps / step
    //   << " update S & R: " << t_SR / outer_steps / step
    //   << std::endl;

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
  polyscope::options::automaticallyComputeSceneExtents = false;
  polyscope::state::lengthScale = 1.;
  polyscope::state::boundingBox = std::tuple<glm::vec3, glm::vec3>{
    {a(0),a(1),a(2)},{b(0),b(1),b(2)}};

  std::cout << "V : " << meshV.rows() << " T: " << meshT.rows() << std::endl;
  meshV0 = meshV;

  // Initial simulation setup
  config = std::make_shared<SimConfig>();
  config->plane_d = a(1);
  config->inner_steps=2;
  material_config = std::make_shared<MaterialConfig>();
  material = std::make_shared<StableNeohookean>(material_config);
  tet_object = std::make_shared<TetrahedralObject>(meshV, meshT,
      config, material, material_config);
  tet_object->init();

  // Show the gui
  polyscope::show();

  return 0;
}

