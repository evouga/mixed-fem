#include "polyscope/polyscope.h"

// libigl
#include <igl/unique_simplices.h>
#include <igl/cat.h>
#include <igl/remove_unreferenced.h>
#include <igl/boundary_facets.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <igl/readMESH.h>
#include <igl/edges.h>
#include <igl/per_face_normals.h>

// Polyscope
#include "polyscope/messages.h"
#include "polyscope/curve_network.h"
#include "polyscope/surface_mesh.h"
#include "args/args.hxx"
#include "json/json.hpp"

#include "simulator.h"
#include "objects/simulation_object.h"
#include "materials/material_model.h"

using namespace Eigen;

// The mesh, Eigen representation
MatrixXd meshV;  // verts
MatrixXi meshE;  // edges
MatrixXi meshF;  // faces (for output only)
MatrixXd meshN;  // normals 
MatrixXd meshBN; // binormals 
VectorXi pinnedV;

polyscope::CurveNetwork* curve = nullptr;

bool enable_slide = false;
bool enable_ext = true;
bool warm_start = true;
bool floor_collision = false;
bool export_sim = false;

double t_coll=0, t_asm = 0, t_precond=0, t_rhs = 0, t_solve = 0, t_SR = 0; 

using namespace mfem;
std::shared_ptr<SimConfig> config;
std::shared_ptr<MaterialModel> material;
std::shared_ptr<MaterialConfig> material_config;
std::shared_ptr<SimObject> object;

// ------------------------------------ //

void simulation_step() {
  Simulator sim(object, config);
  sim.step();
  meshV = object->vertices();
}

void callback() {

  static bool simulating = false;
  static bool show_pinned = false;

  ImGui::PushItemWidth(100);

  //ImGui::Checkbox("floor collision",&floor_collision);
  ImGui::Checkbox("force",&floor_collision);
  ImGui::Checkbox("warm start",&warm_start);
  ImGui::Checkbox("external forces",&enable_ext);
  ImGui::Checkbox("simulate",&simulating);
  ImGui::Checkbox("export",&export_sim);
  //if(ImGui::Button("show pinned")) {
  //} 

  static int step = 0;
  static int export_step = 0;
  if(ImGui::Button("sim step") || simulating) {
    //simulation_step();
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
    // std::cout << "STEP: " << step << std::endl;
    // std::cout << "[Avg Time ms] " 
    //   << " collision: " << t_coll / solver_steps / step
    //   << " rhs: " << t_rhs / solver_steps / step
    //   << " preconditioner: " << t_precond / solver_steps / step
    //   << " KKT assembly: " << t_asm / solver_steps / step
    //   << " cg.solve(): " << t_solve / solver_steps / step
    //   << " update S & R: " << t_SR / solver_steps / step
    //   << std::endl;
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
  config = std::make_shared<SimConfig>();
  config->plane_d = a(1);
  config->inner_steps=1;
  config->outer_steps=10;
  config->thickness = 1e-2;
  config->density = 100;
  config->beta = 100;
  double ym = 1e6;
  double pr = 0.45;
  material_config = std::make_shared<MaterialConfig>();
  material_config->mu = ym/(2.0*(1.0+pr));
  material_config->la = (ym*pr)/((1.0+pr)*(1.0-2.0*pr));
  material = std::make_shared<StableNeohookean>(material_config);
  object = std::make_shared<RodObject>(meshV, meshE, meshN, meshBN,
      config, material, material_config);
  object->init();


  // Show the gui
  polyscope::show();

  return 0;
}

