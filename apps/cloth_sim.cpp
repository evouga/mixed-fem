// #include "polyscope/polyscope.h"

// // libigl
// #include <igl/read_triangle_mesh.h>
// #include <igl/remove_unreferenced.h>
// #include <igl/readOBJ.h>
// #include <igl/writeOBJ.h>
// #include <igl/per_face_normals.h>

// // Polyscope
// #include "polyscope/messages.h"
// #include "polyscope/point_cloud.h"
// #include "polyscope/surface_mesh.h"
// #include "args/args.hxx"
// #include "json/json.hpp"

// #include "simulator.h"
// #include "objects/simulation_object.h"
// #include "materials/material_model.h"

// using namespace Eigen;

// // The mesh, Eigen representation
// MatrixXd meshV; // verts
// MatrixXi meshF; // faces
// MatrixXd meshN; // normals 

// VectorXd pinnedV;

// // UI switches
// bool enable_slide = false;
// bool enable_ext = true;
// bool warm_start = true;
// bool floor_collision = false;
// bool export_sim = false;


// double t_coll=0, t_asm = 0, t_precond=0, t_rhs = 0, t_solve = 0, t_SR = 0; 

// using namespace mfem;
// std::shared_ptr<SimConfig> config;
// std::shared_ptr<MaterialModel> material;
// std::shared_ptr<MaterialConfig> material_config;
// std::shared_ptr<SimObject> object;

// // ------------------------------------ //

// void simulation_step() {
//   Simulator sim(object, config);
//   sim.step();
//   meshV = object->vertices();
// }

// void callback() {

//   static bool simulating = false;
//   static bool show_pinned = false;

//   ImGui::PushItemWidth(100);

//   //ImGui::Checkbox("floor collision",&floor_collision);
//   ImGui::Checkbox("force",&floor_collision);
//   ImGui::Checkbox("warm start",&warm_start);
//   ImGui::Checkbox("external forces",&enable_ext);
//   ImGui::Checkbox("slide mesh",&enable_slide);
//   ImGui::Checkbox("simulate",&simulating);
//   ImGui::Checkbox("export",&export_sim);
//   //if(ImGui::Button("show pinned")) {
//   //} 
//   static int step = 0;
//   static int export_step = 0;

//   if(ImGui::Button("sim step") || simulating) {
//     simulation_step();
//     ++step;
//     polyscope::getSurfaceMesh("input mesh")
//       ->updateVertexPositions(meshV);

//     if (export_sim) {
//       char buffer [50];
//       int n = sprintf(buffer, "../data/cloth/sheet_soft2/%04d.png", export_step); 
//       buffer[n] = 0;
//       polyscope::screenshot(std::string(buffer), true);
//       n = sprintf(buffer, "../data/cloth/sheet_soft2/%04d.obj", export_step++); 
//       buffer[n] = 0;
//       igl::writeOBJ(std::string(buffer),meshV,meshF);
//     }

//     // std::cout << "STEP: " << step << std::endl;
//     // std::cout << "[Avg Time ms] " 
//     //   << " collision: " << t_coll / solver_steps / step
//     //   << " rhs: " << t_rhs / solver_steps / step
//     //   << " preconditioner: " << t_precond / solver_steps / step
//     //   << " KKT assembly: " << t_asm / solver_steps / step
//     //   << " cg.solve(): " << t_solve / solver_steps / step
//     //   << " update S & R: " << t_SR / solver_steps / step
//     //   << std::endl;

//   }
//   ImGui::PopItemWidth();
// }

// int main(int argc, char **argv) {
//   // Configure the argument parser
//   args::ArgumentParser parser("Mixed FEM");
//   args::Positional<std::string> inFile(parser, "mesh", "input mesh");

//   // Parse args
//   try {
//     parser.ParseCLI(argc, argv);
//   } catch (args::Help) {
//     std::cout << parser;
//     return 0;
//   } catch (args::ParseError e) {
//     std::cerr << e.what() << std::endl;

//     std::cerr << parser;
//     return 1;
//   }

//   // Options
//   polyscope::options::autocenterStructures = true;
//   polyscope::view::windowWidth = 1024;
//   polyscope::view::windowHeight = 1024;

//   // Initialize polyscope
//   polyscope::init();

//   std::string filename = args::get(inFile);
//   std::cout << "loading: " << filename << std::endl;

//   // Read the mesh
//   //igl::readOBJ(filename, meshV, meshF);

//   // Read tetmesh
//   igl::read_triangle_mesh(filename,meshV,meshF);
//   Eigen::MatrixXd NV;
//   Eigen::MatrixXi NF;
//   VectorXi VI,VJ;
//   igl::remove_unreferenced(meshV,meshF,NV,NF,VI,VJ);
//   meshV = NV;
//   meshF = NF;

//   meshV.array() /= meshV.maxCoeff();

//   igl::per_face_normals(meshV,meshF,meshN);

//   // Register the mesh with Polyscope
//   polyscope::options::autocenterStructures = false;
//   polyscope::registerSurfaceMesh("input mesh", meshV, meshF);

//   meshN = -meshN;
//   polyscope::getSurfaceMesh("input mesh")->
//     addFaceVectorQuantity("normals", meshN);

//   pinnedV.resize(meshV.rows());
//   pinnedV.setZero();
//   polyscope::getSurfaceMesh("input mesh")
//     ->addVertexScalarQuantity("pinned", pinnedV);

//   // Add the callback
//   polyscope::state::userCallback = callback;

//   // Initial simulation setup
//   config = std::make_shared<SimConfig>();
//   config->inner_steps=2;
//   config->density = 1000.0;
//   config->thickness = 1e-3;
//   material_config = std::make_shared<MaterialConfig>();
//   material = std::make_shared<StableNeohookean>(material_config);
//   object = std::make_shared<TriObject>(meshV, meshF, meshN,
//       config, material, material_config);
//   object->init();

//   // Show the gui
//   polyscope::show();

//   return 0;
// }

int main(int argc, char **argv) {
   return 1;
}