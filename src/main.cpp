#include "polyscope/polyscope.h"

#include <igl/PI.h>
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/boundary_loop.h>
#include <igl/exact_geodesic.h>
#include <igl/gaussian_curvature.h>
#include <igl/invert_diag.h>
#include <igl/lscm.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/readMESH.h>

#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/volume_mesh.h"

#include <iostream>
#include <unordered_set>
#include <utility>

#include "args/args.hxx"
#include "json/json.hpp"

// The mesh, Eigen representation
Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;
Eigen::MatrixXi meshT; // tetrahedra

void callback() {

  static int numPoints = 2000;
  static float param = 3.14;

  ImGui::PushItemWidth(100);

  // Curvature
  if (ImGui::Button("add curvature")) {
    //addCurvatureScalar();
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

  // Add the callback
  polyscope::state::userCallback = callback;

  // Show the gui
  polyscope::show();

  return 0;
}
