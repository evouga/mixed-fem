#include "polyscope_app.h"

#include "mesh/tri2d_mesh.h"

// libigl
#include <igl/IO>
#include <igl/remove_unreferenced.h>
#include <igl/per_face_normals.h>

#include "boundary_conditions.h"
#include <sstream>
#include <fstream>
#include <functional>
#include <string>

using namespace Eigen;
using namespace mfem;

struct PolyscopeTriApp : public PolyscopeApp<2> {

  void init(const std::string& filename) {
    // Read tetmesh
    igl::read_triangle_mesh(filename,meshV,meshF);
    Eigen::MatrixXd NV;
    Eigen::MatrixXi NF;
    VectorXi VI,VJ;
    igl::remove_unreferenced(meshV,meshF,NV,NF,VI,VJ);
    meshV = NV;
    meshF = NF;
    meshV.array() /= meshV.maxCoeff();

    // Truncate z data
    MatrixXd tmp;
    tmp.resize(meshV.rows(),2);
    tmp.col(0) = meshV.col(0);
    tmp.col(1) = meshV.col(1);
    meshV = tmp;

    // Register the mesh with Polyscope
    polyscope::options::autocenterStructures = false;
    srf = polyscope::registerSurfaceMesh2D("input mesh", meshV, meshF);

    // Initial simulation setup
    config = std::make_shared<SimConfig>();

    material_config = std::make_shared<MaterialConfig>();

    material = material_factory.create(material_config->material_model,
        material_config);

    mesh = std::make_shared<Tri2DMesh>(meshV, meshF,
        material, material_config);

    config->optimizer = OptimizerType::OPTIMIZER_NEWTON;
    optimizer = optimizer_factory.create(config->optimizer, mesh, config);
    optimizer->reset();

    BoundaryConditions<2>::get_script_names(bc_list);
  }

} app;


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
  app.init(filename);

  polyscope::view::style = polyscope::view::NavigateStyle::Planar;

  // Add the callback
  polyscope::state::userCallback = std::bind(&PolyscopeApp<2>::callback, app);

  // Show the gui
  polyscope::show();

  return 0;
}