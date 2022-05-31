#include "polyscope_app.h"

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

struct PolyscopeTriApp : public PolyscopeApp {

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

    igl::per_face_normals(meshV,meshF,meshN);
    std::cout << "MESHN\n" << meshN << std::endl;
    // meshN = -meshN;

    // Register the mesh with Polyscope
    polyscope::options::autocenterStructures = false;
    polyscope::registerSurfaceMesh("input mesh", meshV, meshF);

    polyscope::getSurfaceMesh("input mesh")->
    addFaceVectorQuantity("normals", meshN);

    srf = polyscope::registerSurfaceMesh("input mesh", meshV, meshF);
    VectorXi pinnedV;
    pinnedV.resize(meshV.rows());
    pinnedV.setZero();
    srf->addVertexScalarQuantity("pinned", pinnedV);

    // Adjust ground plane
    VectorXd a = meshV.colwise().minCoeff();
    VectorXd b = meshV.colwise().maxCoeff();
    a(1) -= (b(1)-a(1))*0.5;
    polyscope::options::automaticallyComputeSceneExtents = false;
    polyscope::state::lengthScale = 1.;
    polyscope::state::boundingBox = std::tuple<glm::vec3, glm::vec3>{
      {a(0),a(1),a(2)},{b(0),b(1),b(2)}};
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

    meshV0 = meshV;

    // Initial simulation setup
    config = std::make_shared<SimConfig>();
    config->plane_d = a(1);
    config->inner_steps=1;

    material_config = std::make_shared<MaterialConfig>();
    material_config->density = 1e3;
    material_config->thickness = 1e-3;

    material = material_factory.create(material_config);
    config->kappa = material_config->mu;
    mesh = std::make_shared<TriMesh>(meshV, meshF, meshN,
        material, material_config);

    optimizer = optimizer_factory.create(mesh, config);
    optimizer->reset();

    // std::vector<std::string> names;
    BoundaryConditions<3>::get_script_names(bc_list);
  }

  MatrixXd meshN;

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

  // Add the callback
  polyscope::state::userCallback = std::bind(&PolyscopeApp::callback, app);

  // Show the gui
  polyscope::show();

  return 0;
}