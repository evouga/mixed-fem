#include "polyscope_app.h"

// libigl
#include <igl/boundary_facets.h>
#include <igl/IO>
#include <igl/AABB.h>
#include <igl/in_element.h>
#include <igl/barycentric_coordinates.h>

#include "boundary_conditions.h"
#include <sstream>
#include <fstream>
#include <functional>
#include <string>

using namespace Eigen;
using namespace mfem;

struct PolyscopeTetApp : public PolyscopeApp {
  
  virtual void simulation_step() override {
    PolyscopeApp::simulation_step();

    // if skin enabled too
    if (skinV.rows() > 0) {
      skinV.col(0) = lbs * meshV.col(0);
      skinV.col(1) = lbs * meshV.col(1);
      skinV.col(2) = lbs * meshV.col(2);
    }
  }

  void init_skin(const std::string& filename) {
    igl::readOBJ(filename, skinV, skinF);
    double fac = skinV.maxCoeff();
    skinV.array() /= 191.599;
    std::cout << "fac2: " << fac << std::endl;

    // Create skinning transforms
    igl::AABB<MatrixXd,3> aabb;
    aabb.init(meshV, meshT);
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
    srf_skin = polyscope::registerSurfaceMesh("skin mesh",skinV,skinF);
    srf_skin->setEnabled(false);
  }


  void init(const std::string& filename) {
    // Read the mesh
    igl::readMESH(filename, meshV, meshT, meshF);
    double fac = meshV.maxCoeff();
    meshV.array() /= fac;
    std::cout << "fac: " << fac << std::endl;

    // Register the mesh with Polyscope
    polyscope::options::autocenterStructures = false;
    if (meshF.size() == 0){ 
      igl::boundary_facets(meshT, meshF);
    }
    
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
    material = material_factory.create(material_config->material_model,
        material_config);
    config->kappa = material_config->mu;

    mesh = std::make_shared<TetrahedralMesh>(meshV, meshT,
        material, material_config);

    optimizer = optimizer_factory.create(config->optimizer, mesh, config);
    optimizer->reset();

    BoundaryConditions<3>::get_script_names(bc_list);
  }
};

PolyscopeTetApp app;


int main(int argc, char **argv) {

  // omp_set_num_threads(8);
  // Configure the argument parser
  args::ArgumentParser parser("Mixed FEM");
  args::Positional<std::string> inFile(parser, "<rest>.mesh", "est mesh");
  args::Positional<std::string> inSurf(parser, "hires mesh", "hires surface");
  args::ValueFlag<std::string> init_mesh(parser, "sim_v_<step>.dmat", "initial mesh", {'r'});
  args::ValueFlag<std::string> x0_arg(parser, "sim_x0_<step>.dmat", "x0 value for step", {"x0"});
  args::ValueFlag<std::string> v_arg(parser, "sim_v_<step>.dmat", "v value for step", {'v'});

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

  // Check if initial mesh provided
  if (init_mesh) {
    std::string filename = args::get(init_mesh);
    std::cout << "loading initial mesh: " << filename << std::endl;

    // Read the mesh
    MatrixXi tmpT, tmpF;
    igl::readMESH(filename, app.initMeshV, tmpT, tmpF);
    app.initMeshV.array();
    app.optimizer->update_vertices(app.initMeshV);
    app.srf->updateVertexPositions(app.initMeshV);
  }

  if (x0_arg && v_arg) {
    std::string x0_fn = args::get(x0_arg);
    std::string v_fn = args::get(v_arg);
    igl::readDMAT(x0_fn, app.x0);
    igl::readDMAT(v_fn, app.v);
    app.optimizer->set_state(app.x0, app.v);
    app.srf->updateVertexPositions(app.mesh->V_);
  }

  // Check if skinning mesh is provided
  if (inSurf) {
    std::string hires_fn = args::get(inSurf);
    std::cout << "Reading in skinning mesh: " << hires_fn << std::endl;
    app.init_skin(hires_fn);
  }


  // Add the callback
  polyscope::state::userCallback = std::bind(&PolyscopeApp::callback, app);

  // Show the gui
  polyscope::show();

  return 0;
}

