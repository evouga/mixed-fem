#include "polyscope_app.h"

#include "mesh/tri2d_mesh.h"
#include "mesh/meshes.h"

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

std::shared_ptr<Mesh> load_mesh(const std::string& fn) {
  MatrixXd V,NV;
  MatrixXi T,NT;
  igl::read_triangle_mesh(fn,V,T);
  std::cout << "V size: " << V.size() << std::endl;
  VectorXi VI,VJ;
  igl::remove_unreferenced(V,T,NV,NT,VI,VJ);
  V = NV;
  T = NT;
  V.array() /= V.maxCoeff();

  // Truncate z data
  MatrixXd tmp;
  tmp.resize(V.rows(),2);
  tmp.col(0) = V.col(0);
  tmp.col(1) = V.col(1);
  V = tmp;

  MaterialModelFactory material_factory;
  std::shared_ptr<MaterialConfig> material_config =
      std::make_shared<MaterialConfig>();
  std::shared_ptr<MaterialModel> material = material_factory.create(
      material_config->material_model, material_config);

  std::shared_ptr<Mesh> mesh = std::make_shared<Tri2DMesh>(V, T,
      material, material_config);
  return mesh;
}

struct PolyscopeTriApp : public PolyscopeApp<2> {

  virtual void simulation_step() {

    optimizer->step();
    meshV = mesh->vertices();
    size_t start = 0;
    for (size_t i = 0; i < srfs.size(); ++i) {
      size_t sz = meshes[i]->vertices().rows();
      srfs[i]->updateVertexPositions2D(meshV.block(start,0,sz,2));
      start += sz;
    }
  }

  virtual void reset() {
    optimizer->reset();
    for (size_t i = 0; i < srfs.size(); ++i) {
      srfs[i]->updateVertexPositions2D(meshes[i]->V0_);
    }
  }

  void init(const std::vector<std::string>& filenames) {

    for (int i = 0; i < filenames.size(); ++i) {
      mesh = load_mesh(filenames[i]);
      mesh->V_.rowwise() += Eigen::RowVector2d(0,i*3);
      mesh->V0_ = mesh->V_;
      meshV = mesh->V_;
      meshF = mesh->T_;
      meshes.push_back(mesh);
      // Register the mesh with Polyscope
      std::string name = "tri2d_mesh_" + std::to_string(i);
      polyscope::options::autocenterStructures = false;
      srfs.push_back(polyscope::registerSurfaceMesh2D(name, meshV, meshF));
    }

    MaterialModelFactory material_factory;
    material_config =
        std::make_shared<MaterialConfig>();
    std::shared_ptr<MaterialModel> material = material_factory.create(
        material_config->material_model, material_config);

    mesh = std::make_shared<Meshes>(meshes,material,material_config);


    // Initial simulation setup
    config = std::make_shared<SimConfig>();
    //config->bc_type = BC_NULL;
    config->bc_type = BC_HANGENDS;
    config->solver_type = SOLVER_EIGEN_LU;

    //config->optimizer = OptimizerType::OPTIMIZER_NEWTON;
    optimizer = optimizer_factory.create(config->optimizer, mesh, config);
    optimizer->reset();

    BoundaryConditions<2>::get_script_names(bc_list);
  }


  std::vector<polyscope::SurfaceMesh*> srfs;
  std::vector<std::shared_ptr<Mesh>> meshes;
} app;


int main(int argc, char **argv) {
  // Configure the argument parser
  args::ArgumentParser parser("Mixed FEM");
  // args::Positional<std::string> inFile(parser, "mesh", "input mesh");
  args::PositionalList<std::string> pathsList(parser, "paths", "files to commit");

  // Parse args
  std::vector<std::string> files;
  try {
    parser.ParseCLI(argc, argv);
    std::cout << "Paths: " << std::endl;
    for (auto &&path : pathsList) {
        std::cout << ' ' << path;
        files.push_back(path);
    }

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

  app.init(files);

  polyscope::view::style = polyscope::view::NavigateStyle::Planar;

  // Add the callback
  polyscope::state::userCallback = std::bind(&PolyscopeApp<2>::callback, app);

  // Show the gui
  polyscope::show();

  return 0;
}
