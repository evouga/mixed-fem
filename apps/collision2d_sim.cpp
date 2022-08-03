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
#include <algorithm>
#include "variables/collision.h"
#include "variables/displacement.h"
#include "variables/stretch.h"

using namespace Eigen;
using namespace mfem;

std::vector<MatrixXd> vertices;
std::vector<MatrixXi> frame_faces;
polyscope::SurfaceMesh* frame_srf = nullptr; // collision frame mesh

std::shared_ptr<Mesh> load_mesh(const std::string& fn) {
  MatrixXd V,NV;
  MatrixXi T,NT;
  igl::read_triangle_mesh(fn,V,T);
  std::cout << "V size: " << V.size() << std::endl;
  VectorXi VI,VJ;
  igl::remove_unreferenced(V,T,NV,NT,VI,VJ);
  V = NV;
  T = NT;
  // V.array() /= V.maxCoeff();

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
    std::cout << "SIMULATION STEP " << std::endl;
    vertices.clear();
    frame_faces.clear();
    std::cout << "vertices size: " << vertices.size() << std::endl;

    optimizer->step();
    meshV = mesh->vertices();
    size_t start = 0;
    for (size_t i = 0; i < srfs.size(); ++i) {
      size_t sz = meshes[i]->vertices().rows();
      srfs[i]->updateVertexPositions2D(meshV.block(start,0,sz,2));
      start += sz;
    }
  }

  void collision_gui() {

    static bool show_substeps = false;
    static int substep = 0;
    static bool show_frames = true;
    ImGui::Checkbox("Show Substeps",&show_substeps);


    if(!show_substeps) return;
    ImGui::InputInt("Substep", &substep);

    assert(vertices.size() == frame_faces.size());
    if (show_frames && frame_faces.size() > 0) {
      substep = std::clamp(substep, 0, int(vertices.size()-1));

      // Update frame mesh
      frame_srf = polyscope::registerSurfaceMesh2D("frames", vertices[substep],
          frame_faces[substep]);

      // Update regular meshes
      size_t start = 0;
      for (size_t i = 0; i < srfs.size(); ++i) {
        size_t sz = meshes[i]->vertices().rows();
        srfs[i]->updateVertexPositions2D(vertices[substep].block(start,0,sz,2));
        start += sz;
      }

    } else if (frame_srf) {
      frame_srf->setEnabled(false);
    }
  }

  void collision_callback(const SimState<2>& state) {
    std::shared_ptr<Collision<2>> c;
    std::shared_ptr<Displacement<2>> x = state.x_;

    // Determine if variables include the required displacement and collision
    for (size_t i = 0; i < state.vars_.size(); ++i) {
      if(!c) c = std::dynamic_pointer_cast<Collision<2>>(state.vars_[i]);
    }

    if (!x || !c) {
      std::cout << "need displacement and collision yo" << std::endl;
      return;
    }
    int n = c->num_collision_frames();
    MatrixXi Fframe(n,3);

    // Get vertices at current iteration
    VectorXd xt = x->value();
    x->unproject(xt);
    MatrixXd V = Map<MatrixXd>(xt.data(), mesh->V_.cols(), mesh->V_.rows());
    V.transposeInPlace();

    // Add collision frames
    const std::vector<CollisionFrame>& frames = c->frames();
    for (int i = 0; i < n; ++i) {
      Fframe.row(i) = frames[i].E_.transpose();
    }
    vertices.push_back(V);
    frame_faces.push_back(Fframe);
  }

  virtual void reset() {
    optimizer->reset();
    for (size_t i = 0; i < srfs.size(); ++i) {
      srfs[i]->updateVertexPositions2D(meshes[i]->V0_);
    }
    if (frame_srf != nullptr) {
      removeStructure(frame_srf);
      frame_srf = nullptr;
    }
  }

  void init(const std::vector<std::string>& filenames) {

    for (size_t i = 0; i < filenames.size(); ++i) {
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
    meshF = mesh->T_;


    // Initial simulation setup
    config = std::make_shared<SimConfig>();
    //config->bc_type = BC_NULL;
    config->bc_type = BC_HANGENDS;
    config->solver_type = SOLVER_EIGEN_LU;

    state.mesh_ = mesh;
    state.config_ = config;
    state.x_ = std::make_shared<Displacement<2>>(mesh, config);
    state.vars_ = {
      std::make_shared<Stretch<2>>(mesh),
      std::make_shared<Collision<2>>(mesh, config)
    };

    SolverFactory solver_factory;
    state.solver_ = solver_factory.create(config->solver_type, mesh, config);

    //config->optimizer = OptimizerType::OPTIMIZER_NEWTON;
    optimizer = optimizer_factory.create(config->optimizer, state);
    optimizer->reset();

    BoundaryConditions<2>::get_script_names(bc_list);

    optimizer->callback = std::bind(&PolyscopeTriApp::collision_callback, this,
        std::placeholders::_1);

    callback_funcs.push_back(std::bind(&PolyscopeTriApp::collision_gui, this));
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
