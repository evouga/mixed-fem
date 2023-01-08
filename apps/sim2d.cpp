#include "polyscope_app.h"
#include "polyscope/curve_network.h"

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
#include "variables/mixed_collision.h"
#include "variables/mixed_stretch.h"
#include "variables/collision.h"
#include "variables/stretch.h"
#include "variables/displacement.h"

using namespace Eigen;
using namespace mfem;

std::vector<MatrixXd> vertices;
std::vector<MatrixXi> frame_faces;
polyscope::SurfaceMesh* frame_srf = nullptr; // collision frame mesh
polyscope::CurveNetwork* frame_crv = nullptr; // collision frame mesh

struct PolyscopeTriApp : public PolyscopeApp<2> {

  virtual void simulation_step() {
    vertices.clear();
    frame_faces.clear();

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

  template <typename T, typename U>
  bool add_collision_frames(const SimState<2>& state, const U* var) {

    const auto& x = state.x_;

    const T* c = dynamic_cast<const T*>(var);
    if (!c) return false;
    // int n = c->num_collision_frames();

    // Get vertices at current iteration
    VectorXd xt = x->value();
    x->unproject(xt);
    MatrixXd V = Map<MatrixXd>(xt.data(), mesh->V_.cols(), mesh->V_.rows());
    V.transposeInPlace();

    const auto& ipc_mesh = mesh->collision_mesh();
    const Eigen::MatrixXi& E = ipc_mesh.edges();
    const Eigen::MatrixXi& F = ipc_mesh.faces();
    MatrixXd V_srf = ipc_mesh.vertices(V);


    std::vector<std::vector<int>> faces;


    // Add collision frames
    const auto& frames = c->frames();
    int n = frames.size();
    std::cout << "frames size: " << frames.size() << " n: " << c->num_collision_frames() << std::endl;
    for (int i = 0; i < n; ++i) {
      std::array<long, 4> ids = frames[i].vertex_indices(E, F);
      std::vector<int> ids_full;
      for (int j = 0; j < 4; ++j) {
        if (ids[j] == -1) break;
        ids_full.push_back(ipc_mesh.to_full_vertex_id(ids[j]));
      }

      // If 3 vertices, add to triangle frame list
      if (ids_full.size() == 3) {
        faces.push_back(ids_full);
      }
    }
    MatrixXi Fframe(faces.size(), 3);
    for (size_t i = 0; i < faces.size(); ++i) {
      Fframe.row(i) = Map<RowVector3i>(faces[i].data());
    }
    vertices.push_back(V);
    frame_faces.push_back(Fframe);
    return true;
  }

  void collision_callback(const SimState<2>& state) {
    for (size_t i = 0; i < state.mixed_vars_.size(); ++i) {
      if (add_collision_frames<MixedCollision<2>>(state,
          state.mixed_vars_[i].get())) {
        return;
      }
    }
    for (size_t i = 0; i < state.vars_.size(); ++i) {
      if (add_collision_frames<Collision<2>>(state, state.vars_[i].get())) {
        return;
      }
    }
  }

  virtual void reset() {
    optimizer->reset();
    for (size_t i = 0; i < srfs.size(); ++i) {
      srfs[i]->updateVertexPositions2D(meshes[i]->Vref_);
    }
    if (frame_srf != nullptr) {
      removeStructure(frame_srf);
      frame_srf = nullptr;
    }
  }

  // void write_obj(int step) override {

  //   meshV = mesh->vertices();

  //   // If in 2D, pad the matrix
  //   Eigen::MatrixXd tmp(meshV.rows(), 3);
  //   tmp.setZero();
  //   tmp.col(0) = meshV.col(0);
  //   tmp.col(1) = meshV.col(1);
    
  //   size_t start = 0;
  //   for (size_t i = 0; i < srfs.size(); ++i) {
  //     char buffer [50];
  //     int n = sprintf(buffer, "../output/obj/tri_%ld_%04d.obj", i, step); 
  //     buffer[n] = 0;
  //     size_t sz = meshes[i]->vertices().rows();
  //     igl::writeOBJ(std::string(buffer),tmp.block(start,0,sz,3),meshes[i]->T_);
  //     start += sz;
  //   }
  // }

  void init(const std::string& filename) {

    SimState<2> state;
    state.load(filename);
    meshF = state.mesh_->T_;

    BoundaryConditions<2>::get_script_names(bc_list);

    std::shared_ptr<Meshes> m = std::dynamic_pointer_cast<Meshes>(state.mesh_);
    meshes = m->meshes();
    for (size_t i = 0; i < meshes.size(); ++i) {
      // Register the mesh with Polyscope
      std::string name = "tri2d_mesh_" + std::to_string(i);
      polyscope::options::autocenterStructures = false;
      srfs.push_back(polyscope::registerSurfaceMesh2D(name,
          meshes[i]->V_, meshes[i]->T_));
    }
    mesh = state.mesh_;
    config = state.config_;
    material_config = std::make_shared<MaterialConfig>();

    optimizer = optimizer_factory.create(config->optimizer, state);
    optimizer->reset();
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
  args::Positional<std::string> inFile(parser, "json", "input scene json file");

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

  app.init(filename);

  polyscope::view::style = polyscope::view::NavigateStyle::Planar;

  // Add the callback
  polyscope::state::userCallback = std::bind(&PolyscopeApp<2>::callback, app);

  // Show the gui
  polyscope::show();

  return 0;
}
