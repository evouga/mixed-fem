#include "polyscope_app.h"
#include "polyscope/curve_network.h"

#include "mesh/tri2d_mesh.h"
#include "mesh/meshes.h"

// libigl
#include <igl/IO>
#include <igl/remove_unreferenced.h>
#include <igl/per_face_normals.h>

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
#include "implot.h"

using namespace Eigen;
using namespace mfem;


std::vector<Vector2f> displacements;
std::vector<MatrixXd> vertices;
std::vector<MatrixXi> frame_faces;
polyscope::SurfaceMesh* frame_srf = nullptr; // collision frame mesh
polyscope::CurveNetwork* frame_crv = nullptr; // collision frame mesh
                                            
const float max_disp = 5.0f;
std::vector<double> steps;
std::vector<double> values;
int curr_step = 0;
double max_val = 0.0;


struct PolyscopeTriApp : public PolyscopeApp<2> {

  virtual void simulation_step() {
    vertices.clear();
    frame_faces.clear();

    auto& state = optimizer->state();

    // Get state vertices
    MatrixXd V = mesh->vertices();
    MatrixXd tmp = V.transpose();
    VectorXd x = Map<VectorXd>(tmp.data(), V.size());

    double dt = 1.0 / config->timesteps;

    // Update derivatives (for non-mixed energies)
    for (auto& var : state.vars_) {
      var->update(x, dt);
    }
    // Update derivatives for mixed variables
    for (auto& var : state.mixed_vars_) {
      var->update(x, dt);
    }

    // TODO record energy
    // Compute energy
    double h2 = std::pow(dt, 2);
    double val = 0.0;

    for (const auto& var : state.vars_) {
      val += h2 * var->energy(x);  
    }
    steps.push_back(curr_step);
    values.push_back(val);
    max_val = std::max(max_val, val);

    // Take an incremental step for the prescribed
    // displacements of each body
    size_t start = 0;
    for (int i = 0; i < srfs.size(); ++i) {
      size_t sz = meshes[i]->vertices().size();
      VectorXd xi = x.segment(start, sz);
      xi += dt * displacements[i].cast<double>().replicate(sz/2, 1);

      // Reshape xi to matrix
      MatrixXd Vi = Map<MatrixXd>(xi.data(), 2, sz/2);
      Vi.transposeInPlace();

      // Update polyscope mesh
      srfs[i]->updateVertexPositions2D(Vi);

      mesh->V_.block(start/2, 0, sz/2, 2) = Vi;

      start += sz;
    }
    ++curr_step;
  }

  void collision_gui() {

    static bool show_substeps = false;
    static int substep = 0;
    static bool show_frames = true;

    // Create displacement slider per-body
    for (size_t i = 0; i < meshes.size(); ++i) {
      std::stringstream ss;
      ss << "Displacement " << i;
      if (ImGui::SliderFloat2(ss.str().c_str(), &displacements[i][0],
          -max_disp, max_disp)) {

        // Update the vertices for the displaced srf
        MatrixXd V_new = meshes[i]->vertices();
        Vector2d disp(displacements[i][0], displacements[i][1]);
        for (size_t j = 0; j < V_new.rows(); ++j) {
          V_new.row(j) += disp;
        }
        srfs_displaced[i]->updateVertexPositions2D(V_new);
      }
    }

    // Plot the energy
    if (ImPlot::BeginPlot("Barrier Energy")) {
      int limit = config->timesteps;
      ImPlot::SetupAxesLimits(0,limit,0,10);
      ImPlot::PlotLine("Energy", steps.data(),
          values.data(), values.size());
      ImPlot::EndPlot();
    }

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
    steps.clear();
    values.clear();
    optimizer->reset();
    for (size_t i = 0; i < srfs.size(); ++i) {
      srfs[i]->updateVertexPositions2D(meshes[i]->Vref_);
    }
    if (frame_srf != nullptr) {
      removeStructure(frame_srf);
      frame_srf = nullptr;
    }
    curr_step = 0;
  }

  void init(const std::string& filename) {

    SimState<2> state;
    state.load(filename);
    meshF = state.mesh_->T_;

    std::shared_ptr<Meshes> m = std::dynamic_pointer_cast<Meshes>(state.mesh_);
    meshes = m->meshes();

    displacements.resize(meshes.size());
    for (size_t i = 0; i < meshes.size(); ++i) {
      displacements[i].setZero();
    }


    for (size_t i = 0; i < meshes.size(); ++i) {
      // Register the mesh with Polyscope
      std::string name = "tri2d_mesh_" + std::to_string(i);
      polyscope::options::autocenterStructures = false;
      srfs.push_back(polyscope::registerSurfaceMesh2D(name,
          meshes[i]->V_, meshes[i]->T_));

      // Add mesh for final displaced position
      std::string name2 = "tri2d_mesh_" + std::to_string(i) + "_displaced";
      srfs_displaced.push_back(polyscope::registerSurfaceMesh2D(name2,
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
  std::vector<polyscope::SurfaceMesh*> srfs_displaced;
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

  ImPlot::CreateContext();

  std::string filename = args::get(inFile);

  app.init(filename);

  polyscope::view::style = polyscope::view::NavigateStyle::Planar;

  // Add the callback
  polyscope::state::userCallback = std::bind(&PolyscopeApp<2>::callback, app);

  // Show the gui
  polyscope::show();

  return 0;
}
