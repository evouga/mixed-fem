#include "polyscope_app.h"
#include "polyscope/volume_mesh.h"

#include "mesh/tri2d_mesh.h"
#include "mesh/meshes.h"

// libigl
#include <igl/IO>
#include <igl/remove_unreferenced.h>
#include <igl/per_face_normals.h>
#include <igl/slice_mask.h>
#include <igl/boundary_facets.h>

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
std::vector<MatrixXi> frame_tets;
polyscope::SurfaceMesh* frame_srf = nullptr; // collision frame mesh
polyscope::VolumeMesh* frame_tet = nullptr;  // collision frame tetrahedra
polyscope::VolumeMesh* sim_mesh = nullptr;   // mesh tetrahedra
MatrixXi substep_T; // substep tetrahedra
VectorXd substep_stresses;
std::vector<MatrixXi> het_faces;

struct PolyscopTetApp : public PolyscopeApp<3> {

  virtual void simulation_step() override {
    vertices.clear();
    frame_faces.clear();
    frame_tets.clear();

    optimizer->step();
    meshV = mesh->vertices();

    // Update mesh vertex positions
    size_t start = 0;
    for (size_t i = 0; i < srfs.size(); ++i) {
      size_t sz = meshes[i]->vertices().rows();
      srfs[i]->updateVertexPositions(meshV.block(start,0,sz,3));
      start += sz;
    }
  }

  void collision_gui() {

    static bool show_substeps = false;
    static bool show_sim_data = true;
    static int substep = 0;
    static bool show_frames = true;
    ImGui::Checkbox("Show substeps",&show_substeps);
    ImGui::SameLine();
    if (ImGui::Button("Export")) {
      for (int i = 0; i < vertices.size(); ++i) {
        // std::stringstream ss;
        // ss << "substep_" << i << ".obj";
        // igl::writeOBJ(ss.str(), vertices[i], substep_T);
        size_t start = 0;
        for (int j = 0; j < srfs.size(); ++j) {
          std::stringstream ss;
          ss << "../output/mesh/mesh_" << j << "_substep_" << i << ".obj";
          size_t sz = meshes[j]->vertices().rows();
          igl::writeOBJ(ss.str(), vertices[i].block(start,0,sz,3), meshes[j]->F_);
          start += sz;
        }
      }
    }

    if(!show_substeps) return;
    ImGui::InputInt("Substep", &substep);
    ImGui::Checkbox("Show Sim data",&show_sim_data);
    
    // Update regular meshes
    if (vertices.size() > 0) {
      substep = std::clamp(substep, 0, int(vertices.size()-1));
      size_t start = 0;
      for (size_t i = 0; i < srfs.size(); ++i) {
        size_t sz = meshes[i]->vertices().rows();
        srfs[i]->updateVertexPositions(vertices[substep].block(start,0,sz,3));
        start += sz;
      }
    }

    // Visualize collision data
    // Show triangle frames
    if (show_frames && frame_faces.size() > 0) {
      if (frame_srf) {
        removeStructure(frame_srf);
      }
      // Update frame mesh. Have to initialize new mesh since connectivity 
      // changes.
      frame_srf = polyscope::registerSurfaceMesh("frames", vertices[substep],
          frame_faces[substep]);
    } else if (frame_srf) {
      frame_srf->setEnabled(false);
    }

    // Show tetrahedron frames
    if (show_frames && frame_tets.size() > 0) {
      if (frame_tet) {
        removeStructure(frame_tet);
      }
      // Update frame mesh
      frame_tet = polyscope::registerTetMesh("tet_frames", vertices[substep],
          frame_tets[substep]);
    } else if (frame_tet) {
      frame_tet->setEnabled(false);
    }

    if (show_sim_data && substep_T.size() > 0) {
      if (sim_mesh == nullptr) {
        std::cout << "creating sim mesh: " << vertices[substep].rows() << " " << substep_T.rows() << std::endl;
        sim_mesh = polyscope::registerTetMesh("sim_mesh", vertices[substep], substep_T);
        sim_mesh->addVertexScalarQuantity("partition", mesh->partition_ids_);
      } else {
        sim_mesh->updateVertexPositions(vertices[substep]);
      }

      if (substep_stresses.size()) {
        sim_mesh->addCellScalarQuantity("stress", substep_stresses);
      }

      sim_mesh->setEnabled(true);
    } else if (sim_mesh && !show_sim_data) {
      sim_mesh->setEnabled(false);
    }
  }

  // Add the collision stencils for the current simulation state
  template <typename T, typename U>
  bool add_collision_frames(const SimState<3>& state, const U* var) {

    const auto& x = state.x_;

    const T* c = dynamic_cast<const T*>(var);
    if (!c) return false;

    const auto& ipc_mesh = mesh->collision_mesh();
    const Eigen::MatrixXi& E = ipc_mesh.edges();
    const Eigen::MatrixXi& F = ipc_mesh.faces();

    std::vector<std::vector<int>> faces;
    std::vector<std::vector<int>> tets;
    
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
      } else if (ids_full.size() == 4) {
        tets.push_back(ids_full);
      }
    }
    MatrixXi Fframe(faces.size(), 3);
    for (size_t i = 0; i < faces.size(); ++i) {
      Fframe.row(i) = Map<RowVector3i>(faces[i].data());
    }
    MatrixXi Tframe(tets.size(), 4);
    for (size_t i = 0; i < tets.size(); ++i) {
      Tframe.row(i) = Map<RowVector4i>(tets[i].data());
    }
    frame_faces.push_back(Fframe);
    frame_tets.push_back(Tframe);
    return true;
  }

  void collision_callback(const SimState<3>& state) {
    if (substep_T.size() == 0){
      substep_T = state.mesh_->T_;
    }

    // Get vertices at current iteration
    VectorXd xt = state.x_->value();
    state.x_->unproject(xt);
    MatrixXd V = Map<MatrixXd>(xt.data(), state.mesh_->V_.cols(),
        state.mesh_->V_.rows());
    V.transposeInPlace();
    vertices.push_back(V);

    for (size_t i = 0; i < state.mixed_vars_.size(); ++i) {
      // Try and cast variable to collision and add collision frames
      add_collision_frames<MixedCollision<3>>(state,
          state.mixed_vars_[i].get());

      // Similarly try to cast to stretch and add stresses
      const MixedStretch<3>* stretch = 
          dynamic_cast<const MixedStretch<3>*>(state.mixed_vars_[i].get());
      if (stretch) {
        substep_stresses = stretch->max_stresses();
      }
    }
    for (size_t i = 0; i < state.vars_.size(); ++i) {
      add_collision_frames<Collision<3>>(state, state.vars_[i].get());
    }
  }

  virtual void reset() override {
    optimizer->reset();
    for (size_t i = 0; i < srfs.size(); ++i) {
      srfs[i]->updateVertexPositions(meshes[i]->Vinit_);
    }
    if (frame_srf != nullptr) {
      removeStructure(frame_srf);
      frame_srf = nullptr;
    }
    if (frame_tet != nullptr) {
      removeStructure(frame_tet);
      frame_tet = nullptr;
    }
  }

  void write_obj(int step) override {

    std::cout << "mesh mat ids: " << mesh->mat_ids_.size() << std::endl;

    meshV = mesh->vertices();

    // Hack to export heterogeneous objects right now
    // if (het_faces.size() > 0) {

    //   for (size_t i = 0; i < het_faces.size(); ++i) {
    //     char buffer [50];
    //     int n = sprintf(buffer, "../output/obj/tet_%ld_%04d.obj", i, step); 
    //     buffer[n] = 0;
    //     igl::writeOBJ(std::string(buffer),meshV,het_faces[i]);
    //   }
    //   return;
    // }

    size_t start = 0;
    for (size_t i = 0; i < srfs.size(); ++i) {
      char buffer [50];
      int n = sprintf(buffer, "../output/obj/tet_%ld_%04d.obj", i, step); 
      buffer[n] = 0;
      size_t sz = meshes[i]->vertices().rows();
      if (meshes[i]->skinning_data_.empty_) {
        igl::writeOBJ(std::string(buffer),
            meshV.block(start,0,sz,3),meshes[i]->F_);
      } else {
        const SkinningData& sd = meshes[i]->skinning_data_;
        MatrixXd V = sd.W_ * meshV.block(start,0,sz,3);
        igl::writeOBJ(std::string(buffer), V,
            sd.F_, sd.N_, sd.FN_, sd.TC_, sd.FTC_);
      }
      start += sz;
    }
  }

  void init(const std::string& filename) {

    SimState<3> state;
    state.load(filename);
    igl::boundary_facets(state.mesh_->T_, meshF);

    std::shared_ptr<Meshes> m = std::dynamic_pointer_cast<Meshes>(state.mesh_);
    meshes = m->meshes();
    for (size_t i = 0; i < meshes.size(); ++i) {
      // Register the mesh with Polyscope
      MatrixXd F;
      igl::boundary_facets(meshes[i]->T_, F);

      std::string name = "tet_mesh_" + std::to_string(i);
      polyscope::options::autocenterStructures = false;
      srfs.push_back(polyscope::registerSurfaceMesh(name,
          meshes[i]->V_, F));
    }
    mesh = state.mesh_;
    config = state.config_;
    material_config = std::make_shared<MaterialConfig>();

    // Add export meshes for heterogeneous materials
    // std::cout << "MAT IDS: " << mesh->mat_ids_ << std::endl;
    if (mesh->mat_ids_.size() > 0) {
      int i = 0;
      int n = mesh->mat_ids_.maxCoeff() + 1;
      while(true) {
        MatrixXi T;
        // If there is more than one material, then assume the first
        // two are for the heterogenous mesh
        if (i == 0 && n > 1) {
          T = igl::slice_mask(mesh->T_, 
              mesh->mat_ids_.array() == 0 || mesh->mat_ids_.array() == 1, 1);
        } else {
          T = igl::slice_mask(mesh->T_, mesh->mat_ids_.array() == i, 1);
        } 
        if (T.size() == 0) break;
        // Expect the first material ID to be for the surface mesh
        // if (i == 0) {
        //   T = mesh->T_;
        // }
        MatrixXi F;
        igl::boundary_facets(T,F);
        het_faces.push_back(F);
        ++i;
      }

    }

    optimizer = optimizer_factory.create(config->optimizer, state);
    optimizer->reset();
    optimizer->callback = std::bind(&PolyscopTetApp::collision_callback, this,
        std::placeholders::_1);

    callback_funcs.push_back(std::bind(&PolyscopTetApp::collision_gui, this));
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
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;

  // Initialize polyscope
  polyscope::init();

  std::string filename = args::get(inFile);

  app.init(filename);

  // Add the callback
  polyscope::state::userCallback = std::bind(&PolyscopeApp<3>::callback, app);

  // Show the gui
  polyscope::show();

  return 0;
}
