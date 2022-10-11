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
std::vector<MatrixXi> frame_tets;
polyscope::SurfaceMesh* frame_srf = nullptr; // collision frame mesh
polyscope::VolumeMesh* frame_tet = nullptr; // collision frame tetrahedra
std::vector<MatrixXi> het_faces;

struct PolyscopTetApp : public PolyscopeApp<3> {

  virtual void simulation_step() override {
    vertices.clear();
    frame_faces.clear();
    frame_tets.clear();

    optimizer->step();
    meshV = mesh->vertices();
    size_t start = 0;
    for (size_t i = 0; i < srfs.size(); ++i) {
      size_t sz = meshes[i]->vertices().rows();
      srfs[i]->updateVertexPositions(meshV.block(start,0,sz,3));
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

    // Show triangle frames
    if (show_frames && frame_faces.size() > 0) {

      // Update frame mesh
      frame_srf = polyscope::registerSurfaceMesh("frames", vertices[substep],
          frame_faces[substep]);

    } else if (frame_srf) {
      frame_srf->setEnabled(false);
    }

    // Show tetrahedron frames
    if (show_frames && frame_tets.size() > 0) {
      // Update frame mesh
      frame_tet = polyscope::registerTetMesh("tet_frames", vertices[substep],
          frame_tets[substep]);

    } else if (frame_tet) {
      frame_tet->setEnabled(false);
    }
  }

  template <typename T, typename U>
  bool add_collision_frames(const SimState<3>& state, const U* var) {

    const auto& x = state.x_;

    const T* c = dynamic_cast<const T*>(var);
    if (!c) return false;
    int n = c->num_collision_frames();

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
    std::vector<std::vector<int>> tets;
    
    // Add collision frames
    const auto& frames = c->frames();
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
    vertices.push_back(V);
    frame_faces.push_back(Fframe);
    frame_tets.push_back(Tframe);
    return true;
  }

  void collision_callback(const SimState<3>& state) {
    for (size_t i = 0; i < state.mixed_vars_.size(); ++i) {
      if (add_collision_frames<MixedCollision<3>>(state,
          state.mixed_vars_[i].get())) {
        return;
      }
    }
    for (size_t i = 0; i < state.vars_.size(); ++i) {
      if (add_collision_frames<Collision<3>>(state, state.vars_[i].get())) {
        return;
      }
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
        std::cout << "exporting sd V:" << sd.F_.rows() << ", " << sd.F_.cols() << " sd TC: " << sd.TC_.rows() << ", " << sd.TC_.cols() << std::endl;
        std::cout << "W: " << sd.W_.rows() << ", " << sd.W_.cols() << " V: " << sz << std::endl;
        std::cout << "W norm: " << sd.W_.norm() << std::endl;
        MatrixXd V = sd.W_ * meshV.block(start,0,sz,3);
        std::cout << "V size: " << V.rows() << ", " << V.cols() << std::endl;
        std::cout << "V0 size: " << sd.V_.rows() << ", " << sd.V_.cols() << std::endl;
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

    BoundaryConditions<3>::get_script_names(bc_list);

    std::shared_ptr<Meshes> m = std::dynamic_pointer_cast<Meshes>(state.mesh_);
    meshes = m->meshes();
    for (int i = 0; i < meshes.size(); ++i) {
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
    if (mesh->mat_ids_.size() > 0) {
      int i = 0;
      while(true) {
        MatrixXi T = igl::slice_mask(mesh->T_, mesh->mat_ids_.array() == i, 1);
        if (T.size() == 0) break;
        // Expect the first material ID to be for the surface mesh
        if (i == 0) {
          T = mesh->T_;
        }
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
