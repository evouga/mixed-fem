#include "polyscope/polyscope.h"

// libigl
#include <igl/boundary_facets.h>
#include <igl/IO>
#include <igl/AABB.h>
#include <igl/in_element.h>
#include <igl/barycentric_coordinates.h>

// Polyscope
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/volume_mesh.h"
#include "polyscope/surface_mesh.h"
#include "args/args.hxx"
#include "json/json.hpp"

#include "objects/simulation_object.h"
#include "materials/material_model.h"
#include "optimizers/mixed_alm_optimizer.h"
#include "optimizers/mixed_admm_optimizer.h"
#include "optimizers/mixed_sqp_optimizer.h"
#include "optimizers/mixed_sqpr_optimizer.h"
#include "optimizers/newton_optimizer.h"
#include "boundary_conditions.h"
#include <sstream>
#include <fstream>

using namespace Eigen;

// The mesh, Eigen representation
MatrixXd meshV, meshV0, skinV, initMeshV;
MatrixXi meshF, skinF;
MatrixXi meshT; // tetrahedra
SparseMatrixd lbs; // linear blend skinning matrix
VectorXi pinnedV;
VectorXd x0, v;

double t_coll=0, t_asm = 0, t_precond=0, t_rhs = 0, t_solve = 0, t_SR = 0; 

polyscope::SurfaceMesh* srf = nullptr;

using namespace mfem;
std::shared_ptr<SimConfig> config;
std::shared_ptr<MaterialModel> material;
std::shared_ptr<MaterialConfig> material_config;
std::shared_ptr<SimObject> tet_object;
std::shared_ptr<Optimizer> optimizer;

std::vector<std::string> bc_list;

// ------------------------------------ //
std::shared_ptr<MaterialModel> make_material_model(
    std::shared_ptr<MaterialConfig> config) {
  switch (config->material_model) {
  case MATERIAL_SNH:
    return std::make_shared<StableNeohookean>(config);
    break;
  case MATERIAL_NH:
    return std::make_shared<Neohookean>(config);
    break;
  case MATERIAL_FCR:
    return std::make_shared<CorotationalModel>(config);
    break;
  case MATERIAL_FUNG:
    return std::make_shared<Fung>(config);
    break;
  case MATERIAL_ARAP:
    return std::make_shared<ArapModel>(config);
    break;
  } 
  return std::make_shared<ArapModel>(config);
}

std::shared_ptr<Optimizer> make_optimizer(std::shared_ptr<SimObject> object,
    std::shared_ptr<SimConfig> config) {

  switch (config->optimizer) {
  case OPTIMIZER_ALM:
    return std::make_shared<MixedALMOptimizer>(object,config);
    break;
  case OPTIMIZER_ADMM:
    return std::make_shared<MixedADMMOptimizer>(object,config);
    break;
  case OPTIMIZER_SQP:
    return std::make_shared<MixedSQPOptimizer>(object,config);
    break;
  case OPTIMIZER_SQP_PD:
    return std::make_shared<MixedSQPROptimizer>(object,config);
    break;
  case OPTIMIZER_NEWTON:
    return std::make_shared<NewtonOptimizer>(object,config);
    break;
  } 
  return std::make_shared<MixedALMOptimizer>(object,config);
}

// Helper to display a little (?) mark which shows a tooltip when hovered.
static void HelpMarker(const char* desc)
{
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
    {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void simulation_step() {

  optimizer->step();
  meshV = tet_object->vertices();

  // if skin enabled too
  if (skinV.rows() > 0) {
    skinV.col(0) = lbs * meshV.col(0);
    skinV.col(1) = lbs * meshV.col(1);
    skinV.col(2) = lbs * meshV.col(2);
  }
}

void callback() {

  static bool export_obj = false;
  static bool export_mesh = false;
  static bool export_sim_substeps = false;
  static bool simulating = false;
  static bool BDF2 = true;
  static bool show_pinned = false;
  static int step = 0;
  static int export_step = 0;
  static bool sim_dirty = false;
  static int max_steps = 300;

  ImGui::PushItemWidth(100);


  ImGui::Checkbox("export obj",&export_obj);
  ImGui::SameLine();
  ImGui::Checkbox("export substeps",&config->save_substeps);
  ImGui::SameLine();
  ImGui::Checkbox("export mesh",&export_mesh);

  if (ImGui::TreeNode("Material Params")) {

    int type = material_config->material_model;
    if (ImGui::Combo("Material Model", &type,"SNH\0NH\0FCR\0FUNG\0ARAP\0\0")) {
      material_config->material_model = 
          static_cast<MaterialModelType>(type);
      material = make_material_model(material_config);
      tet_object->material_ = material;
    }

    double lo=0.1,hi=0.5;
    if (ImGui::InputScalar("Young's Modulus", ImGuiDataType_Double,
        &material_config->ym, NULL, NULL, "%.3e")) {
      
      Enu_to_lame(material_config->ym, material_config->pr,
          material_config->la, material_config->mu);
      std::cout << "MU: " << material_config->mu;
      std::cout << " LA: " << material_config->la << std::endl;
      config->kappa = material_config->mu;
      sim_dirty = true;    
    }
    // ImGui::SameLine(); 
    // HelpMarker("Young's Modulus");
    if (ImGui::SliderScalar("Poisson's Ratio", ImGuiDataType_Double,
        &material_config->pr, &lo, &hi)) {
      
      Enu_to_lame(material_config->ym, material_config->pr,
          material_config->la, material_config->mu);
      sim_dirty = true;    
    }
    ImGui::TreePop();
  }

  ImGui::SetNextItemOpen(true, ImGuiCond_Once);
  if (ImGui::TreeNode("Sim Params")) {

    if (ImGui::InputDouble("Timestep", &config->h, 0,0,"%.5f")) {
      config->h2 = config->h*config->h;
      config->ih2 = 1.0/config->h/config->h;
    }

    int type = config->optimizer;
    if (ImGui::Combo("Optimizer", &type,"ALM\0ADMM\0SQP\0SQP_PD\0NEWTON\0\0")) {
      config->optimizer = static_cast<OptimizerType>(type);
      optimizer = make_optimizer(tet_object, config);
      optimizer->reset();
    }

    ImGui::InputInt("Max Newton Iters", &config->outer_steps);
    ImGui::InputInt("Max CG Iters", &config->max_iterative_solver_iters);
    ImGui::InputInt("Max LS Iters", &config->ls_iters);
    ImGui::InputDouble("CG Tol", &config->itr_tol,0,0,"%.5g");
    ImGui::InputDouble("Newton Tol", &config->newton_tol,0,0,"%.5g");
    ImGui::InputDouble("Collision stiffness", &config->collision_stiffness,0,0,"%.5g");

    //ImGui::InputInt("Inner Steps", &config->inner_steps);
    if (ImGui::InputFloat3("Body Force", config->ext, 3)) {
      sim_dirty = true;
    }

    if (config->optimizer == OPTIMIZER_ALM
        || config->optimizer == OPTIMIZER_ADMM) {
      ImGui::InputDouble("kappa", &config->kappa,0,0,"%.5g");
      ImGui::SameLine(); 
      ImGui::InputDouble("max kappa", &config->max_kappa, 0,0,"%.5g");
      ImGui::InputDouble("constraint tol",&config->constraint_tol, 0,0,"%.5g");
      ImGui::InputDouble("lamda update tol",&config->update_zone_tol,0,0,"%.5g");
    }
    // ImGui::Checkbox("floor collision",&config->floor_collision);
    // ImGui::Checkbox("warm start",&config->warm_start);


    type = config->bc_type;
    const char* combo_preview_value = bc_list[type].c_str(); 
    if (ImGui::BeginCombo("Boundary Condition", combo_preview_value)) {
      for (int n = 0; n < bc_list.size(); ++n) {
        const bool is_selected = (type == n);
        if (ImGui::Selectable(bc_list[n].c_str(), is_selected)) {
          type = n;
          config->bc_type = static_cast<BCScriptType>(type);
          optimizer->reset();
        }

        // Set the initial focus when opening the combo
        // (scrolling + keyboard navigation focus)
        if (is_selected)
          ImGui::SetItemDefaultFocus();
    }
      ImGui::EndCombo();
    }


    ImGui::TreePop();
  }

  ImGui::Checkbox("Output timing info",&config->show_timing);

  ImGui::Checkbox("simulate",&simulating);
  ImGui::SameLine();
  if(ImGui::Button("step") || simulating) {
    std::cout << "Timestep: " << step << std::endl;
    simulation_step();
    ++step;
    srf->updateVertexPositions(meshV);

    polyscope::SurfaceMesh* skin_mesh;
    if ((skin_mesh = polyscope::getSurfaceMesh("skin mesh")) &&
        skin_mesh->isEnabled()) {
      skin_mesh->updateVertexPositions(skinV);
    }

    if (export_obj) {
      char buffer [50];
      int n = sprintf(buffer, "../output/obj/tet_%04d.obj", export_step++); 
      buffer[n] = 0;
      if (skinV.rows() > 0)
        igl::writeOBJ(std::string(buffer),skinV,skinF);
      else
        igl::writeOBJ(std::string(buffer),meshV,meshF);
    }
    if (export_mesh) {
      char buffer [50];
      int n = sprintf(buffer, "../output/mesh/tet_%04d.mesh", step); 
      buffer[n] = 0;
      igl::writeMESH(std::string(buffer),meshV, meshT, meshF);
      std::ofstream outfile;
      outfile.open(std::string(buffer), std::ios_base::app); 
      outfile << "End"; 
    }

    if (config->save_substeps) {
      char buffer[50];
      int n;
      MatrixXd x(optimizer->step_x[0].size(), optimizer->step_x.size());
      for (int i = 0; i < optimizer->step_x.size(); ++i) {
        x.col(i) = optimizer->step_x[i];
      }
      // Save the file names
      // n = sprintf(buffer, "../output/sim_x_%04d.dmat", step); 
      // buffer[n] = 0;
      // igl::writeDMAT(std::string(buffer), x);

      n = sprintf(buffer, "../output/sim_x0_%04d.dmat", step); 
      buffer[n] = 0;
      igl::writeDMAT(std::string(buffer), optimizer->step_x0);

      n = sprintf(buffer, "../output/sim_v_%04d.dmat", step); 
      buffer[n] = 0;
      igl::writeDMAT(std::string(buffer), optimizer->step_v);
    }

  }
  ImGui::SameLine();
  if(ImGui::Button("reset")) {
    optimizer->reset();

    if (initMeshV.size() != 0) {
      optimizer->update_vertices(initMeshV);
      srf->updateVertexPositions(initMeshV);
    }

    if (x0.size() != 0) {
      optimizer->set_state(x0, v);
      srf->updateVertexPositions(tet_object->V_);
    } else {
      srf->updateVertexPositions(tet_object->V0_);
    }
    export_step = 0;
    step = 0;
  }
  ImGui::SameLine();
  ImGui::Checkbox("BDF2", &BDF2); 
  
  if(BDF2) {
    optimizer->wx_ = 1.; optimizer->wx0_ = -7./3.; optimizer->wx1_ = 5./3.; optimizer->wx2_ = -1./3; optimizer->wdt_ = 2./3.;
  } else {
    optimizer->wx_ = 1.0; optimizer->wx0_ = -2.0; optimizer->wx1_ = 1.; optimizer->wx2_ = 0.0; optimizer->wdt_ = 1.0;
  }

  if (step >= max_steps) {
    simulating = false;
  }
  ImGui::InputInt("Max Steps", &max_steps);
  ImGui::PopItemWidth();
}

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

  // Read the mesh
  igl::readMESH(filename, meshV, meshT, meshF);
  double fac = meshV.maxCoeff();
  meshV.array() /= fac;

  // Register the mesh with Polyscope
  polyscope::options::autocenterStructures = false;
  if (meshF.size() == 0){ 
    igl::boundary_facets(meshT, meshF);
  }
  
  srf = polyscope::registerSurfaceMesh("input mesh", meshV, meshF);

  pinnedV.resize(meshV.rows());
  pinnedV.setZero();
  srf->addVertexScalarQuantity("pinned", pinnedV);

  // Check if skinning mesh is provided
  if (inSurf) {
    std::string hiresFn = args::get(inSurf);
    std::cout << "Reading in skinning mesh: " << hiresFn << std::endl;
    igl::readOBJ(hiresFn, skinV, skinF);
    skinV.array() /= fac;
    //polyscope::SurfaceMesh* skin_mesh;
    //skin_mesh = polyscope::registerSurfaceMesh("skin mesh", skinV, skinF);
    //skin_mesh->setEnabled(false);

    // Create skinning transforms
    igl::AABB<MatrixXd,3> aabb;
    aabb.init(meshV,meshT);
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
  }
  polyscope::SurfaceMesh* skin_mesh;
  skin_mesh = polyscope::registerSurfaceMesh("skin mesh", skinV, skinF);
  skin_mesh->setEnabled(false);

  // Add the callback
  polyscope::state::userCallback = callback;

  // Adjust ground plane
  VectorXd a = meshV.colwise().minCoeff();
  VectorXd b = meshV.colwise().maxCoeff();
  a(1) -= (b(1)-a(1))*0.5;
  polyscope::options::automaticallyComputeSceneExtents = false;
  polyscope::state::lengthScale = 1.;
  polyscope::state::boundingBox = std::tuple<glm::vec3, glm::vec3>{
    {a(0),0,a(2)},{b(0),b(1),b(2)}};
  //polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  std::cout << "V : " << meshV.rows() << " T: " << meshT.rows() << std::endl;
  meshV0 = meshV;

  // Initial simulation setup
  config = std::make_shared<SimConfig>();
  config->plane_d = a(1);
  config->inner_steps=1;

  material_config = std::make_shared<MaterialConfig>();
  material = make_material_model(material_config);
  config->kappa = material_config->mu;

  tet_object = std::make_shared<TetrahedralObject>(meshV, meshT,
      material, material_config);

  optimizer = make_optimizer(tet_object, config);
  optimizer->reset();

  // std::vector<std::string> names;
  BoundaryConditions<3>::get_script_names(bc_list);

  // Check if initial mesh provided
  if (init_mesh) {
    std::string filename = args::get(init_mesh);
    std::cout << "loading initial mesh: " << filename << std::endl;

    // Read the mesh
    MatrixXi tmpT, tmpF;
    igl::readMESH(filename, initMeshV, tmpT, tmpF);
    initMeshV.array();
    optimizer->update_vertices(initMeshV);
    srf->updateVertexPositions(initMeshV);
  }

  if (x0_arg && v_arg) {
    std::string x0_fn = args::get(x0_arg);
    std::string v_fn = args::get(v_arg);
    igl::readDMAT(x0_fn, x0);
    igl::readDMAT(v_fn, v);
    optimizer->set_state(x0, v);
    srf->updateVertexPositions(tet_object->V_);
  }


  // Show the gui
  polyscope::show();

  return 0;
}

