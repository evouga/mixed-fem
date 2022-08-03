#pragma once

#include <igl/IO>

// Polyscope
#include "polyscope/polyscope.h"
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/volume_mesh.h"
#include "polyscope/surface_mesh.h"
#include "args/args.hxx"
#include "json/json.hpp"

#include "mesh/mesh.h"
#include "optimizers/optimizer.h"
#include "energies/material_model.h"

#include "factories/solver_factory.h"
#include "factories/optimizer_factory.h"
#include "factories/integrator_factory.h"
#include "factories/material_model_factory.h"

namespace mfem {

  template <typename Factory, typename TypeEnum>
  bool FactoryCombo(const char* id, TypeEnum& type) {
    static Factory factory;
    const std::vector<std::string>& names = factory.names();
    std::string name = factory.name_by_type(type);
    bool ret = false;;

    if (ImGui::BeginCombo(id, name.c_str())) {
      for (size_t i = 0; i < names.size(); ++i) {
        TypeEnum type_i = factory.type_by_name(names[i]);
        const bool is_selected = (type_i == type);
        if (ImGui::Selectable(names[i].c_str(), is_selected)) {
          type = type_i;
          //optimizer->reset();
          ret = true;
        }

        // Set the initial focus when opening the combo
        // (scrolling + keyboard navigation focus)
        if (is_selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }
    return ret;
  }

  template <int DIM>
  struct PolyscopeApp {
    
    // Helper to display a little (?) mark which shows a tooltip when hovered.
    static void HelpMarker(const char* desc) {
      ImGui::TextDisabled("(?)");
      if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
      }
    }

    virtual void simulation_step() {

      optimizer->step();
      meshV = mesh->vertices();

      if (DIM == 3) {
        srf->updateVertexPositions(meshV);
      } else {
        srf->updateVertexPositions2D(meshV);
      }
    }

    virtual void reset() {
      optimizer->reset();

      if (initMeshV.size() != 0) {
        optimizer->update_vertices(initMeshV);
        srf->updateVertexPositions(initMeshV);
      }

      if (x0.size() != 0) {
        optimizer->set_state(x0, v);
        if (DIM == 3) {
          srf->updateVertexPositions(mesh->V_);
        } else {
          srf->updateVertexPositions2D(mesh->V_);
        }          
      } else {
        if (DIM == 3) {
          srf->updateVertexPositions(mesh->V0_);
        } else {
          srf->updateVertexPositions2D(mesh->V0_);
        }
      }
    }

    virtual void callback() {

      static bool export_obj = false;
      static bool export_mesh = false;
      static bool export_sim_substeps = false;
      static bool simulating = false;
      static int step = 0;
      static int export_step = 0;
      static int max_steps = 300;

      ImGui::PushItemWidth(100);


      ImGui::Checkbox("export obj",&export_obj);
      ImGui::SameLine();
      ImGui::Checkbox("export substeps",&config->save_substeps);
      ImGui::SameLine();
      ImGui::Checkbox("export mesh",&export_mesh);

      for (size_t i = 0; i < callback_funcs.size(); ++i) {
        callback_funcs[i]();
      }

      if (ImGui::TreeNode("Material Params")) {

        if (FactoryCombo<MaterialModelFactory, MaterialModelType>(
            "Material Model", material_config->material_model)) {
          material = material_factory.create(material_config->material_model,
              material_config);
          mesh->material_ = material;
        }

        double lo=0.1,hi=0.5;
        if (ImGui::InputScalar("Young's Modulus", ImGuiDataType_Double,
            &material_config->ym, NULL, NULL, "%.3e")) {
          
          Enu_to_lame(material_config->ym, material_config->pr,
              material_config->la, material_config->mu);
          config->kappa = material_config->mu;

          if (config->optimizer == OPTIMIZER_SQP_BENDING) {
            double E = material_config->ym;
            double nu = material_config->pr;
            double thickness = material_config->thickness;
            config->kappa = E / (24 * (1.0 - nu * nu))
                * thickness * thickness * thickness;
          }
        }
        // ImGui::SameLine(); 
        // HelpMarker("Young's Modulus");
        if (ImGui::SliderScalar("Poisson's Ratio", ImGuiDataType_Double,
            &material_config->pr, &lo, &hi)) {
          
          Enu_to_lame(material_config->ym, material_config->pr,
              material_config->la, material_config->mu);

          if (config->optimizer == OPTIMIZER_SQP_BENDING) {
            double E = material_config->ym;
            double nu = material_config->pr;
            double thickness = material_config->thickness;
            config->kappa = E / (24 * (1.0 - nu * nu))
                * thickness * thickness * thickness;
          }
        }

        if (config->optimizer == OPTIMIZER_SQP_BENDING) {
          if (ImGui::InputDouble("Thickness", &material_config->thickness)) {
            double E = material_config->ym;
            double nu = material_config->pr;
            double thickness = material_config->thickness;
            config->kappa = E / (24 * (1.0 - nu * nu))
                * thickness * thickness * thickness;
          }

          ImGui::InputDouble("Density", &material_config->density);
        }
        ImGui::TreePop();
      }

      ImGui::SetNextItemOpen(true, ImGuiCond_Once);
      if (ImGui::TreeNode("Sim Params")) {

        if (ImGui::InputDouble("Timestep", &config->h, 0,0,"%.5f")) {
          config->h2 = config->h*config->h;
          config->ih2 = 1.0/config->h/config->h;
        }

        if (FactoryCombo<OptimizerFactory<DIM>, OptimizerType>(
            "Optimizer", config->optimizer)) {
          optimizer = optimizer_factory.create(config->optimizer, state);
          optimizer->reset();
        }

        ImGui::InputInt("Max Newton Iters", &config->outer_steps);
        ImGui::InputInt("Max LS Iters", &config->ls_iters);
        ImGui::InputDouble("Newton Tol", &config->newton_tol,0,0,"%.5g");

        if (config->solver_type == SolverType::SOLVER_AFFINE_PCG) {
          ImGui::InputInt("Max CG Iters", &config->max_iterative_solver_iters);
          ImGui::InputDouble("CG Tol", &config->itr_tol,0,0,"%.5g");
        }

        if (ImGui::InputFloat3("Body Force", config->ext, 3)) {
        }

        ImGui::InputDouble("kappa", &config->kappa,0,0,"%.5g");
        if (config->optimizer == OPTIMIZER_ALM
            || config->optimizer == OPTIMIZER_ADMM) {
          ImGui::InputDouble("kappa", &config->kappa,0,0,"%.5g");
          ImGui::SameLine(); 
          ImGui::InputDouble("max kappa", &config->max_kappa, 0,0,"%.5g");
          ImGui::InputDouble("constraint tol",&config->constraint_tol, 0,0,"%.5g");
          ImGui::InputDouble("lamda update tol",&config->update_zone_tol,0,0,"%.5g");
        }
        if (config->optimizer == OPTIMIZER_SQP_BENDING) {
          ImGui::InputDouble("kappa", &config->kappa,0,0,"%.5g");
        }
        // ImGui::Checkbox("floor collision",&config->floor_collision);
        // ImGui::Checkbox("warm start",&config->warm_start);


        int type = config->bc_type;
        const char* combo_preview_value = bc_list[type].c_str(); 
        if (ImGui::BeginCombo("Boundary Condition", combo_preview_value)) {
          for (int n = 0; n < bc_list.size(); ++n) {
            const bool is_selected = (type == n);
            if (ImGui::Selectable(bc_list[n].c_str(), is_selected)) {
              type = n;
              config->bc_type = static_cast<BCScriptType>(type);
              optimizer->reset();
              //std::cout << "begin fixed" << std::endl;
              //for (int i = 0; i < mesh->fixed_vertices_.size();++i) {
              //  std::cout << mesh->fixed_vertices_[i] << std::endl;
              //}
            }

            // Set the initial focus when opening the combo
            // (scrolling + keyboard navigation focus)
            if (is_selected)
              ImGui::SetItemDefaultFocus();
        }
          ImGui::EndCombo();
        }

        if (FactoryCombo<SolverFactory, SolverType>(
            "Linear Solver", config->solver_type)) {
          optimizer->reset();
        }

        if (FactoryCombo<IntegratorFactory, TimeIntegratorType>(
            "Integrator", config->ti_type)) {
          optimizer->reset();
        }

        ImGui::TreePop();
      }

      ImGui::Checkbox("Output optimizer data",&config->show_data);
      ImGui::SameLine();
      ImGui::Checkbox("Output timing info",&config->show_timing);

      ImGui::Checkbox("simulate",&simulating);
      ImGui::SameLine();
      if(ImGui::Button("step") || simulating) {
        std::cout << "Timestep: " << step << std::endl;
        simulation_step();
        ++step;

        if (export_obj) {
          char buffer [50];
          int n = sprintf(buffer, "../output/obj/tet_%04d.obj", export_step++); 
          buffer[n] = 0;
          if (skinV.rows() > 0)
            igl::writeOBJ(std::string(buffer),skinV,skinF);
          else {
            if (meshV.cols() == 3) {
              igl::writeOBJ(std::string(buffer),meshV,meshF);
            } else {
              // If in 2D, pad the matrix
              Eigen::MatrixXd tmp(meshV.rows(), 3);
              tmp.setZero();
              tmp.col(0) = meshV.col(0);
              tmp.col(1) = meshV.col(1);
              igl::writeOBJ(std::string(buffer),tmp,meshF);

            }

          }
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
          Eigen::MatrixXd x(optimizer->step_x[0].size(),
              optimizer->step_x.size());
          for (size_t i = 0; i < optimizer->step_x.size(); ++i) {
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
        reset();
        export_step = 0;
        step = 0;
      }
      if (step >= max_steps) {
        simulating = false;
      }
      ImGui::InputInt("Max Steps", &max_steps);
      ImGui::PopItemWidth();
    }

    polyscope::SurfaceMesh* srf = nullptr;
    polyscope::SurfaceMesh* srf_skin = nullptr;

//    std::function<void(const std::vector<std::shared_ptr<Variable<DIM>>>&)> callback;

    std::vector<std::function<void()>> callback_funcs;

    // The mesh, Eigen representation
    Eigen::MatrixXd meshV, meshV0, skinV, initMeshV;
    Eigen::MatrixXi meshF, skinF;
    Eigen::MatrixXi meshT; // tetrahedra
    Eigen::SparseMatrixd lbs; // linear blend skinning matrix
    Eigen::VectorXd x0, v;

    MaterialModelFactory material_factory;
    OptimizerFactory<DIM> optimizer_factory;
    SolverFactory solver_factory;

    std::shared_ptr<MaterialConfig> material_config;
    std::shared_ptr<MaterialModel> material;
    std::shared_ptr<Optimizer<DIM>> optimizer;
    std::shared_ptr<SimConfig> config;
    std::shared_ptr<Mesh> mesh;

    std::vector<std::string> bc_list;
    SimState<DIM> state;

  };

  
  
}
