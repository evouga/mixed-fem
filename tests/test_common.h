#pragma once

#include "finitediff.hpp"
#include "igl/readMESH.h"
#include "simulator.h"

namespace Test {

  using namespace Eigen;
  using namespace mfem;
  using namespace fd;

  struct App {

    App() {
      std::string filename = "../models/two_tets.mesh";
      //std::string filename = "../models/tet.mesh";
      MatrixXd meshV;
      MatrixXi meshF;
      MatrixXi meshT;

      // Read the mesh
      igl::readMESH(filename, meshV, meshT, meshF);
      double fac = meshV.maxCoeff();
      meshV.array() /= fac;


      // Initialize simulator
      config = std::make_shared<SimConfig>();
      material_config = std::make_shared<MaterialConfig>();
      material = std::make_shared<StableNeohookean>(material_config);
      obj = std::make_shared<TetrahedralObject>(meshV,
          meshT, config, material, material_config);
      obj->init();
      sim = std::make_shared<Simulator>(obj,config);

      // Perform simulation step
      config->inner_steps=1;
      sim->step();
    }

    std::shared_ptr<SimConfig> config;
    std::shared_ptr<MaterialConfig> material_config;
    std::shared_ptr<MaterialModel> material;
    std::shared_ptr<SimObject> obj;
    std::shared_ptr<Simulator> sim;
  };
};