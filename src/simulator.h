#pragma once

#include <Eigen/Dense>
#include <EigenTypes.h>
#include <memory>

#include "materials/material_model.h"
#include "objects/simulation_object.h"

#include "config.h"

namespace mfem {

  using SparseMatrixdRowMajor = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  class Simulator {
  public:

    Simulator(std::shared_ptr<SimObject> object)
        : object_(object) {}

    void init();

    // Perform a single simulation step
    void step();

  private:

    SimConfig config_;
    std::shared_ptr<SimObject> object_;

    // Debug timing variables
    double t_coll = 0;
    double t_asm = 0;
    double t_precond = 0;
    double t_rhs = 0;
    double t_solve = 0;
    double t_SR = 0; 
  };
}