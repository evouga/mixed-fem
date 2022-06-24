#pragma once

#include "EigenTypes.h"

namespace mfem {

  class Mesh;

  // Base class for mixed fem variables.
  template<int dim>
  class MixedVariable {
  public:
    
    MixedVariable(std::shared_ptr<Mesh> mesh) : mesh_(mesh) {
    }

    // Evaluate the energy associated with the mixed variable
    // y - Mixed variable
    virtual double energy(const Eigen::VectorXd& y) = 0;

    // Evaluate the energy associated with the mixed variable constraint
    // x - Nodal displacements 
    // y - Mixed variable
    virtual double constraint_value(const Eigen::VectorXd& x,
        const Eigen::VectorXd& y) = 0;

    virtual void update(const Eigen::VectorXd& x, double dt) = 0;
    virtual void reset() = 0;

  protected:

    std::shared_ptr<Mesh> mesh_;

  };

}
