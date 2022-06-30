#pragma once

#include "EigenTypes.h"

namespace mfem {

  class Mesh;

  // Base class for mixed fem variables.
  template<int dim>
  class Variable {
  public:
    
    Variable(std::shared_ptr<Mesh> mesh) : mesh_(mesh) {
    }

    // Evaluate the energy associated with the variable
    // x - variable value
    virtual double energy(const Eigen::VectorXd& x) = 0;

    // Update the state given a new set of displacements
    // x  - Nodal displacements
    // dt - Timestep size
    virtual void update(const Eigen::VectorXd& x, double dt) = 0;

    // Resets the state
    virtual void reset() = 0;

    // Build and return the right-hand-side of schur-complement
    // reduced system of equations
    virtual Eigen::VectorXd rhs() = 0;

    // Gradient of the energy
    virtual Eigen::VectorXd gradient() = 0;

    // Left-hand-side of the system
    virtual const Eigen::SparseMatrix<double, Eigen::RowMajor>& lhs() = 0;

    // Returns the updates from the mixed solves
    virtual Eigen::VectorXd& delta() = 0;

    virtual Eigen::VectorXd& value() = 0;

  protected:

    std::shared_ptr<Mesh> mesh_;

  };

}
