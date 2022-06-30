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

    // Update the state given a new set of displacements
    // x  - Nodal displacements
    // dt - Timestep size
    virtual void update(const Eigen::VectorXd& x, double dt) = 0;

    // Resets the state
    virtual void reset() = 0;

    // Build and return the right-hand-side of schur-complement
    // reduced system of equations
    virtual Eigen::VectorXd rhs() = 0;

    // Gradient of the energy with respect to x 
    virtual Eigen::VectorXd gradient() = 0;

    // Gradient of the energy with respect to mixed variable
    virtual Eigen::VectorXd gradient_mixed() = 0;

    // Left-hand-side of the schur complement system with the mixed
    // variable eliminated
    virtual const Eigen::SparseMatrix<double, Eigen::RowMajor>& lhs() = 0;

    // Given the solution for displacements, solve the updates of the
    // mixed variables
    // dx - nodal displacement deltas
    virtual void solve(const Eigen::VectorXd& dx) = 0;

    // Returns the updates from the mixed solves
    virtual Eigen::VectorXd& delta() = 0;

    // Returns variable values
    virtual Eigen::VectorXd& value() = 0;

    // Returns lagrange multipliers
    virtual Eigen::VectorXd& lambda() = 0;

  protected:

    std::shared_ptr<Mesh> mesh_;

  };

}
