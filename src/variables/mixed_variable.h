#pragma once

#include "EigenTypes.h"
#include "variable.h"

namespace mfem {

  class Mesh;

  // Base class for mixed fem variables.
  template<int DIM>
  class MixedVariable : public Variable<DIM> {
  public:
    
    MixedVariable(std::shared_ptr<Mesh> mesh) : Variable<DIM>(mesh) {
    }

    // Evaluate the energy associated with the variable
    // x - Nodal displacements 
    // y - Mixed variable
    virtual double energy(const Eigen::VectorXd& x,
        const Eigen::VectorXd& y) = 0;

    // Evaluate the energy associated with the mixed variable constraint
    // x - Nodal displacements 
    // y - Mixed variable
    virtual double constraint_value(const Eigen::VectorXd& x,
        const Eigen::VectorXd& y) = 0;

    // Update the state given a new set of displacements
    // x  - Nodal displacements
    // dt - Timestep size
    virtual void update(const Eigen::VectorXd& x, double dt) = 0;

    // Gradient of the energy with respect to mixed variable
    virtual Eigen::VectorXd gradient_mixed() = 0;

    // Given the solution for displacements, solve the updates of the
    // mixed variables
    // dx - nodal displacement deltas
    virtual void solve(const Eigen::VectorXd& dx) = 0;

    // Returns lagrange multipliers
    virtual Eigen::VectorXd& lambda() = 0;

    virtual void evaluate_constraint(const Eigen::VectorXd&,
        Eigen::VectorXd&) {}
    virtual void hessian(Eigen::SparseMatrix<double>&) {}
    virtual void jacobian_x(Eigen::SparseMatrix<double>&) {}
    virtual void jacobian_mixed(Eigen::SparseMatrix<double>&) {}

  };

}
