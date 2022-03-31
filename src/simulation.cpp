#include "simulation.h"

using namespace Eigen;
using namespace mfem;

void Simulation::build_rhs() {
  int sz = qt_.size() + T_.rows()*9;
  rhs_.resize(sz);
  rhs_.setZero();

  // Positional forces 
//   rhs_.segment(0, qt_.size()) = f_ext_ + ih2*M_*(q0_ - q1_);

//   // Lagrange multiplier forces
//   #pragma omp parallel for
//   for (int i = 0; i < meshT.rows(); ++i) {
//     // 1. W * st term +  - W * Hinv * g term
//     //rhs.segment(qt.size() + 9*i, 9) = vols(i) * arap_rhs(R[i]);
//     //rhs.segment(qt.size() + 9*i, 9) = vols(i) * corotational_rhs(R[i],
//     //    S[i], mu, lambda);
//     rhs_.segment(qt.size() + 9*i, 9) = vols(i) * neohookean_rhs(R[i],
//         S[i], Hinv[i], g[i]);
//   }

//   // 3. Jacobian term
//   rhs_.segment(qt.size(), 9*meshT.rows()) -= Jw*(P.transpose()*qt+b);
}

void Simulation::init() {

}