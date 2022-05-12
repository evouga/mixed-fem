#include "optimizer_data.h"
#include <igl/writeDMAT.h>
#include <iostream>
#include <iomanip>      // std::setw

using namespace mfem;
using namespace Eigen;

void OptimizerData::clear() {
  energy_residuals_.clear();
  energies_.clear();
  egrad_.clear();
  egrad_x_.clear();
  egrad_s_.clear();
  egrad_la_.clear();
}

void OptimizerData::write() const {
  int sz = energy_residuals_.size();
  MatrixXd mat(sz,3);
}


void OptimizerData::print_data() const {
  int sz = energies_.size();

  std::cout <<
        "┌──────┬────────────┬────────────┬────────────┬──────────────┬──────────────┬──────────────┐\n"
        "│ Iter │   Energy   │ Energy Res │   ||RHS||  │ ||grad_x E|| │ ||grad_s E|| │ ||grad_l E|| │ \n"
        "├──────┼────────────┼────────────┼────────────┼──────────────┼──────────────┼──────────────┤\n"
        ;
  for (int i = 0; i < sz; ++i) {
    std::cout << "│ " << std::setw(4) << i << " │ "
      << std::scientific << std::setprecision(4) << std::setw(5) << energies_[i] << " │ "
      << std::scientific << std::setprecision(4) << std::setw(5) << energy_residuals_[i] << " │ "
      << std::scientific << std::setprecision(4) << std::setw(5) << egrad_[i] << " │ "
      << std::scientific << std::setw(12) << egrad_x_[i] << " │ "
      << std::scientific << std::setw(12) << egrad_s_[i] << " │ "
      << std::scientific << std::setw(12) << egrad_la_[i] << " │\n";
  }
  std::cout <<
        "└──────┴────────────┴────────────┴────────────┴──────────────┴──────────────┴──────────────┘\n";

}