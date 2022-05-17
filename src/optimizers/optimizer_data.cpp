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
  timer.reset();
}

void OptimizerData::write() const {
  int sz = energy_residuals_.size();
  MatrixXd mat(sz,3);
}


void OptimizerData::print_data(bool print_timing) const {
  int sz = energies_.size();

  /*std::cout <<
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
  
  if (print_timing) {
    timer.print();
  }*/
}

void Timer::start(const std::string& key) {

  auto it = times_.find(key);

  if (it == times_.end()) {
    times_[key] = std::make_tuple(Time::now(), 0.0, 0);
  } else {
    T& tup = times_[key];
    std::get<0>(tup) = Time::now();
  }
}

void Timer::stop(const std::string& key) {
  auto end = Time::now();
  auto it = times_.find(key);

  if (it == times_.end()) {
    std::cerr << "Invalid timer key: " << key << std::endl;
  } else {
    T& tup = it->second;
    std::chrono::duration<double, std::milli> fp_ms = end - std::get<0>(tup);
    std::get<1>(tup) += fp_ms.count(); // add to total time
    std::get<2>(tup) += 1;             // increment measurement count
  }
}

double Timer::total(const std::string& key) const{
  auto it = times_.find(key);
  if (it != times_.end()) {
    auto p = it->second;
    return std::get<1>(p);
  }
  return 0;
}

double Timer::average(const std::string& key) const {
  auto it = times_.find(key);
  if (it != times_.end()) {
    auto p = it->second;
    return std::get<1>(p) / std::get<2>(p);
  }
  return 0;
}

void Timer::print() const {
  std::cout << "Timing (in ms): " << std::endl;
  auto it = times_.begin();
  while(it != times_.end()) {
    std::string key = it->first;
    const T& tup = it->second;
    double t = std::get<1>(tup);
    int n = std::get<2>(tup);

    std::cout << "  [" << std::setw(10) << key << "] "
        << std::fixed << " Avg: " << std::setw(10) << t/((double) n)
        << "   Total: " << std::setw(10) << t << std::endl;
    ++it;
  }
}