#include "simulator.h"
#include "svd/svd3x3_sse.h"

#include <igl/volume.h>
#include <chrono>

using namespace std::chrono;
using namespace Eigen;
using namespace mfem;

void Simulator::step() {

  
  // Warm start solver
  if (config_->warm_start) {
    object_->warm_start();
  }
  object_->update_gradients();

  int la_steps = 1;//config_->inner_steps;
  double kappa0 = config_->kappa;

  std::cout << "/////////////////////////////////////////////" << std::endl;
  std::cout << "Simulation step " << std::endl;

  for (int ii=0; ii < la_steps; ++ii) {
    // std::cout << "* La step: " << ii << std::endl;
    // std::cout << "-------------------------------------------" << std::endl;

    int i = 0;
    double grad_norm;
    bool ls_done;
    do {
      std::cout << "* Newton step: " << i << std::endl;
      // Do substep on each objects
      auto start = high_resolution_clock::now();
      object_->substep(i==0, grad_norm);
      auto end = high_resolution_clock::now();
      double t1 = duration_cast<nanoseconds>(end-start).count()/1e6;

      start = high_resolution_clock::now();
      ls_done = object_->linesearch();
      end = high_resolution_clock::now();
      double t2 = duration_cast<nanoseconds>(end-start).count()/1e6;

      start = high_resolution_clock::now();
      object_->update_gradients();
      end = high_resolution_clock::now();
      double t3 = duration_cast<nanoseconds>(end-start).count()/1e6;
      std::cout << "  - ! Timing Substep time: " << t1
          << " Linesearch: " << t2
          << " Update gradients: " << t3 << std::endl;
      
      object_->update_lambdas(ii, grad_norm);


      if (ls_done) {
        std::cout << "  - Linesearch done " << std::endl;
        break;
      }
      ++i;
    } while (i < config_->outer_steps && grad_norm > config_->newton_tol);

  }

  //config_->kappa = kappa0;
  object_->update_positions();
}
