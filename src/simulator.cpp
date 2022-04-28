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

  int la_steps = config_->inner_steps;
  
  std::cout << "/////////////////////////////////////////////" << std::endl;
  std::cout << "Simulation step " << std::endl;

  for (int ii=0; ii < la_steps; ++ii) {
    std::cout << "* La step: " << ii << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    for (int i = 0; i < config_->outer_steps; ++i) {
      
      std::cout << "* Newton step: " << i << std::endl;
      //object_->update_gradients();
      
      for (int j = 0; j < 1; ++j) {
        // Do substep on each objects
        object_->substep(i==0 && j==0);
      }

      object_->linesearch();
      object_->update_gradients();

    }
    object_->update_lambdas(ii);
  }

  object_->update_positions();
}
