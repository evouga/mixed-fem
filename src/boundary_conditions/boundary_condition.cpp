#include "boundary_condition.h"

using namespace mfem;

BoundaryCondition::BoundaryCondition(Eigen::MatrixXd& V,
    const BoundaryConditionConfig& config) : config_(config) {

  bbox.setZero();
  int cols = V.cols();
  bbox.block(0,0,1,cols) = V.row(0);
  bbox.block(1,0,1,cols) = V.row(0);
  for(int i = 1; i < V.rows(); i++) {
    const Eigen::RowVectorXd& v = V.row(i);
    for(int d = 0; d < cols; d++) {
      if(v[d] < bbox(0, d)) {
          bbox(0, d) = v[d];
      }
      if(v[d] > bbox(1, d)) {
          bbox(1, d) = v[d];
      }
    }
  }

}