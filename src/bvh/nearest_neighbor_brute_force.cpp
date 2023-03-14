#include "nearest_neighbor_brute_force.h"
#include <limits>// std::numeric_limits<double>::infinity();

void nearest_neighbor_brute_force(
  const Eigen::MatrixXd & points,
  const Eigen::RowVector3d & query,
  int & I,
  double & sqrD)
{
  sqrD = std::numeric_limits<double>::infinity();
  for(int p = 0;p<points.rows();p++)
  {
    const double psqrD = (query-points.row(p)).squaredNorm();
    if(psqrD < sqrD)
    {
      sqrD = psqrD;
      I = p;
    }
  }
}

