#include "point_box_squared_distance.h"
#include <iostream>

double point_box_squared_distance(
  const Eigen::RowVector3d & query,
  const BoundingBox & box)
{
  Eigen::RowVector3d seg(0,0,0);
  seg = seg.array().max((box.min_corner - query).array()).eval();
  seg = seg.array().max((query-box.max_corner).array()).eval();
  return seg.squaredNorm();
}
