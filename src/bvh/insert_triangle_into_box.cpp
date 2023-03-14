#include "insert_triangle_into_box.h"

void insert_triangle_into_box(
  const Eigen::RowVector3d & a,
  const Eigen::RowVector3d & b,
  const Eigen::RowVector3d & c,
  BoundingBox & B)
{
  B.min_corner = B.min_corner.array().min(a.array());
  B.max_corner = B.max_corner.array().max(a.array());
  B.min_corner = B.min_corner.array().min(b.array());
  B.max_corner = B.max_corner.array().max(b.array());
  B.min_corner = B.min_corner.array().min(c.array());
  B.max_corner = B.max_corner.array().max(c.array());
}


