#include "insert_box_into_box.h"
void insert_box_into_box(
  const BoundingBox & A,
  BoundingBox & B)
{
  B.min_corner = B.min_corner.array().min(A.min_corner.array());
  B.max_corner = B.max_corner.array().max(A.max_corner.array());
}

