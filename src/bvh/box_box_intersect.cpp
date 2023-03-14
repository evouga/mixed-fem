#include "box_box_intersect.h"
bool box_box_intersect(
  const BoundingBox & A,
  const BoundingBox & B)
{
  // https://gamedev.stackexchange.com/a/913/66665
  for(int c = 0;c<3;c++)
  {
    if(A.max_corner(c) < B.min_corner(c)) return false;
    if(B.max_corner(c) < A.min_corner(c)) return false;
  }
  return true;
}

