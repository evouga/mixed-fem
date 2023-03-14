#include "ray_intersect_box.h"
#include <iostream>

bool ray_intersect_box(
  const Ray & ray,
  const BoundingBox& box,
  const double min_t,
  const double max_t)
{
  // Consider each dimension
  Eigen::Vector3d tmin;
  Eigen::Vector3d tmax;
  for(int c = 0;c<3;c++)
  {
    const double & xmin = box.min_corner(c);
    const double & xmax = box.max_corner(c);
    const double & xe = ray.origin(c);
    const double & xd = ray.direction(c);
    const double a = 1.0/xd;
    if(a>=0)
    {
      tmin(c) = a*(xmin-xe);
      tmax(c) = a*(xmax-xe);
    }else         
    {             
      // flip sign
      tmax(c) = a*(xmin-xe);
      tmin(c) = a*(xmax-xe);
    }
  }
  // Now we'd like to know if the interval [min_t,max_t] intersects
  // [tmin(0),tmax(0] ∩ [tmin(1),tmax(1] ∩ [tmin(2),tmax(2]
  //
  // This is just 
  // (([min_t,max_t] ∩ [tmin(0),tmax(0]) ∩ [tmin(1),tmax(1)]) ∩ [tmin(2),tmax(2)]
  const double out_min = std::max(min_t,tmin.maxCoeff());
  const double out_max = std::min(max_t,tmax.minCoeff());
  if(out_max < out_min)
  {
    return false;
  }
  return true;
}
