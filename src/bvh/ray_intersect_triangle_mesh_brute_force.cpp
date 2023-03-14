#include "ray_intersect_triangle_mesh_brute_force.h"
#include "ray_intersect_triangle.h"
#include <limits> // std::numeric_limits<double>::infinity
#include <cmath> // std::isfinite

bool ray_intersect_triangle_mesh_brute_force(
  const Ray & ray,
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  const double min_t,
  const double max_t,
  double & hit_t,
  int & hit_f)
{
  // initialize 
  hit_t = std::numeric_limits<double>::infinity();
  // Loop over triangles
  for(int f = 0;f < F.rows(); f++)
  {
    double ft;
    if(ray_intersect_triangle(
      ray,
      V.row(F(f,0)), 
      V.row(F(f,1)), 
      V.row(F(f,2)),
      min_t,
      hit_t,
      ft))
    {
      hit_t = ft;
      hit_f = f;
    }
  }
  return std::isfinite(hit_t);
}
