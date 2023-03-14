#include "AABBTree.h"

bool AABBTree::ray_intersect(
  const Ray& ray,
  const double min_t,
  const double max_t,
  double & t,
  std::shared_ptr<Object> & descendant) const 
{
  // Does ray hit this box?
  {
    if(!ray_intersect_box( ray,box,min_t,max_t))
    {
      return false;
    }
  }
  t = max_t;
  // Does ray hit children?
  double left_t;
  std::shared_ptr<Object> left_descendant;
  if(left && left->ray_intersect(ray,min_t,t,left_t,left_descendant))
  {
    // Left hit!
    t = left_t;
    // If a descendant wasn't returned then left was already a leaf...
    //
    // This is ugly, we're adding ad hoc shit to account for not knowing
    // whether left is a tree or a leaf... All this just to use the same
    // ->ray_intersect call above... gross...
    if(left_descendant)
    {
      descendant = left_descendant;
    }else
    {
      descendant = left;
    }
  }
  double right_t;
  std::shared_ptr<Object> right_descendant;
  if(right && right->ray_intersect(ray,min_t,/**/t,right_t,right_descendant))
  {
    // Even better right hit!
    t = right_t;
    if(right_descendant)
    {
      descendant = right_descendant;
    }else
    {
      descendant = right;
    }
  }
  return !!descendant;
}

