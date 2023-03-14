#include "AABBTree.h"
#include "insert_box_into_box.h"
#include <iostream>
#include <string>

AABBTree::AABBTree(
  const std::vector<std::shared_ptr<Object> > & objects,
  int a_depth): 
  depth(std::move(a_depth)), 
  num_leaves(objects.size())
{
  for(const auto & object : objects)
  {
    insert_box_into_box(object->box,box);
  }
  switch(objects.size())
  {
    default:
    {
      std::vector<std::shared_ptr<Object> > left_objects;
      std::vector<std::shared_ptr<Object> > right_objects;
      int split_axis;
      (box.max_corner - box.min_corner).maxCoeff(&split_axis);
      const double split = 
        0.5*(box.max_corner(split_axis) + box.min_corner(split_axis));
      for(const auto & object : objects)
      {
        if(object->box.center()(split_axis) < split)
        {
          left_objects.emplace_back(object);
        }else
        {
          right_objects.emplace_back(object);
        }
      }
      // Huh. There's a failure mode where a single long triangle determines
      // the bounds of the entire box and it and everything else goes to one
      // side.
      if(left_objects.size() == 0)
      {
        left_objects.emplace_back(right_objects.back());
        right_objects.pop_back();
      }
      if(right_objects.size() == 0)
      {
        right_objects.emplace_back(left_objects.back());
        left_objects.pop_back();
      }
      left = std::make_shared<AABBTree>( left_objects , depth+1);
      right = std::make_shared<AABBTree>( right_objects, depth+1 );
      break;
    }
    case 2:
      right = objects[1];
      // Fall through
    case 1:
      left = objects[0];
      // Fall through
  }
};
