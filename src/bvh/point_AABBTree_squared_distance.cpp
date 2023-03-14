#include "point_AABBTree_squared_distance.h"
#include <queue> // std::priority_queue

bool point_AABBTree_squared_distance(
    const Eigen::RowVector3d & query,
    const std::shared_ptr<AABBTree> & root,
    const double min_sqrd,
    const double max_sqrd,
    double & sqrd,
    std::shared_ptr<Object> & descendant)
{
  sqrd = max_sqrd;
  std::priority_queue< 
    std::pair< double, std::shared_ptr<Object> > ,
    std::vector< std::pair< double, std::shared_ptr<Object> >  >,
    std::greater< std::pair< double, std::shared_ptr<Object> > > >
    Q;

  Q.emplace(point_box_squared_distance(query,root->box),root);
  while(!Q.empty())
  {
    // Pop the queue!
    double box_sqrd = Q.top().first;
    std::shared_ptr<Object> object = Q.top().second;
    Q.pop();
    if(box_sqrd < sqrd)
    {
      // object might be closer
      std::shared_ptr<AABBTree> subtree = 
        std::dynamic_pointer_cast<AABBTree>(object);
      if(subtree)
      {
        if(subtree->left)  Q.emplace( point_box_squared_distance(query,subtree->left->box), subtree->left);
        if(subtree->right) Q.emplace( point_box_squared_distance(query,subtree->right->box), subtree->right);
      }else
      {
        double object_sqrd;
        if(object->point_squared_distance(
          query,min_sqrd,sqrd,object_sqrd,descendant))
        {
          sqrd = object_sqrd;
          descendant = object;
        }
      }
    }
    //std::cout<<box_sqrd<<std::endl;
  }
  return false;
}
