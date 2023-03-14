#include "find_all_intersecting_pairs_using_AABBTrees.h"
#include "box_box_intersect.h"
#include <list>

#include "MeshTriangle.h"
#include <iostream>
void find_all_intersecting_pairs_using_AABBTrees(
  const std::shared_ptr<AABBTree> & rootA,
  const std::shared_ptr<AABBTree> & rootB,
  std::vector<std::pair<std::shared_ptr<Object>,std::shared_ptr<Object> > > & 
    leaf_pairs)
{
  std::list< std::pair<std::shared_ptr<Object>,std::shared_ptr<Object> > > Q;
  const auto insert_if = [&Q](
      const std::shared_ptr<Object> & A,
      const std::shared_ptr<Object> & B)
  {
    if(A && B && box_box_intersect(A->box,B->box)) 
    { 
      Q.emplace_back(A,B); 
    }
  };
  insert_if(rootA,rootB);
  while(!Q.empty())
  {
    const std::shared_ptr<Object> nodeA = Q.front().first;
    const std::shared_ptr<Object> nodeB = Q.front().second;
    assert(box_box_intersect(nodeA->box,nodeB->box));
    Q.pop_front();
    std::shared_ptr<AABBTree> subtreeA =
      std::dynamic_pointer_cast<AABBTree>(nodeA);
    std::shared_ptr<AABBTree> subtreeB =
      std::dynamic_pointer_cast<AABBTree>(nodeB);
    if(subtreeA && subtreeB)
    {
      insert_if(subtreeA->left ,subtreeB->left);
      insert_if(subtreeA->right,subtreeB->left);
      insert_if(subtreeA->right,subtreeB->right);
      insert_if(subtreeA->left ,subtreeB->right);
    }else if(subtreeA && !subtreeB)
    {
      insert_if(subtreeA->left ,nodeB);
      insert_if(subtreeA->right,nodeB);
    }else if(subtreeB && !subtreeA)
    {
      insert_if(nodeA,subtreeB->left );
      insert_if(nodeA,subtreeB->right);
    }else // both are not subtrees; both are leaves
    {
      assert(!subtreeA);
      assert(!subtreeB);
      // We already know they overlap because they're in the queue
      assert(box_box_intersect(nodeA->box,nodeB->box));
      leaf_pairs.emplace_back(nodeA,nodeB);
    }
  }
}
