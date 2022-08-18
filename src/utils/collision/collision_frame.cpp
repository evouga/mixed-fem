#include "collision_frame.h"
#include "point_edge_frame.h"
#include "point_point_frame.h"

using namespace mfem;
using namespace Eigen;

template<> template<> std::unique_ptr<CollisionFrame<2>>
CollisionFrame<2>::make_collision_frame<Vector3i,POINT_EDGE>(
    const VectorXd& x, const MatrixBase<Vector3i>& E) {
  using VecD = Vector<double,2>;
  const VecD& a = x.segment<2>(2*E(0));
  const VecD& b = x.segment<2>(2*E(1));
  const VecD& p = x.segment<2>(2*E(2));
  VecD v = b - a;
  VecD w = p - a;
  // Check for Point-Point frame with {p, a}
  double c1 = w.dot(v);
  if (c1 <= 0) {
    return std::make_unique<PointPointFrame<2>>(E(0),E(2));
  } else { 
    double c2 = v.squaredNorm();
    // Check for Point-Point with {p,b}
    if (c2 <= c1) {
      return std::make_unique<PointPointFrame<2>>(E(1),E(2));
    } else {
      return std::make_unique<PointEdgeFrame<2>>(E);
    }
  }
}

template<> template<> std::unique_ptr<CollisionFrame<3>>
CollisionFrame<3>::make_collision_frame<Eigen::Vector4i,POINT_TRIANGLE>(
    const Eigen::VectorXd& x, const Eigen::MatrixBase<Eigen::Vector4i>& E) {

}

template<> template<> std::unique_ptr<CollisionFrame<3>>
CollisionFrame<3>::make_collision_frame<Eigen::Vector4i,EDGE_EDGE>(
    const Eigen::VectorXd& x, const Eigen::MatrixBase<Eigen::Vector4i>& E) {

}

template class mfem::CollisionFrame<2>;
template class mfem::CollisionFrame<3>;

// template std::unique_ptr<CollisionFrame<2>> 
// CollisionFrame<2>::make_collision_frame<Eigen::Vector3d, POINT_EDGE>(
//     const Eigen::VectorXd& x, const Eigen::MatrixBase<Eigen::Vector3d>& E);
// template std::unique_ptr<CollisionFrame<2,3>> 
// CollisionFrame<2,3>::make_collision_frame<Eigen::Vector3d, POINT_EDGE>(
//     const Eigen::VectorXd& x, const Eigen::MatrixBase<Eigen::Vector3d>& E);
