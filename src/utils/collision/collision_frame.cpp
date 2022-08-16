#include "collision_frame.h"
#include "point_edge_frame.h"

using namespace mfem;

template<> template<> std::unique_ptr<CollisionFrame<2>>
CollisionFrame<2>::make_collision_frame<Eigen::Vector3i,POINT_EDGE>(
    const Eigen::VectorXd& x, const Eigen::MatrixBase<Eigen::Vector3i>& E) {

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