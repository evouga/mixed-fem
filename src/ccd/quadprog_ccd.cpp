#include "quadprog_ccd.h"
#include "EigenTypes.h"


#ifdef SIM_USE_MOSEK
#include "igl/mosek/mosek_quadprog.h"
#endif


using namespace Eigen;

namespace mfem { 

template<int DIM>
bool evaluate_certificate(const SpacetimeAABB& b0,
    const SpacetimeAABB& b1) {

}


template<int DIM>
double quadprog_ccd(
    const ipc::CollisionMesh& mesh,
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1) {


}

}

template double mfem::quadprog_ccd<2>(const ipc::CollisionMesh& mesh, const Eigen::MatrixXd& V0, const Eigen::MatrixXd& V1);
template double mfem::quadprog_ccd<3>(const ipc::CollisionMesh& mesh, const Eigen::MatrixXd& V0, const Eigen::MatrixXd& V1);
template bool mfem::evaluate_certificate<2>(const SpacetimeAABB& b0, const SpacetimeAABB& b1);
template bool mfem::evaluate_certificate<3>(const SpacetimeAABB& b0, const SpacetimeAABB& b1);
