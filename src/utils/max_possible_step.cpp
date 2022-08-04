#include "max_possible_step.h"
#include "Distance.h"
#include "CTCD.h"

using namespace Eigen;

template <int DIM>
double mfem::max_possible_step(const VectorXd& x1, const VectorXd& x2,
    const MatrixXi& F) {

  double min_step = 1.0;
  //double eta0 = 1e-8;
  double eta0 = 0.1;
  //std::cout << " max_possible step: " << F << std::endl;

  #pragma omp parallel for reduction(min:min_step)
  for (int i = 0; i < F.rows(); ++i) {
    for (int j = 0; j < F.rows(); ++j) {

      if (F(i,0) == F(j,0) || F(i,0) == F(j,1) || F(i,1) == F(j,0)
          || F(i,1) == F(j,1)) {
        continue;
      }
      const Vector2d& p0_2d_start = x1.segment<2>(2*F(i,0));
      const Vector2d& p1_2d_start = x1.segment<2>(2*F(i,1));
      const Vector2d& q0_2d_start = x1.segment<2>(2*F(j,0));
      const Vector2d& q1_2d_start = x1.segment<2>(2*F(j,1));
      const Vector2d& p0_2d_end = x2.segment<2>(2*F(i,0));
      const Vector2d& p1_2d_end = x2.segment<2>(2*F(i,1));
      const Vector2d& q0_2d_end = x2.segment<2>(2*F(j,0));
      const Vector2d& q1_2d_end = x2.segment<2>(2*F(j,1));

      Vector3d p0start(p0_2d_start(0), p0_2d_start(1), 0);
      Vector3d p1start(p1_2d_start(0), p1_2d_start(1), 0);
      Vector3d q0start(q0_2d_start(0), q0_2d_start(1), 0);
      Vector3d q1start(q1_2d_start(0), q1_2d_start(1), 0);
      Vector3d p0end(p0_2d_end(0), p0_2d_end(1), 0);
      Vector3d p1end(p1_2d_end(0), p1_2d_end(1), 0);
      Vector3d q0end(q0_2d_end(0), q0_2d_end(1), 0);
      Vector3d q1end(q1_2d_end(0), q1_2d_end(1), 0);
      double t = 1;
      double d_sqrt;
      double tmp;
      d_sqrt = Distance::edgeEdgeDistance(p0start,p1start,q0start,
          q1start,tmp,tmp,tmp,tmp).norm();
      double eta = d_sqrt * eta0;
      //std::cout << "D_SQRT: " << d_sqrt << std::endl;
      if (CTCD::edgeEdgeCTCD(p0start,p1start,q0start,q1start,
            p0end,p1end,q0end,q1end,eta,t)) {
        if (t <= 1e-6) {
          t = 1;
          // Try again with minimum tolerance, if nothing hit, just set
          // to maximum step size 
          if (!CTCD::edgeEdgeCTCD(p0start,p1start,q0start,q1start,
                      p0end,p1end,q0end,q1end,1e-12,t)) {
            //t = 1;
          }
        } 
        // If t is 0, then we (problably) have parallel edge failure,
        // so set to max step size. Yolo
        t = (t < 1e-12) ? 1 : t;
        min_step = std::min(min_step,t);
      }
    }
  }
  return min_step;
}

// Explicit instantiation for 2D/3D
template double mfem::max_possible_step<3>(const VectorXd& x1,
        const VectorXd& x2, const MatrixXi& F);
template double mfem::max_possible_step<2>(const VectorXd& x1,
        const VectorXd& x2, const MatrixXi& F);
