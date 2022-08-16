#include "max_possible_step.h"
#include "Distance.h"
#include "CTCD.h"

using namespace Eigen;

namespace {
  // Distance edge-edge between segments (x0,x1) and (x2,x3)
  // From CTCD::Distance
  template <int DIM>
  double dist_EE(const Vector<double,DIM>& x0, const Vector<double,DIM>& x1,
      const Vector<double,DIM>& x2, const Vector<double,DIM>& x3) {
  
    Vector<double,DIM> d1 = x1 - x0;
    Vector<double,DIM> d2 = x3 - x2;
    Vector<double,DIM> r = x0 - x2;
      
    double a = d1.squaredNorm();
    double e = d2.squaredNorm();
    double f = d2.dot(r);
    double s,t;

    double c = d1.dot(r);
    double b = d1.dot(d2);
    double denom = a*e-b*b;
    if(denom != 0.0)  {
      s = std::clamp((b*f-c*e)/denom, 0.0, 1.0);
    } else {
      //parallel edges and/or degenerate edges; values of s doesn't matter
      s = 0;
    }
    double tnom = b*s + f;
    if(tnom < 0 || e == 0)
      {
        t = 0;
        if(a == 0)
    s = 0;
        else
    s = std::clamp(-c/a, 0.0, 1.0);  
      }
    else if(tnom > e)
      {
        t = 1.0;
        if(a == 0)
    s = 0;
        else
    s = std::clamp( (b-c)/a, 0.0, 1.0);
      }
    else
      t = tnom/e;	    

    Vector<double,DIM> c1 = x0 + s*d1;
    Vector<double,DIM> c2 = x2 + t*d2;

    double p0bary = 1.0-s;
    double p1bary = s;
    double q0bary = 1.0-t;
    double q1bary = t;
    return (c2-c1).norm();
  }
}

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

template <int DIM>
double mfem::additive_ccd(const VectorXd& x, const VectorXd& p,
    const MatrixXi& F) {
    
  using VecD = Vector<double,DIM>;
  double min_step = 1.0;
  double s = 0.1; // scaling factor
  double t_c = 1.0;

  //#pragma omp parallel for reduction(min:min_step)
  for (int i = 0; i < F.rows(); ++i) {
    for (int j = 0; j < F.rows(); ++j) {

      if (F(i,0) == F(j,0) || F(i,0) == F(j,1) || F(i,1) == F(j,0)
          || F(i,1) == F(j,1)) {
        continue;
      }
      // Edge pairs (x0,x1) and (x2,x3) with displacement pairs
      // (p0,p1) and (p2,p3), respectively.
      VecD x0 = x.segment<DIM>(DIM*F(i,0));
      VecD x1 = x.segment<DIM>(DIM*F(i,1));
      VecD x2 = x.segment<DIM>(DIM*F(j,0));
      VecD x3 = x.segment<DIM>(DIM*F(j,1));
      VecD p0 = p.segment<DIM>(DIM*F(i,0));
      VecD p1 = p.segment<DIM>(DIM*F(i,1));
      VecD p2 = p.segment<DIM>(DIM*F(j,0));
      VecD p3 = p.segment<DIM>(DIM*F(j,1));

      double d = dist_EE(x0, x1, x2, x3);
      VecD p_bar = (p0 + p1 + p2 + p3) / 4.0;
      p0 -= p_bar;
      p1 -= p_bar;
      p2 -= p_bar;
      p3 -= p_bar;

      double l_p = std::max(p0.norm(), p1.norm())
                 + std::max(p2.norm(), p3.norm());

      if (l_p <= 1e-16) {
        continue;
      }

      double g = s * d;
      double t = 0.0;
      double t_l = (1.0 - s) * d / l_p;

      bool valid = true;
      int cnt = 0;
      while (true) {
        x0 += t_l * p0;
        x1 += t_l * p1;
        x2 += t_l * p2;
        x3 += t_l * p3;

        d = dist_EE(x0, x1, x2, x3);

        if (t > 0.0 && d < g) {
          break;
        }
        t += t_l;
        if (t > t_c) {
          // false
          valid = false;
          break;
        }
        ++cnt;
        //if (cnt > 10) {
        //  std::cout << "CNT: " << cnt << std::endl;
        //  std::cout << "g: " << g << " t: " << t << " d: " << d << std::endl;
        //  std::cout << " l_p : "<< l_p << " d: " << d << std::endl;
        //}
        t_l = 0.9 * d / l_p;
        //std::cout << "d: " << d << " l_p: " << l_p << std::endl;



      }
      if (valid) {
        // std::cout << "  t: " << t << " d: " << d << std::endl;
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
template double mfem::additive_ccd<3>(const VectorXd& x1,
        const VectorXd& x2, const MatrixXi& F);
template double mfem::additive_ccd<2>(const VectorXd& x1,
        const VectorXd& x2, const MatrixXi& F);        
