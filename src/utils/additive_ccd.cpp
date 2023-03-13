#include "additive_ccd.h"
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/point_triangle.hpp>

using namespace Eigen;

namespace {

  // Compute Additive CCD value for primitive pair.
  // x1        - primitive 1 initial positions
  // x2        - primitive 2 initial positions
  // p1        - primitive 1 proposed direction
  // p2        - primitive 2 proposed direction
  // dist_func - distance function for primitive pair
  // t         - final CCD estime (0, 1]
  //
  // returns whether primitives intersect
  template <typename Derived1, typename Derived2, typename DistanceFunc>
  bool accd_primitive(
      MatrixBase<Derived1>& x1, MatrixBase<Derived2>& x2,
      MatrixBase<Derived1>& p1, MatrixBase<Derived2>& p2,
      DistanceFunc dist_func, double& t) {
    
    using VecD = Matrix<double, 1, Derived1::ColsAtCompileTime>;
    
    double s = 0.1; // scaling factor
    double t_c = 1.0;

    // Average the deltas
    VecD p_bar = (p1.colwise().sum() + p2.colwise().sum())
               / (p1.rows() + p2.rows());

    // Subtract off the average
    p1.rowwise() -= p_bar;
    p2.rowwise() -= p_bar;

    double l_p = p1.rowwise().norm().maxCoeff()
               + p2.rowwise().norm().maxCoeff();

    if (l_p <= 1e-12) {
      return false;
    }

    bool valid = true;

    double d = dist_func(x1, x2);
    double g = s * d;
    t = 0.0;
    double t_l = (1.0 - s) * d / l_p;

    int cnt = 0;
    while (true) {
      x1 += t_l * p1;
      x2 += t_l * p2;

      d = dist_func(x1, x2);

      if (t > 0.0 && d < g) {
        break;
      }
      t += t_l;
      if (t > t_c) {
        valid = false;
        break;
      }
      t_l = 0.9 * d / l_p;
      if(++cnt > 1000) {
        // std::cout << "t: " << t << " d: " << d << " t_l" << t_l << " l_p: " << l_p << std::endl;
        // std::cout << "CNT > 1000" << cnt << std::endl;
        break;
      }
    }
    return valid;
  }
}

template <int DIM>
double ipc::additive_ccd(const VectorXd& x, const VectorXd& p,
    const ipc::CollisionMesh& mesh, ipc::Candidates& candidates,
    double dhat) {
    
  double min_step = 1.0;
  double s = 0.1; // scaling factor
  double t_c = 1.0;

  MatrixXd V1 = Map<const MatrixXd>(x.data(), DIM, x.size() / DIM);
  V1.transposeInPlace();

  VectorXd x2 = x + p;
  MatrixXd V2 = Map<const MatrixXd>(x2.data(), DIM, x.size() / DIM);
  V2.transposeInPlace();

  V1 = mesh.vertices(V1);
  V2 = mesh.vertices(V2);
  MatrixXd P = V2-V1;

  candidates.clear();
  ipc::construct_collision_candidates(mesh, V1, V2, candidates, dhat / 2.0);//,
      //ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE);
  // std::cout << "Construct collision candidates 2" << std::endl;


  const Eigen::MatrixXi& E = mesh.edges();
  const Eigen::MatrixXi& F = mesh.faces();
  const Eigen::MatrixXi& F2E = mesh.faces_to_edges();

  // std::cout << "N canddiates: " << candidates.size() << std::endl;
  // std::cout << "EE candidates: " << candidates.ee_candidates.size() << std::endl;
  // std::cout << "EV candidates: " << candidates.ev_candidates.size() << std::endl;
  // std::cout << "FV candidates: " << candidates.fv_candidates.size() << std::endl;

  // Edge-vertex distance checks
  if constexpr (DIM == 2) {

    #pragma omp parallel for reduction(min : min_step)
    for (int i = 0; i < candidates.ev_candidates.size(); ++i) {
      const auto& ev_candidate = candidates.ev_candidates[i];
      const auto& [ei, vi] = ev_candidate;
      long e0i = E(ei, 0), e1i = E(ei, 1);

      auto dist = [](const Matrix<double, 1, DIM>& a,
                    const Matrix<double, 2, DIM>& b) {

        PointEdgeDistanceType dtype = point_edge_distance_type(
          a.row(0), b.row(0), b.row(1));

        DistanceMode dmode = DistanceMode::SQRT;

        double distance = point_edge_distance(
            a.row(0), b.row(0), b.row(1), dtype, dmode);

        return distance;
      };

      Matrix<double, 1, DIM> x_a, p_a;
      Matrix<double, 2, DIM> x_b, p_b;
      x_a.row(0) = V1.row(vi);
      x_b.row(0) = V1.row(e0i);
      x_b.row(1) = V1.row(e1i);
      p_a.row(0) = P.row(vi);
      p_b.row(0) = P.row(e0i);
      p_b.row(1) = P.row(e1i);
      double t;

      if (accd_primitive(x_a, x_b, p_a, p_b, dist, t)) {
        min_step = std::min(min_step,t);
      }
    }

  } else {

    // std::cout << "EDGE EDGE " << std::endl;
    // Edge-edge distance checks
    #pragma omp parallel for reduction(min : min_step)
    for (int i = 0; i < candidates.ee_candidates.size(); ++i) {
      const auto& ee_candidate = candidates.ee_candidates[i];
      const auto& [eai, ebi] = ee_candidate;
      long ea0i = E(eai, 0), ea1i = E(eai, 1);
      long eb0i = E(ebi, 0), eb1i = E(ebi, 1);

      auto dist = [](const Matrix<double, 2, DIM>& a,
                    const Matrix<double, 2, DIM>& b) {
        EdgeEdgeDistanceType dtype = edge_edge_distance_type(
            a.row(0), a.row(1), b.row(0), b.row(1));

        DistanceMode dmode = DistanceMode::SQRT;
        double distance = edge_edge_distance(
            a.row(0), a.row(1), b.row(0), b.row(1), dtype, dmode);

        return distance;
      };

      Matrix<double, 2, DIM> x_a, x_b, p_a, p_b;
      x_a.row(0) = V1.row(ea0i);
      x_a.row(1) = V1.row(ea1i);
      x_b.row(0) = V1.row(eb0i);
      x_b.row(1) = V1.row(eb1i);
      p_a.row(0) = P.row(ea0i);
      p_a.row(1) = P.row(ea1i);
      p_b.row(0) = P.row(eb0i);
      p_b.row(1) = P.row(eb1i);
      double t;
      if (accd_primitive(x_a, x_b, p_a, p_b, dist, t)) {
        min_step = std::min(min_step,t);
      }
    }

    // std::cout << "Face Vertex" << std::endl;
    // Face-vertex distance checks
    #pragma omp parallel for reduction(min : min_step)
    for (int i = 0; i < candidates.fv_candidates.size(); ++i) {
      const auto& fv_candidate = candidates.fv_candidates[i];
      const auto& [fi, vi] = fv_candidate;
      long f0i = F(fi, 0), f1i = F(fi, 1), f2i = F(fi, 2);

      auto dist = [](const Matrix<double, 1, DIM>& a,
                    const Matrix<double, 3, DIM>& b) {
        // Compute distance type
        PointTriangleDistanceType dtype = point_triangle_distance_type(
            a.row(0), b.row(0), b.row(1), b.row(2));

        DistanceMode dmode = DistanceMode::SQRT;

        double distance = point_triangle_distance(
            a.row(0), b.row(0), b.row(1), b.row(2), dtype, dmode);

        return distance;
      };

      Matrix<double, 1, DIM> x_a, p_a;
      Matrix<double, 3, DIM> x_b, p_b;
      x_a.row(0) = V1.row(vi);
      x_b.row(0) = V1.row(f0i);
      x_b.row(1) = V1.row(f1i);
      x_b.row(2) = V1.row(f2i);
      p_a.row(0) = P.row(vi);
      p_b.row(0) = P.row(f0i);
      p_b.row(1) = P.row(f1i);
      p_b.row(2) = P.row(f2i);
      double t;
      if (accd_primitive(x_a, x_b, p_a, p_b, dist, t)) {
        min_step = std::min(min_step,t);
      }
    }
  }
  
  return min_step;
}

// Explicit instantiation for 2D/3D          
template double ipc::additive_ccd<3>(const VectorXd& x1,
        const VectorXd& x2, const ipc::CollisionMesh& mesh,
        ipc::Candidates& candidates, double dhat);
template double ipc::additive_ccd<2>(const VectorXd& x1,
        const VectorXd& x2, const ipc::CollisionMesh& mesh,
        ipc::Candidates& candidates, double dhat);        
