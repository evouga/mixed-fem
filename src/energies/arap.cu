// #include "arap.cuh"

// using namespace Eigen;

// template <typename T>
// using Vec6 = Matrix<T, 6, 1>;

// template <typename T>
// using Mat6 = Matrix<T, 6, 6>;

// namespace mfem {

//   template <typename Scalar> __device__
//   Scalar local_energy(Vec6<Scalar>& S, Scalar mu) {
//     return (mu*(pow(S(0)-1.0,2.0)+pow(S(1)-1.0,2.0)+pow(S(2)-1.0,2.0)
//         +(S(3)*S(3))*2.0+(S(4)*S(4))*2.0+(S(5)*S(5))*2.0))/2.0;
//   }
  
//   template <typename Scalar> __device__
//   Vec6<Scalar> local_gradient(const Vec6<Scalar>& S, Scalar mu) {
//     Vec6<Scalar> g;
//     g(0) = (mu*(S(0)*2.0-2.0))/2.0;
//     g(1) = (mu*(S(1)*2.0-2.0))/2.0;
//     g(2) = (mu*(S(2)*2.0-2.0))/2.0;
//     g(3) = S(3)*mu*2.0;
//     g(4) = S(4)*mu*2.0;
//     g(5) = S(5)*mu*2.0;
//     return g;    
//   }

//   template <typename Scalar> __device__
//   Mat6<Scalar> local_hessian(const Vec6<Scalar>& S, Scalar mu) {
//     Vec6<Scalar> tmp; tmp << 1,1,1,2,2,2;
//     Mat6<Scalar> tmp2; tmp2.setZero();
//     tmp2.diagonal() = tmp;
//     return tmp2 * mu;
//   }

//   template float local_energy<float, 6>(const Eigen::Matrix<float, 6, 1>&, float);
//   template Eigen::Matrix<float, 6, 1> local_gradient<float, 6>(const Eigen::Matrix<float, 6, 1>&, float);
//   template Eigen::Matrix<float, 6, 6> local_hessian<float, 6>(const Eigen::Matrix<float, 6, 1>&, float);

//   template double local_energy<double, 6>(const Eigen::Matrix<double, 6, 1>&, double);
//   template Eigen::Matrix<double, 6, 1> local_gradient<double, 6>(const Eigen::Matrix<double, 6, 1>&, double);
//   template Eigen::Matrix<double, 6, 6> local_hessian<double, 6>(const Eigen::Matrix<double, 6, 1>&, double);

// }
