#include "iARAP_gpu.cuh"
// #include "quartic.h"
// #include <cuda/std/complex>
#include "quartic.cuh"
#include "math_constants.h"

using namespace Eigen;

template<typename Scalar> __host__ __device__
void mfem::ComputeEigenvector0(Scalar a00, Scalar a01, Scalar a02, Scalar a11,
                                Scalar a12, Scalar a22, Scalar eval0, Eigen::Matrix<Scalar, 3, 1> &evec0)
{
  typedef Eigen::Matrix<Scalar, 3, 1> Vector;
  Vector row0 = {a00 - eval0, a01, a02};
  Vector row1 = {a01, a11 - eval0, a12};
  Vector row2 = {a02, a12, a22 - eval0};
  Vector r0xr1 = row0.cross(row1);
  Vector r0xr2 = row0.cross(row2);
  Vector r1xr2 = row1.cross(row2);
  Scalar d0 = r0xr1.dot(r0xr1);
  Scalar d1 = r0xr2.dot(r0xr2);
  Scalar d2 = r1xr2.dot(r1xr2);

  Scalar dmax = d0;
  int32_t imax = 0;
  if (d1 > dmax)
  {
    dmax = d1;
    imax = 1;
  }
  if (d2 > dmax)
  {
    imax = 2;
  }

  if (imax == 0)
  {
    evec0 = r0xr1 / std::sqrt(d0);
  }
  else if (imax == 1)
  {
    evec0 = r0xr2 / std::sqrt(d1);
  }
  else
  {
    evec0 = r1xr2 / std::sqrt(d2);
  }
}

template<typename Scalar> __host__ __device__
void mfem::ComputeOrthogonalComplement(
    Eigen::Matrix<Scalar, 3, 1> const &W,
    Eigen::Matrix<Scalar, 3, 1> &U,
    Eigen::Matrix<Scalar, 3, 1> &V) {

  Scalar invLength;
  if (std::fabs(W[0]) > std::fabs(W[1]))
  {
    invLength = (Scalar)1 / std::sqrt(W[0] * W[0] + W[2] * W[2]);
    U = {-W[2] * invLength, (Scalar)0, +W[0] * invLength};
  }
  else
  {
    invLength = (Scalar)1 / std::sqrt(W[1] * W[1] + W[2] * W[2]);
    U = {(Scalar)0, +W[2] * invLength, -W[1] * invLength};
  }
  V = W.cross(U);
}

template<typename Scalar> __host__ __device__
void mfem::ComputeEigenvector1(Scalar a00, Scalar a01, Scalar a02, Scalar a11,
                                Scalar a12, Scalar a22, Eigen::Matrix<Scalar, 3, 1> const &evec0,
                                Scalar eval1, Eigen::Matrix<Scalar, 3, 1> &evec1)
{
  Matrix<Scalar, 3, 1> U, V;
  ComputeOrthogonalComplement(evec0, U, V);

  Matrix<Scalar, 3, 1> AU =
      {
          a00 * U[0] + a01 * U[1] + a02 * U[2],
          a01 * U[0] + a11 * U[1] + a12 * U[2],
          a02 * U[0] + a12 * U[1] + a22 * U[2]};

  Matrix<Scalar, 3, 1> AV =
      {
          a00 * V[0] + a01 * V[1] + a02 * V[2],
          a01 * V[0] + a11 * V[1] + a12 * V[2],
          a02 * V[0] + a12 * V[1] + a22 * V[2]};

  Scalar m00 = U[0] * AU[0] + U[1] * AU[1] + U[2] * AU[2] - eval1;
  Scalar m01 = U[0] * AV[0] + U[1] * AV[1] + U[2] * AV[2];
  Scalar m11 = V[0] * AV[0] + V[1] * AV[1] + V[2] * AV[2] - eval1;

  Scalar absM00 = std::fabs(m00);
  Scalar absM01 = std::fabs(m01);
  Scalar absM11 = std::fabs(m11);
  Scalar maxAbsComp;
  if (absM00 >= absM11)
  {
    maxAbsComp = std::max(absM00, absM01);
    if (maxAbsComp > (Scalar)0)
    {
      if (absM00 >= absM01)
      {
        m01 /= m00;
        m00 = (Scalar)1 / std::sqrt((Scalar)1 + m01 * m01);
        m01 *= m00;
      }
      else
      {
        m00 /= m01;
        m01 = (Scalar)1 / std::sqrt((Scalar)1 + m00 * m00);
        m00 *= m01;
      }
      evec1 = m01 * U - m00 * V;
    }
    else
    {
      evec1 = U;
    }
  }
  else
  {
    maxAbsComp = std::max(absM11, absM01);
    if (maxAbsComp > (Scalar)0)
    {
      if (absM11 >= absM01)
      {
        m01 /= m11;
        m11 = (Scalar)1 / std::sqrt((Scalar)1 + m01 * m01);
        m01 *= m11;
      }
      else
      {
        m11 /= m01;
        m01 = (Scalar)1 / std::sqrt((Scalar)1 + m11 * m11);
        m11 *= m01;
      }
      evec1 = m11 * U - m01 * V;
    }
    else
    {
      evec1 = U;
    }
  }
}

template<typename Scalar> __device__
Matrix<Scalar, 3, 3> mfem::rotation(const Matrix<Scalar, 3, 3> &F) {
  Matrix<Scalar, 9, 9> dRdF;
  return rotation(F, false, dRdF);
}

template<typename Scalar> __device__
Matrix<Scalar, 3, 3> mfem::rotation(const Matrix<Scalar, 3, 3> &F, bool compute_gradients,
    Matrix<Scalar, 9, 9>& dRdF) {

  Scalar i1 = F.squaredNorm();
  Scalar i2 = (F.transpose() * F).squaredNorm();
  Scalar J = F.determinant();
  Scalar a = 0;
  Scalar b = -2 * i1;
  Scalar c = -8 * J;
  Scalar d = i1 * i1 - 2 * (i1 * i1 - i2);

  thrust::complex<Scalar> solutions[4];
  solve_quartic(a, b, c, d, solutions);

  Scalar x1 = solutions[0].real();
  Scalar x2 = solutions[1].real();
  Scalar x3 = solutions[3].real();
  Scalar x4 = solutions[2].real();
  Scalar sig1 = (x1 + x4) / 2;
  Scalar sig2 = (x2 + x4) / 2;
  Scalar sig3 = (x3 + x4) / 2;
  Scalar f = solutions[2].real();
  // in this quartic solver, the solutions can be reduced to the singular values as follows:
  // solutions[0] = s1 - s2 - s3
  // solutions[1] = s2 - s1 - s3
  // solutions[2] = s1 + s2 + s3
  // solutions[3] = s3 - s1 - s2

  if (sig1 > sig2)
  {
    std::swap(sig1, sig2);
  }
  if (sig1 > sig3)
  {
    std::swap(sig1, sig3);
  }
  if (sig2 > sig3)
  {
    std::swap(sig2, sig3);
  }
  Scalar f1 = (2 * f * f + 2 * i1) / (4 * f * f * f - 4 * i1 * f - 8 * J);
  Scalar f2 = -2 / (4 * f * f * f - 4 * i1 * f - 8 * J);
  Scalar fJ = (8 * f) / (4 * f * f * f - 4 * i1 * f - 8 * J);
  Eigen::Matrix<Scalar, 3, 3> g1 = 2 * F;
  Eigen::Matrix<Scalar, 3, 3> g2 = 4 * F * F.transpose() * F;
  Eigen::Matrix<Scalar, 3, 3> gJ;
  gJ.col(0) = F.col(1).cross(F.col(2));
  gJ.col(1) = F.col(2).cross(F.col(0));
  gJ.col(2) = F.col(0).cross(F.col(1));
  Matrix<Scalar, 3, 3> R = f1 * g1 + f2 * g2 + fJ * gJ;
  Matrix<Scalar, 3, 3> S = R.transpose() * F;

  if (compute_gradients) {
    Matrix<Scalar, 3, 1> V0, V1, V2;

    Matrix<Scalar, 3, 3> V;
    Scalar norm = S(0, 1) * S(0, 1) + S(0, 2) * S(0, 2) + S(1, 2) * S(1, 2);
    if (norm > 0)
    {
      Scalar q = (S(0, 0) + S(1, 1) + S(2, 2)) / (Scalar)3;
      Scalar b00 = S(0, 0) - q;
      Scalar b11 = S(1, 1) - q;
      Scalar b22 = S(2, 2) - q;
      Scalar p = std::sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * (Scalar)2) / (Scalar)6);
      Scalar c00 = b11 * b22 - S(1, 2) * S(1, 2);
      Scalar c01 = S(0, 1) * b22 - S(1, 2) * S(0, 2);
      Scalar c02 = S(0, 1) * S(1, 2) - b11 * S(0, 2);
      Scalar det = (b00 * c00 - S(0, 1) * c01 + S(0, 2) * c02) / (p * p * p);

      Scalar halfDet = det * (Scalar)0.5;
      halfDet = std::min(std::max(halfDet, (Scalar)-1), (Scalar)1);

      if (halfDet >= (Scalar)0)
      {
        ComputeEigenvector0(S(0, 0), S(0, 1), S(0, 2), S(1, 1), S(1, 2), S(2, 2), sig3, V2);
        ComputeEigenvector1(S(0, 0), S(0, 1), S(0, 2), S(1, 1), S(1, 2), S(2, 2), V2, sig2, V1);
        V0 = V1.cross(V2);
      }
      else
      {
        ComputeEigenvector0(S(0, 0), S(0, 1), S(0, 2), S(1, 1), S(1, 2), S(2, 2), sig1, V0);
        ComputeEigenvector1(S(0, 0), S(0, 1), S(0, 2), S(1, 1), S(1, 2), S(2, 2), V0, sig2, V1);
        V2 = V0.cross(V1);
      }
      V.col(2) = V0;
      V.col(1) = V1;
      V.col(0) = V2;
    }
    else
    {
      // The matrix is diagonal.
      V = Matrix<Scalar, 3, 3>::Identity();
    }
    Matrix<Scalar, 3, 3> Sigma = Matrix<Scalar, 3, 3>::Identity();
    Sigma(0, 0) = sig3;
    Sigma(1, 1) = sig2;
    Sigma(2, 2) = sig1;

    Matrix<Scalar, 3, 3> U = R * V;
    const Scalar SQRT_HALF = Scalar(CUDART_SQRT_HALF_F);

    Eigen::Matrix<Scalar, 3, 3> T0{
        {0, -1, 0},
        {1, 0, 0},
        {0, 0, 0}};
    T0 = SQRT_HALF * U * T0 * V.transpose();
    Eigen::Matrix<Scalar, 3, 3> T1{
        {0, 0, 0},
        {0, 0, 1},
        {0, -1, 0}};
    T1 = SQRT_HALF * U * T1 * V.transpose();
    Eigen::Matrix<Scalar, 3, 3> T2{
        {0, 0, 1},
        {0, 0, 0},
        {-1, 0, 0}};
    T2 = SQRT_HALF * U * T2 * V.transpose();
    Eigen::Matrix<Scalar, 9, 1> t0, t1, t2;
    t0.block<3, 1>(0, 0) = T0.col(0);
    t0.block<3, 1>(3, 0) = T0.col(1);
    t0.block<3, 1>(6, 0) = T0.col(2);
    t1.block<3, 1>(0, 0) = T1.col(0);
    t1.block<3, 1>(3, 0) = T1.col(1);
    t1.block<3, 1>(6, 0) = T1.col(2);
    t2.block<3, 1>(0, 0) = T2.col(0);
    t2.block<3, 1>(3, 0) = T2.col(1);
    t2.block<3, 1>(6, 0) = T2.col(2);
    Scalar sx = Sigma(0, 0);
    Scalar sy = Sigma(1, 1);
    Scalar sz = Sigma(2, 2);
    Scalar lambda0 = 2 / (sx + sy);
    Scalar lambda1 = 2 / (sz + sy);
    Scalar lambda2 = 2 / (sx + sz);

    if (sx + sy < 2)
      lambda0 = 1;
    if (sz + sy < 2)
      lambda1 = 1;
    if (sx + sz < 2)
      lambda2 = 1;
    //
    //////
    dRdF = lambda0 * (t0 * t0.transpose());
    dRdF += lambda1 * (t1 * t1.transpose());
    dRdF += lambda2 * (t2 * t2.transpose());
  }
  return R;
}

template void mfem::ComputeEigenvector0<float>(float,float,float,float,float,float,float,Eigen::Matrix<float, 3, 1>&);
template void mfem::ComputeEigenvector1<float>(float,float,float,float,float,float, Eigen::Matrix<float, 3, 1> const &, float, Eigen::Matrix<float, 3, 1>&);
template void mfem::ComputeOrthogonalComplement<float>(Eigen::Matrix<float, 3, 1> const&, Eigen::Matrix<float, 3, 1>&, Eigen::Matrix<float, 3, 1>&);
template Eigen::Matrix<float, 3, 3>  mfem::rotation<float>(const Eigen::Matrix<float, 3, 3>&);
template Eigen::Matrix<float, 3, 3>  mfem::rotation<float>(const Eigen::Matrix<float, 3, 3>&, bool, Eigen::Matrix<float, 9, 9>&);


template void mfem::ComputeEigenvector0<double>(double,double,double,double,double,double,double,Eigen::Matrix<double, 3, 1>&);
template void mfem::ComputeEigenvector1<double>(double,double,double,double,double,double, Eigen::Matrix<double, 3, 1> const &, double, Eigen::Matrix<double, 3, 1>&);
template void mfem::ComputeOrthogonalComplement<double>(Eigen::Matrix<double, 3, 1> const&, Eigen::Matrix<double, 3, 1>&, Eigen::Matrix<double, 3, 1>&);
template Eigen::Matrix<double, 3, 3>  mfem::rotation<double>(const Eigen::Matrix<double, 3, 3>&);
template Eigen::Matrix<double, 3, 3>  mfem::rotation<double>(const Eigen::Matrix<double, 3, 3>&, bool, Eigen::Matrix<double, 9, 9>&);