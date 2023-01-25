#include "ccd_gpu.h"

#include "ccdgpu/helper.cuh"

using namespace Eigen;

template <typename Scalar, int DIM>
Scalar ipc::accd_gpu(const thrust::device_vector<Scalar>& x1,
    const thrust::device_vector<Scalar>& x2,
    const ipc::CollisionMesh& mesh,
    Scalar dhat) {
  
  int rows = x1.size() / DIM;

  MatrixXd V1(DIM, rows);
  MatrixXd V2(DIM, rows);

  // Copy x_full_ to V1 and x2_full_ to V2
  cudaMemcpy(V1.data(), thrust::raw_pointer_cast(x1.data()), 
      x1.size() * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(V2.data(), thrust::raw_pointer_cast(x2.data()), 
      x2.size() * sizeof(double), cudaMemcpyDeviceToHost);
  V1.transposeInPlace();
  V2.transposeInPlace();

  V1 = mesh.vertices(V1);
  V2 = mesh.vertices(V2);
  const Eigen::MatrixXi& E = mesh.edges();
  const Eigen::MatrixXi& F = mesh.faces();

  double min_distance = 0.0;
  double tolerance = 1e-6;
  double max_iterations = 1e7;

  double alpha = 1.0;
  // const double step_size = ccd::gpu::compute_toi_strategy(
  //     V1, V2, E, F, max_iterations, min_distance, tolerance);
  // if (step_size < 1.0) {
  //   alpha = 0.8 * step_size;
  // }

}

// Explicit instantiation        
template double ipc::accd_gpu<double, 3>(
    const thrust::device_vector<double>& x1,
    const thrust::device_vector<double>& x2,
    const ipc::CollisionMesh& mesh, double dhat);