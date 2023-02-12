#include "ccd_gpu.h"

#include "ccdgpu/helper.cuh"
#include "ipc/ipc.hpp"
#include "optimizers/optimizer_data.h"
#include "stq/gpu/io.cuh"
#include "stq/gpu/simulation.cuh"
#include "utils/additive_ccd.h"

using namespace Eigen;

namespace {
  #define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

  inline void gpuAssert(cudaError_t code, const char *file, int line,
                        bool abort = true) {
    if (code != cudaSuccess) {
      spdlog::error("GPUassert: {} {} {:d}", cudaGetErrorString(code), file,
                    line);
      if (abort)
        exit(code);
    }
  }

  // Allocates and copies data to GPU
  template <typename T> T *copy_to_gpu(const T *cpu_data, const int size) {
    T *gpu_data;
    gpuErrchk(cudaMalloc((void **)&gpu_data, sizeof(T) * size));
    gpuErrchk(
      cudaMemcpy(gpu_data, cpu_data, sizeof(T) * size, cudaMemcpyHostToDevice));
    return gpu_data;
  }


  template<typename Scalar>
  Scalar compute_toi_strategy(const Eigen::MatrixXd &V0,
                            const Eigen::MatrixXd &V1, const Eigen::MatrixXi &E,
                            const Eigen::MatrixXi &F, int max_iter,
                            Scalar min_distance, Scalar tolerance) {
    mfem::OptimizerData::get().timer.start("CCD-1-boxes");
    std::vector<stq::gpu::Aabb> boxes;
    stq::gpu::constructBoxes(V0, V1, E, F, boxes);
    spdlog::trace("Finished constructing");
    int N = boxes.size();
    int nbox = 0;
    int devcount = 1;
    mfem::OptimizerData::get().timer.stop("CCD-1-boxes");

    stq::gpu::MemHandler *memhandle = new stq::gpu::MemHandler();

    std::vector<std::pair<int, int>> overlaps;
    std::vector<int> result_list;

    // BROADPHASE
    int2 *d_overlaps;
    int *d_count;
    int bpthreads = 32;
    int npthreads = 1024;
    int start_id = 0;
    int limitGB = 0;

    json j;
    ccd::gpu::Record r(j);

    Scalar earliest_toi = 1.0;

    while (N > start_id) {
      mfem::OptimizerData::get().timer.start("CCD-2-broad");
      stq::gpu::run_sweep_sharedqueue(boxes.data(), memhandle, N, nbox, overlaps,
                            d_overlaps, d_count, bpthreads, start_id, devcount,
                            limitGB);
      mfem::OptimizerData::get().timer.stop("CCD-2-broad");

      // copy overlap count
      int count;
      gpuErrchk(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
      spdlog::trace("Count {:d}", count);

      // Allocate boxes to GPU
      stq::gpu::Aabb *d_boxes = copy_to_gpu(boxes.data(), boxes.size());

      spdlog::trace("Copying vertices");
      Scalar *d_vertices_t0 = copy_to_gpu(V0.data(), V0.size());
      Scalar *d_vertices_t1 = copy_to_gpu(V1.data(), V1.size());

      int Vrows = V0.rows();
      assert(Vrows == V1.rows());
      mfem::OptimizerData::get().timer.start("CCD-3-narrowphase");

      ccd::gpu::run_narrowphase(d_overlaps, d_boxes, memhandle, count, d_vertices_t0,
                      d_vertices_t1, Vrows, npthreads, /*max_iter=*/max_iter,
                      /*tol=*/tolerance,
                      /*ms=*/min_distance, /*allow_zero_toi=*/true, result_list,
                      earliest_toi, r);

      if (earliest_toi < 1e-6) {
        ccd::gpu::run_narrowphase(
          d_overlaps, d_boxes, memhandle, count, d_vertices_t0, d_vertices_t1,
          Vrows, npthreads, /*max_iter=*/-1, /*tol=*/tolerance,
          /*ms=*/0.0, /*allow_zero_toi=*/false, result_list, earliest_toi, r);
        earliest_toi *= 0.8;
      }
      mfem::OptimizerData::get().timer.stop("CCD-3-narrowphase");

      gpuErrchk(cudaDeviceSynchronize());

      gpuErrchk(cudaFree(d_count));
      gpuErrchk(cudaFree(d_overlaps));
      gpuErrchk(cudaFree(d_boxes));
      gpuErrchk(cudaFree(d_vertices_t0));
      gpuErrchk(cudaFree(d_vertices_t1));

      gpuErrchk(cudaDeviceSynchronize());
    }
    return earliest_toi;
  }
}

template <typename Scalar, int DIM>
Scalar ipc::accd_gpu(const thrust::device_vector<Scalar>& x1,
    const thrust::device_vector<Scalar>& x2,
    const ipc::CollisionMesh& mesh,
    Scalar dhat) {
  
  int rows = x1.size() / DIM;

  MatrixXd V1(DIM, rows);
  MatrixXd V2(DIM, rows);

  spdlog::set_level(spdlog::level::trace);

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
  int max_iterations = 1e7;

  double alpha = 1.0;
  mfem::OptimizerData::get().timer.start("CCD-TOI");
  double step_size = compute_toi_strategy(
      V1, V2, E, F, max_iterations, min_distance, tolerance);
  mfem::OptimizerData::get().timer.stop("CCD-TOI");

  // std::cout << "V1 \n" << V1 << std::endl;
  // std::cout << "V2 \n" << V2 << std::endl;

  // double step_size = ipc::compute_collision_free_stepsize(mesh, V1, V2);
  if (step_size < 1.0) {
    alpha = 0.8 * step_size;
  }
  return alpha;
}

template <int DIM>
double ipc::additive_ccd(
    const thrust::device_vector<double>& x1,
    const thrust::device_vector<double>& x2,
    const ipc::CollisionMesh& mesh,
    ipc::Candidates& candidates,
    double dhat) {

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

  double step_size = ipc::additive_ccd<DIM>(V1, V2, mesh, candidates, dhat);

  double alpha = 1.0;
  if (step_size < 1.0) {
    alpha = 0.9 * step_size;
  }
    std::cout << " alpha "<< alpha << std::endl;

  return alpha;
}

// Explicit instantiation        
template double ipc::accd_gpu<double, 3>(
    const thrust::device_vector<double>& x1,
    const thrust::device_vector<double>& x2,
    const ipc::CollisionMesh& mesh, double dhat);

template double ipc::additive_ccd<3>(
    const thrust::device_vector<double>& x1,
    const thrust::device_vector<double>& x2,
    const ipc::CollisionMesh& mesh,
    ipc::Candidates& candidates, double dhat);    