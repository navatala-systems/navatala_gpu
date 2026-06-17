// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `cuda_sparse_mg` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_cuda_navatala_sparse_mg_restrict_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_mg_restrict_f32(const unsigned int* RrowPtr, const unsigned int* RcolIdx, const float* Rvalues, const float* rFine, const unsigned int* nCoarse, float* rCoarse) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nCoarse[0]));
  if ((row < N)) {
    int rs = ((int)(RrowPtr[row]));
    int re = ((int)(RrowPtr[(row + 1)]));
    float acc = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(RcolIdx[k]));
      float rv = Rvalues[k];
      float rf = rFine[col];
      acc = (acc + (rv * rf));
    }
    rCoarse[row] = acc;
  }
}

)kernel";
const char* k_cuda_navatala_sparse_mg_restrict_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_mg_restrict_f64(const unsigned int* RrowPtr, const unsigned int* RcolIdx, const double* Rvalues, const double* rFine, const unsigned int* nCoarse, double* rCoarse) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nCoarse[0]));
  if ((row < N)) {
    int rs = ((int)(RrowPtr[row]));
    int re = ((int)(RrowPtr[(row + 1)]));
    double acc = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(RcolIdx[k]));
      double rv = Rvalues[k];
      double rf = rFine[col];
      acc = (acc + (rv * rf));
    }
    rCoarse[row] = acc;
  }
}

)kernel";
const char* k_cuda_navatala_sparse_mg_prolongate_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_mg_prolongate_f32(const unsigned int* ProwPtr, const unsigned int* PcolIdx, const float* Pvalues, const float* eCoarse, const unsigned int* nFine, float* xFine) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nFine[0]));
  if ((row < N)) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    float pe = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(PcolIdx[k]));
      float pv = Pvalues[k];
      float ec = eCoarse[col];
      pe = (pe + (pv * ec));
    }
    float xold = xFine[row];
    xFine[row] = (xold + pe);
  }
}

)kernel";
const char* k_cuda_navatala_sparse_mg_prolongate_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_mg_prolongate_f64(const unsigned int* ProwPtr, const unsigned int* PcolIdx, const double* Pvalues, const double* eCoarse, const unsigned int* nFine, double* xFine) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nFine[0]));
  if ((row < N)) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    double pe = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(PcolIdx[k]));
      double pv = Pvalues[k];
      double ec = eCoarse[col];
      pe = (pe + (pv * ec));
    }
    double xold = xFine[row];
    xFine[row] = (xold + pe);
  }
}

)kernel";
const char* k_cuda_navatala_sparse_mg_convergence_absolute_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_mg_convergence_absolute_f32(const float* residualNorm, const float* tolerance, unsigned int* converged) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  float norm = residualNorm[0];
  float tol = tolerance[0];
  unsigned int flag = (((norm < tol)) ? (1u) : (0u));
  converged[0] = flag;
}

)kernel";
const char* k_cuda_navatala_sparse_mg_convergence_relative_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_mg_convergence_relative_f32(const float* residualNorm, const float* initialNorm, const float* tolerance, unsigned int* converged) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  float norm = residualNorm[0];
  float init = initialNorm[0];
  float tol = tolerance[0];
  float ratio = (norm / init);
  unsigned int flag = (((ratio < tol)) ? (1u) : (0u));
  converged[0] = flag;
}

)kernel";
const char* k_cuda_navatala_sparse_mg_convergence_absolute_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_mg_convergence_absolute_f64(const double* residualNorm, const double* tolerance, unsigned int* converged) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  double norm = residualNorm[0];
  double tol = tolerance[0];
  unsigned int flag = (((norm < tol)) ? (1u) : (0u));
  converged[0] = flag;
}

)kernel";
const char* k_cuda_navatala_sparse_mg_convergence_relative_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_mg_convergence_relative_f64(const double* residualNorm, const double* initialNorm, const double* tolerance, unsigned int* converged) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  double norm = residualNorm[0];
  double init = initialNorm[0];
  double tol = tolerance[0];
  double ratio = (norm / init);
  unsigned int flag = (((ratio < tol)) ? (1u) : (0u));
  converged[0] = flag;
}

)kernel";
const char* k_cuda_navatala_sparse_mg_compute_residual_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_mg_compute_residual_f32(const float* b, const float* Ax, const unsigned int* n, float* r) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  if ((i < N)) {
    float bi = b[i];
    float axi = Ax[i];
    r[i] = (bi - axi);
  }
}

)kernel";
const char* k_cuda_navatala_sparse_mg_compute_residual_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_mg_compute_residual_f64(const double* b, const double* Ax, const unsigned int* n, double* r) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  if ((i < N)) {
    double bi = b[i];
    double axi = Ax[i];
    r[i] = (bi - axi);
  }
}

)kernel";
const char* k_cuda_navatala_sparse_mg_weighted_restrict_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_mg_weighted_restrict_f32(const unsigned int* RrowPtr, const unsigned int* RcolIdx, const float* Rvalues, const float* rFine, const float* weights, const unsigned int* nCoarse, float* rCoarse) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nCoarse[0]));
  if ((row < N)) {
    int rs = ((int)(RrowPtr[row]));
    int re = ((int)(RrowPtr[(row + 1)]));
    float acc = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(RcolIdx[k]));
      float rv = Rvalues[k];
      float rf = rFine[col];
      float w = weights[col];
      acc = (acc + (rv * (w * rf)));
    }
    rCoarse[row] = acc;
  }
}

)kernel";
const char* k_cuda_navatala_sparse_mg_weighted_restrict_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_mg_weighted_restrict_f64(const unsigned int* RrowPtr, const unsigned int* RcolIdx, const double* Rvalues, const double* rFine, const double* weights, const unsigned int* nCoarse, double* rCoarse) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nCoarse[0]));
  if ((row < N)) {
    int rs = ((int)(RrowPtr[row]));
    int re = ((int)(RrowPtr[(row + 1)]));
    double acc = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(RcolIdx[k]));
      double rv = Rvalues[k];
      double rf = rFine[col];
      double w = weights[col];
      acc = (acc + (rv * (w * rf)));
    }
    rCoarse[row] = acc;
  }
}

)kernel";
const char* k_cuda_navatala_sparse_mg_error_norm_partials_f32 = R"kernel(
#include <cuda_runtime.h>
template <typename T>
__device__ inline T gpu_warp_reduce_sum(T v) {
  unsigned mask = 0xffffffffu;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(mask, v, offset);
  }
  return v;
}
extern "C" __global__ void navatala_sparse_mg_error_norm_partials_f32(const float* error, const unsigned int* n, float* partials) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int lid = (int)(threadIdx.x);
  int N = ((int)(n[0]));
  float sq = (((gid < N)) ? ((error[gid] * error[gid])) : (__uint_as_float(0x00000000u)));
  float warpSum = gpu_warp_reduce_sum(sq);
  __shared__ float sdata[32];
  int lane = (int)(threadIdx.x % warpSize);
  if ((lane == 0)) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  __syncthreads();
  if ((lid < 8)) {
    float val = sdata[lid];
    float finalSum = gpu_warp_reduce_sum(val);
    if ((lid == 0)) {
      int grpId = (int)(blockIdx.x);
      partials[grpId] = finalSum;
    }
  }
}

)kernel";
const char* k_cuda_navatala_sparse_mg_error_norm_partials_f64 = R"kernel(
#include <cuda_runtime.h>
template <typename T>
__device__ inline T gpu_warp_reduce_sum(T v) {
  unsigned mask = 0xffffffffu;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(mask, v, offset);
  }
  return v;
}
extern "C" __global__ void navatala_sparse_mg_error_norm_partials_f64(const double* error, const unsigned int* n, double* partials) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int lid = (int)(threadIdx.x);
  int N = ((int)(n[0]));
  double sq = (((gid < N)) ? ((error[gid] * error[gid])) : (__longlong_as_double(0x0000000000000000ull)));
  double warpSum = gpu_warp_reduce_sum(sq);
  __shared__ double sdata[32];
  int lane = (int)(threadIdx.x % warpSize);
  if ((lane == 0)) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  __syncthreads();
  if ((lid < 8)) {
    double val = sdata[lid];
    double finalSum = gpu_warp_reduce_sum(val);
    if ((lid == 0)) {
      int grpId = (int)(blockIdx.x);
      partials[grpId] = finalSum;
    }
  }
}

)kernel";
const char* k_cuda_navatala_sparse_mg_zero_vector_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_mg_zero_vector_f32(const unsigned int* n, float* v) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  if ((i < N)) {
    v[i] = __uint_as_float(0x00000000u);
  }
}

)kernel";
const char* k_cuda_navatala_sparse_mg_zero_vector_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_mg_zero_vector_f64(const unsigned int* n, double* v) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  if ((i < N)) {
    v[i] = __longlong_as_double(0x0000000000000000ull);
  }
}

)kernel";
const char* k_cuda_navatala_sparse_mg_prolongate_damped_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_mg_prolongate_damped_f32(const unsigned int* ProwPtr, const unsigned int* PcolIdx, const float* Pvalues, const float* eCoarse, const float* omega, const unsigned int* nFine, float* xFine) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nFine[0]));
  if ((row < N)) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    float pe = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(PcolIdx[k]));
      float pv = Pvalues[k];
      float ec = eCoarse[col];
      pe = (pe + (pv * ec));
    }
    float om = omega[0];
    float xold = xFine[row];
    xFine[row] = (xold + (om * pe));
  }
}

)kernel";
const char* k_cuda_navatala_sparse_mg_prolongate_damped_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_mg_prolongate_damped_f64(const unsigned int* ProwPtr, const unsigned int* PcolIdx, const double* Pvalues, const double* eCoarse, const double* omega, const unsigned int* nFine, double* xFine) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nFine[0]));
  if ((row < N)) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    double pe = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(PcolIdx[k]));
      double pv = Pvalues[k];
      double ec = eCoarse[col];
      pe = (pe + (pv * ec));
    }
    double om = omega[0];
    double xold = xFine[row];
    xFine[row] = (xold + (om * pe));
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_restrict_f32[] = {
  { "RrowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "RcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Rvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rFine", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCoarse", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rCoarse", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_restrict_f32 = {
  1,
  "navatala_sparse_mg_restrict_f32",
  "cuda",
  "navatala_sparse_mg_restrict_f32",
  "kernel:cuda:navatala_sparse_mg_restrict_f32",
  "abi-r1:cuda:navatala_sparse_mg_restrict_f32",
  "abi-r1:cuda:navatala_sparse_mg_restrict_f32",
  6,
  kAbiArgs_cuda_navatala_sparse_mg_restrict_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_restrict_f64[] = {
  { "RrowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "RcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Rvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rFine", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCoarse", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rCoarse", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_restrict_f64 = {
  1,
  "navatala_sparse_mg_restrict_f64",
  "cuda",
  "navatala_sparse_mg_restrict_f64",
  "kernel:cuda:navatala_sparse_mg_restrict_f64",
  "abi-r1:cuda:navatala_sparse_mg_restrict_f64",
  "abi-r1:cuda:navatala_sparse_mg_restrict_f64",
  6,
  kAbiArgs_cuda_navatala_sparse_mg_restrict_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_prolongate_f32[] = {
  { "ProwPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "eCoarse", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nFine", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xFine", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_prolongate_f32 = {
  1,
  "navatala_sparse_mg_prolongate_f32",
  "cuda",
  "navatala_sparse_mg_prolongate_f32",
  "kernel:cuda:navatala_sparse_mg_prolongate_f32",
  "abi-r1:cuda:navatala_sparse_mg_prolongate_f32",
  "abi-r1:cuda:navatala_sparse_mg_prolongate_f32",
  6,
  kAbiArgs_cuda_navatala_sparse_mg_prolongate_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_prolongate_f64[] = {
  { "ProwPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "eCoarse", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nFine", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xFine", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_prolongate_f64 = {
  1,
  "navatala_sparse_mg_prolongate_f64",
  "cuda",
  "navatala_sparse_mg_prolongate_f64",
  "kernel:cuda:navatala_sparse_mg_prolongate_f64",
  "abi-r1:cuda:navatala_sparse_mg_prolongate_f64",
  "abi-r1:cuda:navatala_sparse_mg_prolongate_f64",
  6,
  kAbiArgs_cuda_navatala_sparse_mg_prolongate_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_convergence_absolute_f32[] = {
  { "residualNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "tolerance", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "converged", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_convergence_absolute_f32 = {
  1,
  "navatala_sparse_mg_convergence_absolute_f32",
  "cuda",
  "navatala_sparse_mg_convergence_absolute_f32",
  "kernel:cuda:navatala_sparse_mg_convergence_absolute_f32",
  "abi-r1:cuda:navatala_sparse_mg_convergence_absolute_f32",
  "abi-r1:cuda:navatala_sparse_mg_convergence_absolute_f32",
  3,
  kAbiArgs_cuda_navatala_sparse_mg_convergence_absolute_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_convergence_relative_f32[] = {
  { "residualNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "initialNorm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "tolerance", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "converged", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_convergence_relative_f32 = {
  1,
  "navatala_sparse_mg_convergence_relative_f32",
  "cuda",
  "navatala_sparse_mg_convergence_relative_f32",
  "kernel:cuda:navatala_sparse_mg_convergence_relative_f32",
  "abi-r1:cuda:navatala_sparse_mg_convergence_relative_f32",
  "abi-r1:cuda:navatala_sparse_mg_convergence_relative_f32",
  4,
  kAbiArgs_cuda_navatala_sparse_mg_convergence_relative_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_convergence_absolute_f64[] = {
  { "residualNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "tolerance", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "converged", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_convergence_absolute_f64 = {
  1,
  "navatala_sparse_mg_convergence_absolute_f64",
  "cuda",
  "navatala_sparse_mg_convergence_absolute_f64",
  "kernel:cuda:navatala_sparse_mg_convergence_absolute_f64",
  "abi-r1:cuda:navatala_sparse_mg_convergence_absolute_f64",
  "abi-r1:cuda:navatala_sparse_mg_convergence_absolute_f64",
  3,
  kAbiArgs_cuda_navatala_sparse_mg_convergence_absolute_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_convergence_relative_f64[] = {
  { "residualNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "initialNorm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "tolerance", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "converged", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_convergence_relative_f64 = {
  1,
  "navatala_sparse_mg_convergence_relative_f64",
  "cuda",
  "navatala_sparse_mg_convergence_relative_f64",
  "kernel:cuda:navatala_sparse_mg_convergence_relative_f64",
  "abi-r1:cuda:navatala_sparse_mg_convergence_relative_f64",
  "abi-r1:cuda:navatala_sparse_mg_convergence_relative_f64",
  4,
  kAbiArgs_cuda_navatala_sparse_mg_convergence_relative_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_compute_residual_f32[] = {
  { "b", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Ax", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "r", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_compute_residual_f32 = {
  1,
  "navatala_sparse_mg_compute_residual_f32",
  "cuda",
  "navatala_sparse_mg_compute_residual_f32",
  "kernel:cuda:navatala_sparse_mg_compute_residual_f32",
  "abi-r1:cuda:navatala_sparse_mg_compute_residual_f32",
  "abi-r1:cuda:navatala_sparse_mg_compute_residual_f32",
  4,
  kAbiArgs_cuda_navatala_sparse_mg_compute_residual_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_compute_residual_f64[] = {
  { "b", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Ax", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "r", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_compute_residual_f64 = {
  1,
  "navatala_sparse_mg_compute_residual_f64",
  "cuda",
  "navatala_sparse_mg_compute_residual_f64",
  "kernel:cuda:navatala_sparse_mg_compute_residual_f64",
  "abi-r1:cuda:navatala_sparse_mg_compute_residual_f64",
  "abi-r1:cuda:navatala_sparse_mg_compute_residual_f64",
  4,
  kAbiArgs_cuda_navatala_sparse_mg_compute_residual_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_weighted_restrict_f32[] = {
  { "RrowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "RcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Rvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rFine", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCoarse", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rCoarse", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_weighted_restrict_f32 = {
  1,
  "navatala_sparse_mg_weighted_restrict_f32",
  "cuda",
  "navatala_sparse_mg_weighted_restrict_f32",
  "kernel:cuda:navatala_sparse_mg_weighted_restrict_f32",
  "abi-r1:cuda:navatala_sparse_mg_weighted_restrict_f32",
  "abi-r1:cuda:navatala_sparse_mg_weighted_restrict_f32",
  7,
  kAbiArgs_cuda_navatala_sparse_mg_weighted_restrict_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_weighted_restrict_f64[] = {
  { "RrowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "RcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Rvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rFine", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCoarse", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rCoarse", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_weighted_restrict_f64 = {
  1,
  "navatala_sparse_mg_weighted_restrict_f64",
  "cuda",
  "navatala_sparse_mg_weighted_restrict_f64",
  "kernel:cuda:navatala_sparse_mg_weighted_restrict_f64",
  "abi-r1:cuda:navatala_sparse_mg_weighted_restrict_f64",
  "abi-r1:cuda:navatala_sparse_mg_weighted_restrict_f64",
  7,
  kAbiArgs_cuda_navatala_sparse_mg_weighted_restrict_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_error_norm_partials_f32[] = {
  { "error", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partials", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_error_norm_partials_f32 = {
  1,
  "navatala_sparse_mg_error_norm_partials_f32",
  "cuda",
  "navatala_sparse_mg_error_norm_partials_f32",
  "kernel:cuda:navatala_sparse_mg_error_norm_partials_f32",
  "abi-r1:cuda:navatala_sparse_mg_error_norm_partials_f32",
  "abi-r1:cuda:navatala_sparse_mg_error_norm_partials_f32",
  3,
  kAbiArgs_cuda_navatala_sparse_mg_error_norm_partials_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_error_norm_partials_f64[] = {
  { "error", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partials", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_error_norm_partials_f64 = {
  1,
  "navatala_sparse_mg_error_norm_partials_f64",
  "cuda",
  "navatala_sparse_mg_error_norm_partials_f64",
  "kernel:cuda:navatala_sparse_mg_error_norm_partials_f64",
  "abi-r1:cuda:navatala_sparse_mg_error_norm_partials_f64",
  "abi-r1:cuda:navatala_sparse_mg_error_norm_partials_f64",
  3,
  kAbiArgs_cuda_navatala_sparse_mg_error_norm_partials_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_zero_vector_f32[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "v", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_zero_vector_f32 = {
  1,
  "navatala_sparse_mg_zero_vector_f32",
  "cuda",
  "navatala_sparse_mg_zero_vector_f32",
  "kernel:cuda:navatala_sparse_mg_zero_vector_f32",
  "abi-r1:cuda:navatala_sparse_mg_zero_vector_f32",
  "abi-r1:cuda:navatala_sparse_mg_zero_vector_f32",
  2,
  kAbiArgs_cuda_navatala_sparse_mg_zero_vector_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_zero_vector_f64[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "v", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_zero_vector_f64 = {
  1,
  "navatala_sparse_mg_zero_vector_f64",
  "cuda",
  "navatala_sparse_mg_zero_vector_f64",
  "kernel:cuda:navatala_sparse_mg_zero_vector_f64",
  "abi-r1:cuda:navatala_sparse_mg_zero_vector_f64",
  "abi-r1:cuda:navatala_sparse_mg_zero_vector_f64",
  2,
  kAbiArgs_cuda_navatala_sparse_mg_zero_vector_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_prolongate_damped_f32[] = {
  { "ProwPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "eCoarse", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omega", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nFine", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xFine", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_prolongate_damped_f32 = {
  1,
  "navatala_sparse_mg_prolongate_damped_f32",
  "cuda",
  "navatala_sparse_mg_prolongate_damped_f32",
  "kernel:cuda:navatala_sparse_mg_prolongate_damped_f32",
  "abi-r1:cuda:navatala_sparse_mg_prolongate_damped_f32",
  "abi-r1:cuda:navatala_sparse_mg_prolongate_damped_f32",
  7,
  kAbiArgs_cuda_navatala_sparse_mg_prolongate_damped_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_sparse_mg_prolongate_damped_f64[] = {
  { "ProwPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "eCoarse", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omega", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "nFine", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xFine", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_sparse_mg_prolongate_damped_f64 = {
  1,
  "navatala_sparse_mg_prolongate_damped_f64",
  "cuda",
  "navatala_sparse_mg_prolongate_damped_f64",
  "kernel:cuda:navatala_sparse_mg_prolongate_damped_f64",
  "abi-r1:cuda:navatala_sparse_mg_prolongate_damped_f64",
  "abi-r1:cuda:navatala_sparse_mg_prolongate_damped_f64",
  7,
  kAbiArgs_cuda_navatala_sparse_mg_prolongate_damped_f64
};

bool tryGetKernelAbiManifest_cuda_sparse_mg(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_restrict_f32") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_restrict_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_restrict_f64") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_restrict_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_prolongate_f32") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_prolongate_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_prolongate_f64") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_prolongate_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_convergence_absolute_f32") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_convergence_absolute_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_convergence_relative_f32") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_convergence_relative_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_convergence_absolute_f64") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_convergence_absolute_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_convergence_relative_f64") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_convergence_relative_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_compute_residual_f32") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_compute_residual_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_compute_residual_f64") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_compute_residual_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_weighted_restrict_f32") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_weighted_restrict_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_weighted_restrict_f64") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_weighted_restrict_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_error_norm_partials_f32") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_error_norm_partials_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_error_norm_partials_f64") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_error_norm_partials_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_zero_vector_f32") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_zero_vector_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_zero_vector_f64") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_zero_vector_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_prolongate_damped_f32") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_prolongate_damped_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_prolongate_damped_f64") {
    out = &kAbiManifest_cuda_navatala_sparse_mg_prolongate_damped_f64;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_cuda_sparse_mg(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_restrict_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_restrict_f32";
    std::string_view sv(k_cuda_navatala_sparse_mg_restrict_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_restrict_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_restrict_f64";
    std::string_view sv(k_cuda_navatala_sparse_mg_restrict_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_prolongate_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_prolongate_f32";
    std::string_view sv(k_cuda_navatala_sparse_mg_prolongate_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_prolongate_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_prolongate_f64";
    std::string_view sv(k_cuda_navatala_sparse_mg_prolongate_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_convergence_absolute_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_convergence_absolute_f32";
    std::string_view sv(k_cuda_navatala_sparse_mg_convergence_absolute_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_convergence_relative_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_convergence_relative_f32";
    std::string_view sv(k_cuda_navatala_sparse_mg_convergence_relative_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_convergence_absolute_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_convergence_absolute_f64";
    std::string_view sv(k_cuda_navatala_sparse_mg_convergence_absolute_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_convergence_relative_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_convergence_relative_f64";
    std::string_view sv(k_cuda_navatala_sparse_mg_convergence_relative_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_compute_residual_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_compute_residual_f32";
    std::string_view sv(k_cuda_navatala_sparse_mg_compute_residual_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_compute_residual_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_compute_residual_f64";
    std::string_view sv(k_cuda_navatala_sparse_mg_compute_residual_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_weighted_restrict_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_weighted_restrict_f32";
    std::string_view sv(k_cuda_navatala_sparse_mg_weighted_restrict_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_weighted_restrict_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_weighted_restrict_f64";
    std::string_view sv(k_cuda_navatala_sparse_mg_weighted_restrict_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_error_norm_partials_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_error_norm_partials_f32";
    std::string_view sv(k_cuda_navatala_sparse_mg_error_norm_partials_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_error_norm_partials_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_error_norm_partials_f64";
    std::string_view sv(k_cuda_navatala_sparse_mg_error_norm_partials_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_zero_vector_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_zero_vector_f32";
    std::string_view sv(k_cuda_navatala_sparse_mg_zero_vector_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_zero_vector_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_zero_vector_f64";
    std::string_view sv(k_cuda_navatala_sparse_mg_zero_vector_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_prolongate_damped_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_prolongate_damped_f32";
    std::string_view sv(k_cuda_navatala_sparse_mg_prolongate_damped_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_sparse_mg_prolongate_damped_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_sparse_mg_prolongate_damped_f64";
    std::string_view sv(k_cuda_navatala_sparse_mg_prolongate_damped_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

