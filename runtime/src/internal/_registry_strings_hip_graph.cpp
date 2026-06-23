// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `hip_graph` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_hip_navatala_graph_pack_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_pack_f32(const int* indices, const float* src, const int* count, float* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    dst[i] = src[indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]];
  }
}

)kernel";
const char* k_hip_navatala_graph_pack_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_pack_f64(const int* indices, const double* src, const int* count, double* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    dst[i] = src[indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]];
  }
}

)kernel";
const char* k_hip_navatala_graph_pack_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_pack_i32(const int* indices, const int* src, const int* count, int* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    dst[i] = src[indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]];
  }
}

)kernel";
const char* k_hip_navatala_graph_pack_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_pack_u32(const int* indices, const unsigned int* src, const int* count, unsigned int* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    dst[i] = src[indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]];
  }
}

)kernel";
const char* k_hip_navatala_graph_unpack_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_unpack_f32(const int* indices, const float* src, const int* count, float* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    dst[indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]] = src[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
  }
}

)kernel";
const char* k_hip_navatala_graph_unpack_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_unpack_f64(const int* indices, const double* src, const int* count, double* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    dst[indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]] = src[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
  }
}

)kernel";
const char* k_hip_navatala_graph_unpack_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_unpack_i32(const int* indices, const int* src, const int* count, int* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    dst[indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]] = src[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
  }
}

)kernel";
const char* k_hip_navatala_graph_unpack_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_unpack_u32(const int* indices, const unsigned int* src, const int* count, unsigned int* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    dst[indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]] = src[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
  }
}

)kernel";
const char* k_hip_navatala_graph_pack_f32x3 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_pack_f32x3(const int* indices, const float* src, const int* count, const int* n, float* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(3); ++c) {
      int dstLin = ((c * count[0]) + i);
      int srcLin = ((c * n[0]) + indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_unpack_f32x3 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_unpack_f32x3(const int* indices, const float* src, const int* count, const int* n, float* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(3); ++c) {
      int srcLin = ((c * count[0]) + i);
      int dstLin = ((c * n[0]) + indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_pack_f32x6 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_pack_f32x6(const int* indices, const float* src, const int* count, const int* n, float* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(6); ++c) {
      int dstLin = ((c * count[0]) + i);
      int srcLin = ((c * n[0]) + indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_unpack_f32x6 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_unpack_f32x6(const int* indices, const float* src, const int* count, const int* n, float* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(6); ++c) {
      int srcLin = ((c * count[0]) + i);
      int dstLin = ((c * n[0]) + indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_pack_f32x9 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_pack_f32x9(const int* indices, const float* src, const int* count, const int* n, float* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(9); ++c) {
      int dstLin = ((c * count[0]) + i);
      int srcLin = ((c * n[0]) + indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_unpack_f32x9 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_unpack_f32x9(const int* indices, const float* src, const int* count, const int* n, float* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(9); ++c) {
      int srcLin = ((c * count[0]) + i);
      int dstLin = ((c * n[0]) + indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_pack_f64x3 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_pack_f64x3(const int* indices, const double* src, const int* count, const int* n, double* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(3); ++c) {
      int dstLin = ((c * count[0]) + i);
      int srcLin = ((c * n[0]) + indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_unpack_f64x3 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_unpack_f64x3(const int* indices, const double* src, const int* count, const int* n, double* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(3); ++c) {
      int srcLin = ((c * count[0]) + i);
      int dstLin = ((c * n[0]) + indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_pack_f64x6 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_pack_f64x6(const int* indices, const double* src, const int* count, const int* n, double* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(6); ++c) {
      int dstLin = ((c * count[0]) + i);
      int srcLin = ((c * n[0]) + indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_unpack_f64x6 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_unpack_f64x6(const int* indices, const double* src, const int* count, const int* n, double* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(6); ++c) {
      int srcLin = ((c * count[0]) + i);
      int dstLin = ((c * n[0]) + indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_pack_f64x9 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_pack_f64x9(const int* indices, const double* src, const int* count, const int* n, double* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(9); ++c) {
      int dstLin = ((c * count[0]) + i);
      int srcLin = ((c * n[0]) + indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_unpack_f64x9 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_unpack_f64x9(const int* indices, const double* src, const int* count, const int* n, double* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(9); ++c) {
      int srcLin = ((c * count[0]) + i);
      int dstLin = ((c * n[0]) + indices[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_degree_out_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_degree_out_u32(const unsigned int* offsets, const unsigned int* numVertices, unsigned int* degrees) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numV = numVertices[0];
  if (gid < numV) {
    unsigned int hi = offsets[(gid + 1u)];
    unsigned int lo = offsets[gid];
    unsigned int deg = (hi - lo);
    degrees[gid] = deg;
  }
}

)kernel";
const char* k_hip_navatala_graph_degree_in_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_degree_in_u32(const unsigned int* offsets, const unsigned int* numVertices, unsigned int* degrees) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numV = numVertices[0];
  if (gid < numV) {
    unsigned int hi = offsets[(gid + 1u)];
    unsigned int lo = offsets[gid];
    unsigned int deg = (hi - lo);
    degrees[gid] = deg;
  }
}

)kernel";
const char* k_hip_navatala_graph_spmv_weighted_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_spmv_weighted_f32(const unsigned int* offsets, const unsigned int* indices, const float* weights, const float* x, const unsigned int* numVertices, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numV = numVertices[0];
  if (gid < numV) {
    unsigned int base = offsets[gid];
    unsigned int endv = offsets[(gid + 1u)];
    unsigned int rowlen = (endv - base);
    float acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(rowlen); ++k) {
      unsigned int eidx = (base + ((unsigned int)(k)));
      unsigned int col = indices[eidx];
      float w = weights[eidx];
      float xv = x[col];
      acc = (acc + (w * xv));
    }
    y[gid] = acc;
  }
}

)kernel";
const char* k_hip_navatala_graph_spmv_weighted_subgroup_f32 = R"kernel(
#include <hip/hip_runtime.h>
template <typename T>
__device__ inline T gpu_warp_reduce_sum(T v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}
extern "C" __global__ void navatala_graph_spmv_weighted_subgroup_f32(const unsigned int* offsets, const unsigned int* indices, const float* weights, const float* x, const unsigned int* numVertices, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lane = ((unsigned int)((int)(threadIdx.x % warpSize)));
  unsigned int subgroupSize = ((unsigned int)((int)(warpSize)));
  unsigned int row = (gid / subgroupSize);
  unsigned int numV = numVertices[0];
  if (row < numV) {
    unsigned int base = offsets[row];
    unsigned int endv = offsets[(row + 1u)];
    unsigned int rowlen = (endv - base);
    unsigned int rowIters = ((rowlen + (subgroupSize - 1u)) / subgroupSize);
    float laneAcc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(rowIters); ++k) {
      unsigned int rel = ((((unsigned int)(k)) * subgroupSize) + lane);
      if (rel < rowlen) {
        unsigned int eidx = (base + rel);
        unsigned int col = indices[eidx];
        float w = weights[eidx];
        float xv = x[col];
        laneAcc = (laneAcc + (w * xv));
      }
    }
    float rowSum = gpu_warp_reduce_sum(laneAcc);
    if (lane == 0u) {
      y[row] = rowSum;
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_spmv_unweighted_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_spmv_unweighted_f32(const unsigned int* offsets, const unsigned int* indices, const float* x, const unsigned int* numVertices, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numV = numVertices[0];
  if (gid < numV) {
    unsigned int base = offsets[gid];
    unsigned int endv = offsets[(gid + 1u)];
    unsigned int rowlen = (endv - base);
    float acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(rowlen); ++k) {
      unsigned int eidx = (base + ((unsigned int)(k)));
      unsigned int col = indices[eidx];
      float xv = x[col];
      acc = (acc + xv);
    }
    y[gid] = acc;
  }
}

)kernel";
const char* k_hip_navatala_graph_spmv_unweighted_subgroup_f32 = R"kernel(
#include <hip/hip_runtime.h>
template <typename T>
__device__ inline T gpu_warp_reduce_sum(T v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}
extern "C" __global__ void navatala_graph_spmv_unweighted_subgroup_f32(const unsigned int* offsets, const unsigned int* indices, const float* x, const unsigned int* numVertices, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lane = ((unsigned int)((int)(threadIdx.x % warpSize)));
  unsigned int subgroupSize = ((unsigned int)((int)(warpSize)));
  unsigned int row = (gid / subgroupSize);
  unsigned int numV = numVertices[0];
  if (row < numV) {
    unsigned int base = offsets[row];
    unsigned int endv = offsets[(row + 1u)];
    unsigned int rowlen = (endv - base);
    unsigned int rowIters = ((rowlen + (subgroupSize - 1u)) / subgroupSize);
    float laneAcc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(rowIters); ++k) {
      unsigned int rel = ((((unsigned int)(k)) * subgroupSize) + lane);
      if (rel < rowlen) {
        unsigned int eidx = (base + rel);
        unsigned int col = indices[eidx];
        float xv = x[col];
        laneAcc = (laneAcc + xv);
      }
    }
    float rowSum = gpu_warp_reduce_sum(laneAcc);
    if (lane == 0u) {
      y[row] = rowSum;
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_axpy2_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_axpy2_f32(const float* x, const float* z, const float* alpha, const float* beta, const unsigned int* count, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    float alphaV = alpha[0];
    float betaV = beta[0];
    float xv = x[gid];
    float zv = z[gid];
    float yv = ((alphaV * xv) + (betaV * zv));
    y[gid] = yv;
  }
}

)kernel";
const char* k_hip_navatala_graph_scale_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_scale_f32(const float* x, const float* alpha, const unsigned int* count, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    float alphaV = alpha[0];
    float xv = x[gid];
    y[gid] = (alphaV * xv);
  }
}

)kernel";
const char* k_hip_navatala_graph_scale_add_bias_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_scale_add_bias_f32(const float* x, const float* alpha, const float* bias, const unsigned int* count, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    float alphaV = alpha[0];
    float biasV = bias[0];
    float xv = x[gid];
    y[gid] = ((alphaV * xv) + biasV);
  }
}

)kernel";
const char* k_hip_navatala_graph_l1norm_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_l1norm_f32(const float* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      float val = abs(_input[idx]);
      gsAcc = (gsAcc + val);
    }
  }
  sdata[lid] = gsAcc;
  __syncthreads();
  unsigned int redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    unsigned int stride = redStride;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = redStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    result[0] = sdata[0];
  }
}

)kernel";
const char* k_hip_navatala_graph_recip_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_recip_f32(const float* x, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (gid == 0u) {
    float xv = x[0];
    if (xv == __uint_as_float(0x00000000u)) {
      y[0] = __uint_as_float(0x00000000u);
    } else {
      y[0] = (__uint_as_float(0x3f800000u) / xv);
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_l1_diff_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_l1_diff_f32(const float* a, const float* b, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      float av = a[idx];
      float bv = b[idx];
      float d = abs((av - bv));
      gsAcc = (gsAcc + d);
    }
  }
  sdata[lid] = gsAcc;
  __syncthreads();
  unsigned int redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    unsigned int stride = redStride;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = redStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    result[0] = sdata[0];
  }
}

)kernel";
const char* k_hip_navatala_graph_count_diff_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_count_diff_u32(const unsigned int* a, const unsigned int* b, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      unsigned int av = a[idx];
      unsigned int bv = b[idx];
      float diff = (((av == bv)) ? (__uint_as_float(0x00000000u)) : (__uint_as_float(0x3f800000u)));
      gsAcc = (gsAcc + diff);
    }
  }
  sdata[lid] = gsAcc;
  __syncthreads();
  unsigned int redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    unsigned int stride = redStride;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = redStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    result[0] = sdata[0];
  }
}

)kernel";
const char* k_hip_navatala_graph_copy_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_copy_f32(const float* x, const unsigned int* count, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    float xv = x[gid];
    y[gid] = xv;
  }
}

)kernel";
const char* k_hip_navatala_graph_copy_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_copy_u32(const unsigned int* x, const unsigned int* count, unsigned int* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    unsigned int xv = x[gid];
    y[gid] = xv;
  }
}

)kernel";
const char* k_hip_navatala_graph_relax_vertex_pull_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_relax_vertex_pull_f32(const unsigned int* inOffsets, const unsigned int* inSrcs, const float* inWeights, float* dist, const unsigned int* numVertices, const float* inf) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numV = numVertices[0];
  if (gid < numV) {
    unsigned int base = inOffsets[gid];
    unsigned int endv = inOffsets[(gid + 1u)];
    unsigned int deg = (endv - base);
    float infV = inf[0];
    float best = dist[gid];
    for (int k = 0; k < (int)(deg); ++k) {
      unsigned int eidx = (base + ((unsigned int)(k)));
      unsigned int u = inSrcs[eidx];
      float w = inWeights[eidx];
      float du = dist[u];
      if (du < infV) {
        float cand = (du + w);
        float cur = best;
        best = (((cand < cur)) ? (cand) : (cur));
      }
    }
    dist[gid] = best;
  }
}

)kernel";
const char* k_hip_navatala_graph_label_propagate_edges_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_label_propagate_edges_u32(const unsigned int* srcs, const unsigned int* dsts, unsigned int* labels, const unsigned int* numEdges) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numE = numEdges[0];
  if (gid < numE) {
    unsigned int u = srcs[gid];
    unsigned int v = dsts[gid];
    unsigned int lu = labels[u];
    unsigned int lv = labels[v];
    unsigned int prevLv = atomicMin(&(labels[v]), lu);
    unsigned int prevLu = atomicMin(&(labels[u]), lv);
  }
}

)kernel";
const char* k_hip_navatala_graph_symmetrize_reverse_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_symmetrize_reverse_u32(const unsigned int* srcs, const unsigned int* dsts, const unsigned int* numEdges, unsigned int* outSrcs, unsigned int* outDsts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numE = numEdges[0];
  if (gid < numE) {
    unsigned int u = srcs[gid];
    unsigned int v = dsts[gid];
    outSrcs[gid] = v;
    outDsts[gid] = u;
  }
}

)kernel";
const char* k_hip_navatala_graph_apply_renumber_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_apply_renumber_u32(const unsigned int* mapping, const unsigned int* numVertices, unsigned int* newIds) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numV = numVertices[0];
  if (gid < numV) {
    newIds[gid] = mapping[gid];
  }
}

)kernel";
const char* k_hip_navatala_graph_peel_below_threshold_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_peel_below_threshold_u32(unsigned int* _active, const unsigned int* degrees, const unsigned int* threshold, const unsigned int* numVertices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numV = numVertices[0];
  if (gid < numV) {
    unsigned int deg = degrees[gid];
    unsigned int k = threshold[0];
    if (deg < k) {
      _active[gid] = 0u;
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_pair_intersection_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_pair_intersection_u32(const unsigned int* offsets, const unsigned int* indices, const unsigned int* pairsA, const unsigned int* pairsB, const unsigned int* numPairs, unsigned int* inter) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numP = numPairs[0];
  if (gid < numP) {
    unsigned int a = pairsA[gid];
    unsigned int b = pairsB[gid];
    unsigned int baseU = offsets[a];
    unsigned int endU = offsets[(a + 1u)];
    unsigned int baseV = offsets[b];
    unsigned int endV = offsets[(b + 1u)];
    unsigned int degU = (endU - baseU);
    unsigned int interAccum = 0u;
    unsigned int loAccum = 0u;
    unsigned int hiAccum = 0u;
    for (int k = 0; k < (int)(degU); ++k) {
      unsigned int eidx = (baseU + ((unsigned int)(k)));
      unsigned int w = indices[eidx];
      loAccum = baseV;
      hiAccum = endV;
      for (int _bs = 0; _bs < (int)(32u); ++_bs) {
        unsigned int lo = loAccum;
        unsigned int hi = hiAccum;
        if (lo < hi) {
          unsigned int mid = ((lo + hi) / 2u);
          unsigned int midval = indices[mid];
          bool goRight = (midval < w);
          unsigned int newLo = ((goRight) ? ((mid + 1u)) : (lo));
          unsigned int newHi = ((goRight) ? (hi) : (mid));
          loAccum = newLo;
          hiAccum = newHi;
        }
      }
      unsigned int lb = loAccum;
      if (lb < endV) {
        unsigned int cand = indices[lb];
        if (cand == w) {
          interAccum = (interAccum + 1u);
        }
      }
    }
    inter[gid] = interAccum;
  }
}

)kernel";
const char* k_hip_navatala_graph_jaccard_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_jaccard_f32(const unsigned int* inter, const unsigned int* degrees, const unsigned int* pairsA, const unsigned int* pairsB, const unsigned int* numPairs, float* coeff) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numP = numPairs[0];
  if (gid < numP) {
    unsigned int a = pairsA[gid];
    unsigned int b = pairsB[gid];
    unsigned int iU = inter[gid];
    unsigned int dAU = degrees[a];
    unsigned int dBU = degrees[b];
    float iF = ((float)(iU));
    float dAF = ((float)(dAU));
    float dBF = ((float)(dBU));
    coeff[gid] = (((((dAF + dBF) - iF) > __uint_as_float(0x00000000u))) ? ((iF / ((dAF + dBF) - iF))) : (__uint_as_float(0x00000000u)));
  }
}

)kernel";
const char* k_hip_navatala_graph_overlap_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_overlap_f32(const unsigned int* inter, const unsigned int* degrees, const unsigned int* pairsA, const unsigned int* pairsB, const unsigned int* numPairs, float* coeff) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numP = numPairs[0];
  if (gid < numP) {
    unsigned int a = pairsA[gid];
    unsigned int b = pairsB[gid];
    unsigned int iU = inter[gid];
    unsigned int dAU = degrees[a];
    unsigned int dBU = degrees[b];
    float iF = ((float)(iU));
    float dAF = ((float)(dAU));
    float dBF = ((float)(dBU));
    coeff[gid] = ((((((dAF < dBF)) ? (dAF) : (dBF)) > __uint_as_float(0x00000000u))) ? ((iF / (((dAF < dBF)) ? (dAF) : (dBF)))) : (__uint_as_float(0x00000000u)));
  }
}

)kernel";
const char* k_hip_navatala_graph_sorensen_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_sorensen_f32(const unsigned int* inter, const unsigned int* degrees, const unsigned int* pairsA, const unsigned int* pairsB, const unsigned int* numPairs, float* coeff) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numP = numPairs[0];
  if (gid < numP) {
    unsigned int a = pairsA[gid];
    unsigned int b = pairsB[gid];
    unsigned int iU = inter[gid];
    unsigned int dAU = degrees[a];
    unsigned int dBU = degrees[b];
    float iF = ((float)(iU));
    float dAF = ((float)(dAU));
    float dBF = ((float)(dBU));
    coeff[gid] = ((((dAF + dBF) > __uint_as_float(0x00000000u))) ? (((__uint_as_float(0x40000000u) * iF) / (dAF + dBF))) : (__uint_as_float(0x00000000u)));
  }
}

)kernel";
const char* k_hip_navatala_graph_cosine_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_cosine_f32(const unsigned int* inter, const unsigned int* degrees, const unsigned int* pairsA, const unsigned int* pairsB, const unsigned int* numPairs, float* coeff) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numP = numPairs[0];
  if (gid < numP) {
    unsigned int a = pairsA[gid];
    unsigned int b = pairsB[gid];
    unsigned int iU = inter[gid];
    unsigned int dAU = degrees[a];
    unsigned int dBU = degrees[b];
    float iF = ((float)(iU));
    float dAF = ((float)(dAU));
    float dBF = ((float)(dBU));
    coeff[gid] = (((sqrt((dAF * dBF)) > __uint_as_float(0x00000000u))) ? ((iF / sqrt((dAF * dBF)))) : (__uint_as_float(0x00000000u)));
  }
}

)kernel";
const char* k_hip_navatala_graph_uniform_random_walk_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_uniform_random_walk_u32(const unsigned int* offsets, const unsigned int* indices, const unsigned int* startVertices, const unsigned int* maxLen, const unsigned int* numWalks, const unsigned int* seed, unsigned int* walks) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numW = numWalks[0];
  if (gid < numW) {
    unsigned int maxL = maxLen[0];
    unsigned int seedV = seed[0];
    unsigned int rowBase = (gid * maxL);
    unsigned int curAccum = startVertices[gid];
    unsigned int rngAccum = (seedV + gid);
    for (int s = 0; s < (int)(maxL); ++s) {
      unsigned int cur = curAccum;
      unsigned int pos = (rowBase + ((unsigned int)(s)));
      walks[pos] = cur;
      unsigned int off = offsets[cur];
      unsigned int off1 = offsets[(cur + 1u)];
      unsigned int deg = (off1 - off);
      unsigned int rng = rngAccum;
      unsigned int rngNext = ((rng * 1103515245u) + 12345u);
      rngAccum = rngNext;
      if (deg > 0u) {
        unsigned int pick = (rngNext % deg);
        unsigned int nbr = indices[(off + pick)];
        curAccum = nbr;
      }
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_out_degree_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_out_degree_u32(const unsigned int* offsets, const unsigned int* numVertices, unsigned int* degrees) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numV = numVertices[0];
  if (gid < numV) {
    unsigned int hi = offsets[(gid + 1u)];
    unsigned int lo = offsets[gid];
    unsigned int deg = (hi - lo);
    degrees[gid] = deg;
  }
}

)kernel";
const char* k_hip_navatala_graph_in_degree_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_in_degree_u32(const unsigned int* offsets, const unsigned int* numVertices, unsigned int* degrees) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numV = numVertices[0];
  if (gid < numV) {
    unsigned int hi = offsets[(gid + 1u)];
    unsigned int lo = offsets[gid];
    unsigned int deg = (hi - lo);
    degrees[gid] = deg;
  }
}

)kernel";
const char* k_hip_navatala_graph_rbf_affinity_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_rbf_affinity_f32(const float* data, const float* sigma, const unsigned int* n, const unsigned int* d, float* W) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int dVal = d[0];
  float sigmaVal = sigma[0];
  unsigned int totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int i = (gid / nVal);
    unsigned int j = (gid % nVal);
    bool isDiag = (i == j);
    if (isDiag) {
      W[gid] = __uint_as_float(0x00000000u);
    } else {
      float distSq = __uint_as_float(0x00000000u);
      unsigned int rowI = (i * dVal);
      unsigned int rowJ = (j * dVal);
      unsigned int idxI = rowI;
      unsigned int idxJ = rowJ;
      float xi = data[idxI];
      float xj = data[idxJ];
      float diff = (xi - xj);
      float diffSq = (diff * diff);
      float sigmaSq = (sigmaVal * sigmaVal);
      float twoSigmaSq = (__uint_as_float(0x40000000u) * sigmaSq);
      float exponent = ((__uint_as_float(0x00000000u) - diffSq) / twoSigmaSq);
      float affinity = exp(exponent);
      W[gid] = affinity;
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_rbf_affinity_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_rbf_affinity_f64(const double* data, const double* sigma, const unsigned int* n, const unsigned int* d, double* W) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int dVal = d[0];
  double sigmaVal = sigma[0];
  unsigned int totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int i = (gid / nVal);
    unsigned int j = (gid % nVal);
    bool isDiag = (i == j);
    if (isDiag) {
      W[gid] = __longlong_as_double(0x0000000000000000ull);
    } else {
      double distSq = __longlong_as_double(0x0000000000000000ull);
      unsigned int rowI = (i * dVal);
      unsigned int rowJ = (j * dVal);
      unsigned int idxI = rowI;
      unsigned int idxJ = rowJ;
      double xi = data[idxI];
      double xj = data[idxJ];
      double diff = (xi - xj);
      double diffSq = (diff * diff);
      double sigmaSq = (sigmaVal * sigmaVal);
      double twoSigmaSq = (__longlong_as_double(0x4000000000000000ull) * sigmaSq);
      double exponent = ((__longlong_as_double(0x0000000000000000ull) - diffSq) / twoSigmaSq);
      double affinity = exp(exponent);
      W[gid] = affinity;
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_compute_degree_matrix_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_compute_degree_matrix_f32(const float* W, const unsigned int* n, float* D) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int wgid = ((unsigned int)((int)(blockIdx.x)));
  __shared__ float sdata[256];
  unsigned int nVal = n[0];
  unsigned int rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    unsigned int rowBase = (rowIdx * nVal);
    bool colInBounds = (lid < nVal);
    unsigned int elemIdx = (rowBase + lid);
    if (colInBounds) {
      float wVal = W[elemIdx];
      sdata[lid] = wVal;
    } else {
      sdata[lid] = __uint_as_float(0x00000000u);
    }
    __syncthreads();
    unsigned int degF32RedStride = 128u;
    for (int degF32RedStep = 0; degF32RedStep < (int)(8); ++degF32RedStep) {
      unsigned int degF32Stride = degF32RedStride;
      if (lid < degF32Stride) {
        float other = sdata[(lid + degF32Stride)];
        float mine = sdata[lid];
        float sum = (mine + other);
        sdata[lid] = sum;
      }
      unsigned int degF32StrideToHalve = degF32RedStride;
      unsigned int degF32NextStride = (degF32StrideToHalve >> 1u);
      degF32RedStride = degF32NextStride;
      __syncthreads();
    }
    if (lid == 0u) {
      float degreeVal = sdata[0];
      D[rowIdx] = degreeVal;
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_compute_degree_matrix_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_compute_degree_matrix_f64(const double* W, const unsigned int* n, double* D) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int wgid = ((unsigned int)((int)(blockIdx.x)));
  __shared__ double sdata[256];
  unsigned int nVal = n[0];
  unsigned int rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    unsigned int rowBase = (rowIdx * nVal);
    bool colInBounds = (lid < nVal);
    unsigned int elemIdx = (rowBase + lid);
    if (colInBounds) {
      double wVal = W[elemIdx];
      sdata[lid] = wVal;
    } else {
      sdata[lid] = __longlong_as_double(0x0000000000000000ull);
    }
    __syncthreads();
    unsigned int degF64RedStride = 128u;
    for (int degF64RedStep = 0; degF64RedStep < (int)(8); ++degF64RedStep) {
      unsigned int degF64Stride = degF64RedStride;
      if (lid < degF64Stride) {
        double other = sdata[(lid + degF64Stride)];
        double mine = sdata[lid];
        double sum = (mine + other);
        sdata[lid] = sum;
      }
      unsigned int degF64StrideToHalve = degF64RedStride;
      unsigned int degF64NextStride = (degF64StrideToHalve >> 1u);
      degF64RedStride = degF64NextStride;
      __syncthreads();
    }
    if (lid == 0u) {
      double degreeVal = sdata[0];
      D[rowIdx] = degreeVal;
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_compute_laplacian_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_compute_laplacian_f32(const float* W, const float* D, const unsigned int* n, float* L) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int i = (gid / nVal);
    unsigned int j = (gid % nVal);
    bool isDiag = (i == j);
    if (isDiag) {
      float dVal = D[i];
      L[gid] = dVal;
    } else {
      float wVal = W[gid];
      float negW = (__uint_as_float(0x00000000u) - wVal);
      L[gid] = negW;
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_compute_laplacian_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_compute_laplacian_f64(const double* W, const double* D, const unsigned int* n, double* L) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int i = (gid / nVal);
    unsigned int j = (gid % nVal);
    bool isDiag = (i == j);
    if (isDiag) {
      double dVal = D[i];
      L[gid] = dVal;
    } else {
      double wVal = W[gid];
      double negW = (__longlong_as_double(0x0000000000000000ull) - wVal);
      L[gid] = negW;
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_normalized_laplacian_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_normalized_laplacian_f32(const float* W, const float* D, const unsigned int* n, float* L_sym) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int i = (gid / nVal);
    unsigned int j = (gid % nVal);
    float di = D[i];
    float dj = D[j];
    float eps = __uint_as_float(0x2edbe6ffu);
    bool diZero = (di < eps);
    bool djZero = (dj < eps);
    bool isolated = (diZero || djZero);
    bool isDiag = (i == j);
    if (isolated) {
      if (isDiag) {
        L_sym[gid] = __uint_as_float(0x3f800000u);
      } else {
        L_sym[gid] = __uint_as_float(0x00000000u);
      }
    } else {
      float wVal = W[gid];
      float didjProd = (di * dj);
      float sqrtDiDj = sqrt(didjProd);
      float normalized = (wVal / sqrtDiDj);
      if (isDiag) {
        float lSymVal = (__uint_as_float(0x3f800000u) - normalized);
        L_sym[gid] = lSymVal;
      } else {
        float lSymVal = (__uint_as_float(0x00000000u) - normalized);
        L_sym[gid] = lSymVal;
      }
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_normalized_laplacian_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_normalized_laplacian_f64(const double* W, const double* D, const unsigned int* n, double* L_sym) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int i = (gid / nVal);
    unsigned int j = (gid % nVal);
    double di = D[i];
    double dj = D[j];
    double eps = __longlong_as_double(0x3cd203af9ee75616ull);
    bool diZero = (di < eps);
    bool djZero = (dj < eps);
    bool isolated = (diZero || djZero);
    bool isDiag = (i == j);
    if (isolated) {
      if (isDiag) {
        L_sym[gid] = __longlong_as_double(0x3ff0000000000000ull);
      } else {
        L_sym[gid] = __longlong_as_double(0x0000000000000000ull);
      }
    } else {
      double wVal = W[gid];
      double didjProd = (di * dj);
      double sqrtDiDj = sqrt(didjProd);
      double normalized = (wVal / sqrtDiDj);
      if (isDiag) {
        double lSymVal = (__longlong_as_double(0x3ff0000000000000ull) - normalized);
        L_sym[gid] = lSymVal;
      } else {
        double lSymVal = (__longlong_as_double(0x0000000000000000ull) - normalized);
        L_sym[gid] = lSymVal;
      }
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_normalize_eigenvector_rows_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_normalize_eigenvector_rows_f32(const float* V, const unsigned int* n, const unsigned int* k, float* V_norm) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int wgid = ((unsigned int)((int)(blockIdx.x)));
  __shared__ float sdata[256];
  unsigned int nVal = n[0];
  unsigned int kVal = k[0];
  unsigned int rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    unsigned int rowBase = (rowIdx * kVal);
    bool colInBounds = (lid < kVal);
    unsigned int elemIdx = (rowBase + lid);
    if (colInBounds) {
      float vVal = V[elemIdx];
      float vSq = (vVal * vVal);
      sdata[lid] = vSq;
    } else {
      sdata[lid] = __uint_as_float(0x00000000u);
    }
    __syncthreads();
    unsigned int normF32RedStride = 128u;
    for (int normF32RedStep = 0; normF32RedStep < (int)(8); ++normF32RedStep) {
      unsigned int normF32Stride = normF32RedStride;
      if (lid < normF32Stride) {
        float other = sdata[(lid + normF32Stride)];
        float mine = sdata[lid];
        float sum = (mine + other);
        sdata[lid] = sum;
      }
      unsigned int normF32StrideToHalve = normF32RedStride;
      unsigned int normF32NextStride = (normF32StrideToHalve >> 1u);
      normF32RedStride = normF32NextStride;
      __syncthreads();
    }
    if (lid == 0u) {
      float sumSq = sdata[0];
      float normVal = sqrt(sumSq);
      sdata[0] = normVal;
    }
    __syncthreads();
    if (colInBounds) {
      float norm = sdata[0];
      float vVal2 = V[elemIdx];
      float eps = __uint_as_float(0x2edbe6ffu);
      bool normZero = (norm < eps);
      if (normZero) {
        V_norm[elemIdx] = __uint_as_float(0x00000000u);
      } else {
        float normalized = (vVal2 / norm);
        V_norm[elemIdx] = normalized;
      }
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_normalize_eigenvector_rows_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_normalize_eigenvector_rows_f64(const double* V, const unsigned int* n, const unsigned int* k, double* V_norm) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int wgid = ((unsigned int)((int)(blockIdx.x)));
  __shared__ double sdata[256];
  unsigned int nVal = n[0];
  unsigned int kVal = k[0];
  unsigned int rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    unsigned int rowBase = (rowIdx * kVal);
    bool colInBounds = (lid < kVal);
    unsigned int elemIdx = (rowBase + lid);
    if (colInBounds) {
      double vVal = V[elemIdx];
      double vSq = (vVal * vVal);
      sdata[lid] = vSq;
    } else {
      sdata[lid] = __longlong_as_double(0x0000000000000000ull);
    }
    __syncthreads();
    unsigned int normF64RedStride = 128u;
    for (int normF64RedStep = 0; normF64RedStep < (int)(8); ++normF64RedStep) {
      unsigned int normF64Stride = normF64RedStride;
      if (lid < normF64Stride) {
        double other = sdata[(lid + normF64Stride)];
        double mine = sdata[lid];
        double sum = (mine + other);
        sdata[lid] = sum;
      }
      unsigned int normF64StrideToHalve = normF64RedStride;
      unsigned int normF64NextStride = (normF64StrideToHalve >> 1u);
      normF64RedStride = normF64NextStride;
      __syncthreads();
    }
    if (lid == 0u) {
      double sumSq = sdata[0];
      double normVal = sqrt(sumSq);
      sdata[0] = normVal;
    }
    __syncthreads();
    if (colInBounds) {
      double norm = sdata[0];
      double vVal2 = V[elemIdx];
      double eps = __longlong_as_double(0x3cd203af9ee75616ull);
      bool normZero = (norm < eps);
      if (normZero) {
        V_norm[elemIdx] = __longlong_as_double(0x0000000000000000ull);
      } else {
        double normalized = (vVal2 / norm);
        V_norm[elemIdx] = normalized;
      }
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_spectral_embedding_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_spectral_embedding_f32(const float* eigenvectors, const unsigned int* n, const unsigned int* k, float* embedding) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int kVal = k[0];
  unsigned int totalElems = (nVal * kVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int row = (gid / kVal);
    unsigned int col = (gid % kVal);
    unsigned int srcIdx = ((col * nVal) + row);
    float eigVal = eigenvectors[srcIdx];
    embedding[gid] = eigVal;
  }
}

)kernel";
const char* k_hip_navatala_graph_spectral_embedding_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_spectral_embedding_f64(const double* eigenvectors, const unsigned int* n, const unsigned int* k, double* embedding) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int kVal = k[0];
  unsigned int totalElems = (nVal * kVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int row = (gid / kVal);
    unsigned int col = (gid % kVal);
    unsigned int srcIdx = ((col * nVal) + row);
    double eigVal = eigenvectors[srcIdx];
    embedding[gid] = eigVal;
  }
}

)kernel";
const char* k_hip_navatala_graph_rmat_generator_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_rmat_generator_f32(const float* random, const float* randomWeights, const float* a, const float* b, const float* c, const float* d, const unsigned int* n, const unsigned int* scale, const unsigned int* numEdges, unsigned int* srcNodes, unsigned int* dstNodes, float* weights, unsigned int* edgeCount) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numE = numEdges[0u];
  bool inBounds = (gid < numE);
  if (inBounds) {
    float probA = a[0u];
    float probB = b[0u];
    float probC = c[0u];
    unsigned int scaleVal = scale[0u];
    float ab = (probA + probB);
    float abc = (ab + probC);
    unsigned int srcAccum = 0u;
    unsigned int dstAccum = 0u;
    for (int level = 0; level < (int)(scaleVal); ++level) {
      unsigned int levelU32 = ((unsigned int)(level));
      unsigned int randIdx = ((gid * scaleVal) + levelU32);
      float r = random[randIdx];
      unsigned int bitPos = ((scaleVal - levelU32) - 1u);
      unsigned int bitMask = (1u << bitPos);
      bool inQ01 = (r >= probA);
      bool inQ10 = (r >= ab);
      bool inQ11 = (r >= abc);
      unsigned int srcBit = ((inQ10) ? (bitMask) : (0u));
      unsigned int currentSrc = srcAccum;
      unsigned int newSrc = (currentSrc | srcBit);
      srcAccum = newSrc;
      bool dstBitCond = ((inQ01 && (!inQ10)) || inQ11);
      unsigned int dstBit = ((dstBitCond) ? (bitMask) : (0u));
      unsigned int currentDst = dstAccum;
      unsigned int newDst = (currentDst | dstBit);
      dstAccum = newDst;
    }
    unsigned int src = srcAccum;
    unsigned int dst = dstAccum;
    float w = randomWeights[gid];
    unsigned int slot = atomicAdd(&(edgeCount[0u]), 1u);
    srcNodes[slot] = src;
    dstNodes[slot] = dst;
    weights[slot] = w;
  }
}

)kernel";
const char* k_hip_navatala_graph_rmat_generator_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_rmat_generator_f64(const double* random, const double* randomWeights, const double* a, const double* b, const double* c, const double* d, const unsigned int* n, const unsigned int* scale, const unsigned int* numEdges, unsigned int* srcNodes, unsigned int* dstNodes, double* weights, unsigned int* edgeCount) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numE = numEdges[0u];
  bool inBounds = (gid < numE);
  if (inBounds) {
    double probA = a[0u];
    double probB = b[0u];
    double probC = c[0u];
    unsigned int scaleVal = scale[0u];
    double ab = (probA + probB);
    double abc = (ab + probC);
    unsigned int srcAccum = 0u;
    unsigned int dstAccum = 0u;
    for (int level = 0; level < (int)(scaleVal); ++level) {
      unsigned int levelU32 = ((unsigned int)(level));
      unsigned int randIdx = ((gid * scaleVal) + levelU32);
      double r = random[randIdx];
      unsigned int bitPos = ((scaleVal - levelU32) - 1u);
      unsigned int bitMask = (1u << bitPos);
      bool inQ01 = (r >= probA);
      bool inQ10 = (r >= ab);
      bool inQ11 = (r >= abc);
      unsigned int srcBit = ((inQ10) ? (bitMask) : (0u));
      unsigned int currentSrc = srcAccum;
      unsigned int newSrc = (currentSrc | srcBit);
      srcAccum = newSrc;
      bool dstBitCond = ((inQ01 && (!inQ10)) || inQ11);
      unsigned int dstBit = ((dstBitCond) ? (bitMask) : (0u));
      unsigned int currentDst = dstAccum;
      unsigned int newDst = (currentDst | dstBit);
      dstAccum = newDst;
    }
    unsigned int src = srcAccum;
    unsigned int dst = dstAccum;
    double w = randomWeights[gid];
    unsigned int slot = atomicAdd(&(edgeCount[0u]), 1u);
    srcNodes[slot] = src;
    dstNodes[slot] = dst;
    weights[slot] = w;
  }
}

)kernel";
const char* k_hip_navatala_graph_erdos_renyi_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_erdos_renyi_f32(const float* random, const float* randomWeights, const float* p, const unsigned int* n, unsigned int* srcNodes, unsigned int* dstNodes, float* weights, unsigned int* edgeCount) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numN = n[0u];
  unsigned int nMinus1 = (numN - 1u);
  unsigned int totalPairs = ((numN * nMinus1) / 2u);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    float prob = p[0u];
    float gidF = ((float)(gid));
    float discriminant = (__uint_as_float(0x3f800000u) + (__uint_as_float(0x41000000u) * gidF));
    float sqrtDisc = sqrt(discriminant);
    float iFloat = ((__uint_as_float(0x3f800000u) + sqrtDisc) / __uint_as_float(0x40000000u));
    unsigned int i = ((unsigned int)(floor(iFloat)));
    unsigned int iTriangle = ((i * (i - 1u)) / 2u);
    unsigned int j = (gid - iTriangle);
    float r = random[gid];
    bool includeEdge = (r < prob);
    if (includeEdge) {
      float w = randomWeights[gid];
      unsigned int slot = atomicAdd(&(edgeCount[0u]), 1u);
      srcNodes[slot] = j;
      dstNodes[slot] = i;
      weights[slot] = w;
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_erdos_renyi_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_erdos_renyi_f64(const double* random, const double* randomWeights, const double* p, const unsigned int* n, unsigned int* srcNodes, unsigned int* dstNodes, double* weights, unsigned int* edgeCount) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numN = n[0u];
  unsigned int nMinus1 = (numN - 1u);
  unsigned int totalPairs = ((numN * nMinus1) / 2u);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    double prob = p[0u];
    double gidF = ((double)(gid));
    double discriminant = (__longlong_as_double(0x3ff0000000000000ull) + (__longlong_as_double(0x4020000000000000ull) * gidF));
    double sqrtDisc = sqrt(discriminant);
    double iFloat = ((__longlong_as_double(0x3ff0000000000000ull) + sqrtDisc) / __longlong_as_double(0x4000000000000000ull));
    unsigned int i = ((unsigned int)(floor(iFloat)));
    unsigned int iTriangle = ((i * (i - 1u)) / 2u);
    unsigned int j = (gid - iTriangle);
    double r = random[gid];
    bool includeEdge = (r < prob);
    if (includeEdge) {
      double w = randomWeights[gid];
      unsigned int slot = atomicAdd(&(edgeCount[0u]), 1u);
      srcNodes[slot] = j;
      dstNodes[slot] = i;
      weights[slot] = w;
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_barabasi_albert_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_barabasi_albert_f32(const unsigned int* cumDegree, const float* random, const float* randomWeights, const unsigned int* newNode, const unsigned int* totalDegree, const unsigned int* m, const unsigned int* numExistingNodes, unsigned int* srcNodes, unsigned int* dstNodes, float* weights, unsigned int* edgeCount) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numM = m[0u];
  bool inBounds = (gid < numM);
  if (inBounds) {
    unsigned int newNodeIdx = newNode[0u];
    unsigned int totalDeg = totalDegree[0u];
    unsigned int numExisting = numExistingNodes[0u];
    float r = random[gid];
    float totalDegF = ((float)(totalDeg));
    float targetDegF = (r * totalDegF);
    unsigned int targetDeg = ((unsigned int)(targetDegF));
    unsigned int loAccum = 0u;
    unsigned int hiAccum = numExisting;
    for (int _ = 0; _ < (int)(32u); ++_) {
      unsigned int lo = loAccum;
      unsigned int hi = hiAccum;
      bool stillSearching = (lo < hi);
      if (stillSearching) {
        unsigned int mid = ((lo + hi) / 2u);
        unsigned int cumVal = cumDegree[mid];
        bool goRight = (cumVal <= targetDeg);
        unsigned int newLo = ((goRight) ? ((mid + 1u)) : (lo));
        unsigned int newHi = ((goRight) ? (hi) : (mid));
        loAccum = newLo;
        hiAccum = newHi;
      }
    }
    unsigned int finalLo = loAccum;
    unsigned int targetNode = (((finalLo > 0u)) ? ((finalLo - 1u)) : (0u));
    float w = randomWeights[gid];
    unsigned int slot = atomicAdd(&(edgeCount[0u]), 1u);
    srcNodes[slot] = newNodeIdx;
    dstNodes[slot] = targetNode;
    weights[slot] = w;
  }
}

)kernel";
const char* k_hip_navatala_graph_barabasi_albert_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_barabasi_albert_f64(const unsigned int* cumDegree, const double* random, const double* randomWeights, const unsigned int* newNode, const unsigned int* totalDegree, const unsigned int* m, const unsigned int* numExistingNodes, unsigned int* srcNodes, unsigned int* dstNodes, double* weights, unsigned int* edgeCount) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numM = m[0u];
  bool inBounds = (gid < numM);
  if (inBounds) {
    unsigned int newNodeIdx = newNode[0u];
    unsigned int totalDeg = totalDegree[0u];
    unsigned int numExisting = numExistingNodes[0u];
    double r = random[gid];
    double totalDegF = ((double)(totalDeg));
    double targetDegF = (r * totalDegF);
    unsigned int targetDeg = ((unsigned int)(targetDegF));
    unsigned int loAccum = 0u;
    unsigned int hiAccum = numExisting;
    for (int _ = 0; _ < (int)(32u); ++_) {
      unsigned int lo = loAccum;
      unsigned int hi = hiAccum;
      bool stillSearching = (lo < hi);
      if (stillSearching) {
        unsigned int mid = ((lo + hi) / 2u);
        unsigned int cumVal = cumDegree[mid];
        bool goRight = (cumVal <= targetDeg);
        unsigned int newLo = ((goRight) ? ((mid + 1u)) : (lo));
        unsigned int newHi = ((goRight) ? (hi) : (mid));
        loAccum = newLo;
        hiAccum = newHi;
      }
    }
    unsigned int finalLo = loAccum;
    unsigned int targetNode = (((finalLo > 0u)) ? ((finalLo - 1u)) : (0u));
    double w = randomWeights[gid];
    unsigned int slot = atomicAdd(&(edgeCount[0u]), 1u);
    srcNodes[slot] = newNodeIdx;
    dstNodes[slot] = targetNode;
    weights[slot] = w;
  }
}

)kernel";
const char* k_hip_navatala_graph_compute_modularity_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_compute_modularity_f32(const float* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const unsigned int* clusters, const unsigned int* n, const unsigned int* numClusters, const float* totalEdgeWeight, float* modularity) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  float totalWeight = totalEdgeWeight[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int myCluster = clusters[gid];
    unsigned int rowStart = rowPtr[gid];
    unsigned int gidPlus1 = (gid + 1u);
    unsigned int rowEnd = rowPtr[gidPlus1];
    float degreeAccum = __uint_as_float(0x00000000u);
    float intraWeightAccum = __uint_as_float(0x00000000u);
    for (int idx = 0; idx < (int)(rowEnd); ++idx) {
      unsigned int idxU32 = ((unsigned int)(idx));
      bool idxInRange = (idxU32 >= rowStart);
      if (idxInRange) {
        float edgeWeight = adjacencyValues[idxU32];
        unsigned int neighbor = colInd[idxU32];
        unsigned int neighborCluster = clusters[neighbor];
        float currentDegree = degreeAccum;
        float newDegree = (currentDegree + edgeWeight);
        degreeAccum = newDegree;
        bool sameCluster = (neighborCluster == myCluster);
        float weightToAdd = ((sameCluster) ? (edgeWeight) : (__uint_as_float(0x00000000u)));
        float currentIntraWeight = intraWeightAccum;
        float newIntraWeight = (currentIntraWeight + weightToAdd);
        intraWeightAccum = newIntraWeight;
      }
    }
    float degree = degreeAccum;
    float intraWeight = intraWeightAccum;
    float invTotalWeight = (__uint_as_float(0x3f800000u) / totalWeight);
    float eContrib = (intraWeight * invTotalWeight);
    float degreeNorm = (degree * invTotalWeight);
    float aContrib = (degreeNorm * degreeNorm);
    float contribution = (eContrib - aContrib);
    atomicAdd(&modularity[0], contribution);
  }
}

)kernel";
const char* k_hip_navatala_graph_compute_modularity_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_compute_modularity_f64(const double* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const unsigned int* clusters, const unsigned int* n, const unsigned int* numClusters, const double* totalEdgeWeight, double* modularity) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  double totalWeight = totalEdgeWeight[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int myCluster = clusters[gid];
    unsigned int rowStart = rowPtr[gid];
    unsigned int gidPlus1 = (gid + 1u);
    unsigned int rowEnd = rowPtr[gidPlus1];
    double degreeAccum = __longlong_as_double(0x0000000000000000ull);
    double intraWeightAccum = __longlong_as_double(0x0000000000000000ull);
    for (int idx = 0; idx < (int)(rowEnd); ++idx) {
      unsigned int idxU32 = ((unsigned int)(idx));
      bool idxInRange = (idxU32 >= rowStart);
      if (idxInRange) {
        double edgeWeight = adjacencyValues[idxU32];
        unsigned int neighbor = colInd[idxU32];
        unsigned int neighborCluster = clusters[neighbor];
        double currentDegree = degreeAccum;
        double newDegree = (currentDegree + edgeWeight);
        degreeAccum = newDegree;
        bool sameCluster = (neighborCluster == myCluster);
        double weightToAdd = ((sameCluster) ? (edgeWeight) : (__longlong_as_double(0x0000000000000000ull)));
        double currentIntraWeight = intraWeightAccum;
        double newIntraWeight = (currentIntraWeight + weightToAdd);
        intraWeightAccum = newIntraWeight;
      }
    }
    double degree = degreeAccum;
    double intraWeight = intraWeightAccum;
    double invTotalWeight = (__longlong_as_double(0x3ff0000000000000ull) / totalWeight);
    double eContrib = (intraWeight * invTotalWeight);
    double degreeNorm = (degree * invTotalWeight);
    double aContrib = (degreeNorm * degreeNorm);
    double contribution = (eContrib - aContrib);
    atomicAdd(&modularity[0], contribution);
  }
}

)kernel";
const char* k_hip_navatala_graph_modularity_matrix_vec_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_modularity_matrix_vec_f32(const float* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const float* x, const float* degree, const float* totalEdgeWeight, const unsigned int* n, const float* dotDX, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  float totalWeight = totalEdgeWeight[0];
  float dotDXVal = dotDX[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int rowStart = rowPtr[gid];
    unsigned int gidPlus1 = (gid + 1u);
    unsigned int rowEnd = rowPtr[gidPlus1];
    float axValAccum = __uint_as_float(0x00000000u);
    for (int idx = 0; idx < (int)(rowEnd); ++idx) {
      unsigned int idxU32 = ((unsigned int)(idx));
      bool idxInRange = (idxU32 >= rowStart);
      if (idxInRange) {
        float aij = adjacencyValues[idxU32];
        unsigned int j = colInd[idxU32];
        float xj = x[j];
        float product = (aij * xj);
        float currentAxVal = axValAccum;
        float newAxVal = (currentAxVal + product);
        axValAccum = newAxVal;
      }
    }
    float axVal = axValAccum;
    float di = degree[gid];
    float correction = ((di * dotDXVal) / totalWeight);
    float yVal = (axVal - correction);
    y[gid] = yVal;
  }
}

)kernel";
const char* k_hip_navatala_graph_modularity_matrix_vec_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_modularity_matrix_vec_f64(const double* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const double* x, const double* degree, const double* totalEdgeWeight, const unsigned int* n, const double* dotDX, double* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  double totalWeight = totalEdgeWeight[0];
  double dotDXVal = dotDX[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int rowStart = rowPtr[gid];
    unsigned int gidPlus1 = (gid + 1u);
    unsigned int rowEnd = rowPtr[gidPlus1];
    double axValAccum = __longlong_as_double(0x0000000000000000ull);
    for (int idx = 0; idx < (int)(rowEnd); ++idx) {
      unsigned int idxU32 = ((unsigned int)(idx));
      bool idxInRange = (idxU32 >= rowStart);
      if (idxInRange) {
        double aij = adjacencyValues[idxU32];
        unsigned int j = colInd[idxU32];
        double xj = x[j];
        double product = (aij * xj);
        double currentAxVal = axValAccum;
        double newAxVal = (currentAxVal + product);
        axValAccum = newAxVal;
      }
    }
    double axVal = axValAccum;
    double di = degree[gid];
    double correction = ((di * dotDXVal) / totalWeight);
    double yVal = (axVal - correction);
    y[gid] = yVal;
  }
}

)kernel";
const char* k_hip_navatala_graph_construct_indicator_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_construct_indicator_f32(const unsigned int* clusters, const unsigned int* clusterIndex, const unsigned int* n, float* indicator) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int targetCluster = clusterIndex[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int nodeCluster = clusters[gid];
    bool isTarget = (nodeCluster == targetCluster);
    float indicatorVal = ((isTarget) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    indicator[gid] = indicatorVal;
  }
}

)kernel";
const char* k_hip_navatala_graph_construct_indicator_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_construct_indicator_f64(const unsigned int* clusters, const unsigned int* clusterIndex, const unsigned int* n, double* indicator) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int targetCluster = clusterIndex[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int nodeCluster = clusters[gid];
    bool isTarget = (nodeCluster == targetCluster);
    double indicatorVal = ((isTarget) ? (__longlong_as_double(0x3ff0000000000000ull)) : (__longlong_as_double(0x0000000000000000ull)));
    indicator[gid] = indicatorVal;
  }
}

)kernel";
const char* k_hip_navatala_graph_compute_edge_cut_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_compute_edge_cut_f32(const float* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const unsigned int* clusters, const unsigned int* n, float* edgeCut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int myCluster = clusters[gid];
    unsigned int rowStart = rowPtr[gid];
    unsigned int gidPlus1 = (gid + 1u);
    unsigned int rowEnd = rowPtr[gidPlus1];
    float localCut = __uint_as_float(0x00000000u);
    for (int edgeIdx = 0; edgeIdx < (int)(rowEnd); ++edgeIdx) {
      unsigned int edgeIdxU = ((unsigned int)(edgeIdx));
      bool edgeInRange = (edgeIdxU >= rowStart);
      if (edgeInRange) {
        unsigned int neighbor = colInd[edgeIdxU];
        unsigned int neighborCluster = clusters[neighbor];
        bool crossesBoundary = (myCluster != neighborCluster);
        if (crossesBoundary) {
          float edgeWeight = adjacencyValues[edgeIdxU];
          float currentCut = localCut;
          float newCut = (currentCut + edgeWeight);
          localCut = newCut;
        }
      }
    }
    float finalLocalCut = localCut;
    bool hasContribution = (finalLocalCut > __uint_as_float(0x00000000u));
    if (hasContribution) {
      atomicAdd(&edgeCut[0], finalLocalCut);
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_compute_edge_cut_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_compute_edge_cut_f64(const double* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const unsigned int* clusters, const unsigned int* n, double* edgeCut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int myCluster = clusters[gid];
    unsigned int rowStart = rowPtr[gid];
    unsigned int gidPlus1 = (gid + 1u);
    unsigned int rowEnd = rowPtr[gidPlus1];
    double localCut = __longlong_as_double(0x0000000000000000ull);
    for (int edgeIdx = 0; edgeIdx < (int)(rowEnd); ++edgeIdx) {
      unsigned int edgeIdxU = ((unsigned int)(edgeIdx));
      bool edgeInRange = (edgeIdxU >= rowStart);
      if (edgeInRange) {
        unsigned int neighbor = colInd[edgeIdxU];
        unsigned int neighborCluster = clusters[neighbor];
        bool crossesBoundary = (myCluster != neighborCluster);
        if (crossesBoundary) {
          double edgeWeight = adjacencyValues[edgeIdxU];
          double currentCut = localCut;
          double newCut = (currentCut + edgeWeight);
          localCut = newCut;
        }
      }
    }
    double finalLocalCut = localCut;
    bool hasContribution = (finalLocalCut > __longlong_as_double(0x0000000000000000ull));
    if (hasContribution) {
      atomicAdd(&edgeCut[0], finalLocalCut);
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_compute_partition_cost_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_compute_partition_cost_f32(const float* edgeCutPerCluster, const unsigned int* clusterSizes, const unsigned int* numClusters, float* cost) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numClustersVal = numClusters[0];
  bool inBounds = (gid < numClustersVal);
  if (inBounds) {
    float clusterEdgeCut = edgeCutPerCluster[gid];
    unsigned int clusterSize = clusterSizes[gid];
    bool sizeIsZero = (clusterSize == 0u);
    float clusterSizeF = ((float)(clusterSize));
    float normalizedCost = ((sizeIsZero) ? (__uint_as_float(0x00000000u)) : ((clusterEdgeCut / clusterSizeF)));
    bool hasContribution = (normalizedCost > __uint_as_float(0x00000000u));
    if (hasContribution) {
      atomicAdd(&cost[0], normalizedCost);
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_compute_partition_cost_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_compute_partition_cost_f64(const double* edgeCutPerCluster, const unsigned int* clusterSizes, const unsigned int* numClusters, double* cost) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numClustersVal = numClusters[0];
  bool inBounds = (gid < numClustersVal);
  if (inBounds) {
    double clusterEdgeCut = edgeCutPerCluster[gid];
    unsigned int clusterSize = clusterSizes[gid];
    bool sizeIsZero = (clusterSize == 0u);
    double clusterSizeF = ((double)(clusterSize));
    double normalizedCost = ((sizeIsZero) ? (__longlong_as_double(0x0000000000000000ull)) : ((clusterEdgeCut / clusterSizeF)));
    bool hasContribution = (normalizedCost > __longlong_as_double(0x0000000000000000ull));
    if (hasContribution) {
      atomicAdd(&cost[0], normalizedCost);
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_analyze_partition_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_analyze_partition_f32(const float* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const unsigned int* clusters, const unsigned int* n, const unsigned int* numClusters, float* edgeCut, float* cost) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int numClustersVal = numClusters[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int myCluster = clusters[gid];
    unsigned int rowStart = rowPtr[gid];
    unsigned int gidPlus1 = (gid + 1u);
    unsigned int rowEnd = rowPtr[gidPlus1];
    float nodeEdgeCut = __uint_as_float(0x00000000u);
    for (int edgeIdx = 0; edgeIdx < (int)(rowEnd); ++edgeIdx) {
      unsigned int edgeIdxU = ((unsigned int)(edgeIdx));
      bool edgeInRange = (edgeIdxU >= rowStart);
      if (edgeInRange) {
        unsigned int neighbor = colInd[edgeIdxU];
        unsigned int neighborCluster = clusters[neighbor];
        bool crossesBoundary = (myCluster != neighborCluster);
        if (crossesBoundary) {
          float edgeWeight = adjacencyValues[edgeIdxU];
          float currentEdgeCut = nodeEdgeCut;
          float newEdgeCut = (currentEdgeCut + edgeWeight);
          nodeEdgeCut = newEdgeCut;
        }
      }
    }
    float finalNodeEdgeCut = nodeEdgeCut;
    bool hasEdgeCut = (finalNodeEdgeCut > __uint_as_float(0x00000000u));
    if (hasEdgeCut) {
      atomicAdd(&edgeCut[0], finalNodeEdgeCut);
    }
    float nValF = ((float)(nVal));
    float nodeCostContrib = (finalNodeEdgeCut / nValF);
    bool hasCostContrib = (nodeCostContrib > __uint_as_float(0x00000000u));
    if (hasCostContrib) {
      atomicAdd(&cost[0], nodeCostContrib);
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_analyze_partition_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_analyze_partition_f64(const double* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const unsigned int* clusters, const unsigned int* n, const unsigned int* numClusters, double* edgeCut, double* cost) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int numClustersVal = numClusters[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int myCluster = clusters[gid];
    unsigned int rowStart = rowPtr[gid];
    unsigned int gidPlus1 = (gid + 1u);
    unsigned int rowEnd = rowPtr[gidPlus1];
    double nodeEdgeCut = __longlong_as_double(0x0000000000000000ull);
    for (int edgeIdx = 0; edgeIdx < (int)(rowEnd); ++edgeIdx) {
      unsigned int edgeIdxU = ((unsigned int)(edgeIdx));
      bool edgeInRange = (edgeIdxU >= rowStart);
      if (edgeInRange) {
        unsigned int neighbor = colInd[edgeIdxU];
        unsigned int neighborCluster = clusters[neighbor];
        bool crossesBoundary = (myCluster != neighborCluster);
        if (crossesBoundary) {
          double edgeWeight = adjacencyValues[edgeIdxU];
          double currentEdgeCut = nodeEdgeCut;
          double newEdgeCut = (currentEdgeCut + edgeWeight);
          nodeEdgeCut = newEdgeCut;
        }
      }
    }
    double finalNodeEdgeCut = nodeEdgeCut;
    bool hasEdgeCut = (finalNodeEdgeCut > __longlong_as_double(0x0000000000000000ull));
    if (hasEdgeCut) {
      atomicAdd(&edgeCut[0], finalNodeEdgeCut);
    }
    double nValF = ((double)(nVal));
    double nodeCostContrib = (finalNodeEdgeCut / nValF);
    bool hasCostContrib = (nodeCostContrib > __longlong_as_double(0x0000000000000000ull));
    if (hasCostContrib) {
      atomicAdd(&cost[0], nodeCostContrib);
    }
  }
}

)kernel";
const char* k_hip_navatala_graph_transform_eigen_matrix_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_transform_eigen_matrix_f32(const float* eigenvectors, const unsigned int* n, const unsigned int* k, float* transformed) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int kVal = k[0];
  unsigned int totalElems = (kVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int eigIdx = (gid / nVal);
    unsigned int sampleIdx = (gid % nVal);
    unsigned int srcIdx = ((sampleIdx * kVal) + eigIdx);
    float val = eigenvectors[srcIdx];
    transformed[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_graph_transform_eigen_matrix_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_transform_eigen_matrix_f64(const double* eigenvectors, const unsigned int* n, const unsigned int* k, double* transformed) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int kVal = k[0];
  unsigned int totalElems = (kVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int eigIdx = (gid / nVal);
    unsigned int sampleIdx = (gid % nVal);
    unsigned int srcIdx = ((sampleIdx * kVal) + eigIdx);
    double val = eigenvectors[srcIdx];
    transformed[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_graph_compute_cluster_sizes_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_compute_cluster_sizes_f32(const unsigned int* clusters, const unsigned int* n, const unsigned int* numClusters, float* sizes) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int cluster = clusters[gid];
    float one = __uint_as_float(0x3f800000u);
    unsigned int addr = cluster;
    float dummy = atomicAdd(&(sizes[addr]), one);
  }
}

)kernel";
const char* k_hip_navatala_graph_compute_cluster_sizes_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_compute_cluster_sizes_f64(const unsigned int* clusters, const unsigned int* n, const unsigned int* numClusters, double* sizes) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int cluster = clusters[gid];
    double one = __longlong_as_double(0x3ff0000000000000ull);
    unsigned int addr = cluster;
    double dummy = atomicAdd(&(sizes[addr]), one);
  }
}

)kernel";
const char* k_hip_navatala_graph_laplacian_from_adjacency_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_laplacian_from_adjacency_f32(const float* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const unsigned int* n, float* laplacianValues, float* degree) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int rowIdx = gid;
  bool inBounds = (rowIdx < nVal);
  if (inBounds) {
    unsigned int rowStart = rowPtr[rowIdx];
    unsigned int one32 = ((unsigned int)(1));
    unsigned int rowIdxPlusOne = (rowIdx + one32);
    unsigned int rowEnd = rowPtr[rowIdxPlusOne];
    float degreeSumAccum = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)((rowEnd - rowStart)); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int absIdx = (rowStart + jU32);
      float adjVal = adjacencyValues[absIdx];
      float negAdjVal = (__uint_as_float(0x00000000u) - adjVal);
      laplacianValues[absIdx] = negAdjVal;
      float currentDegree = degreeSumAccum;
      float newDegree = (currentDegree + adjVal);
      degreeSumAccum = newDegree;
    }
    float finalDegree = degreeSumAccum;
    degree[rowIdx] = finalDegree;
  }
}

)kernel";
const char* k_hip_navatala_graph_laplacian_from_adjacency_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_laplacian_from_adjacency_f64(const double* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const unsigned int* n, double* laplacianValues, double* degree) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int rowIdx = gid;
  bool inBounds = (rowIdx < nVal);
  if (inBounds) {
    unsigned int rowStart = rowPtr[rowIdx];
    unsigned int one32 = ((unsigned int)(1));
    unsigned int rowIdxPlusOne = (rowIdx + one32);
    unsigned int rowEnd = rowPtr[rowIdxPlusOne];
    double degreeSumAccum = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)((rowEnd - rowStart)); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int absIdx = (rowStart + jU32);
      double adjVal = adjacencyValues[absIdx];
      double negAdjVal = (__longlong_as_double(0x0000000000000000ull) - adjVal);
      laplacianValues[absIdx] = negAdjVal;
      double currentDegree = degreeSumAccum;
      double newDegree = (currentDegree + adjVal);
      degreeSumAccum = newDegree;
    }
    double finalDegree = degreeSumAccum;
    degree[rowIdx] = finalDegree;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_pack_f32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_pack_f32 = {
  1,
  "navatala_graph_pack_f32",
  "hip",
  "navatala_graph_pack_f32",
  "kernel:hip:navatala_graph_pack_f32",
  "abi-r1:hip:navatala_graph_pack_f32",
  "abi-r1:hip:navatala_graph_pack_f32",
  4,
  kAbiArgs_hip_navatala_graph_pack_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_pack_f64[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_pack_f64 = {
  1,
  "navatala_graph_pack_f64",
  "hip",
  "navatala_graph_pack_f64",
  "kernel:hip:navatala_graph_pack_f64",
  "abi-r1:hip:navatala_graph_pack_f64",
  "abi-r1:hip:navatala_graph_pack_f64",
  4,
  kAbiArgs_hip_navatala_graph_pack_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_pack_i32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_pack_i32 = {
  1,
  "navatala_graph_pack_i32",
  "hip",
  "navatala_graph_pack_i32",
  "kernel:hip:navatala_graph_pack_i32",
  "abi-r1:hip:navatala_graph_pack_i32",
  "abi-r1:hip:navatala_graph_pack_i32",
  4,
  kAbiArgs_hip_navatala_graph_pack_i32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_pack_u32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_pack_u32 = {
  1,
  "navatala_graph_pack_u32",
  "hip",
  "navatala_graph_pack_u32",
  "kernel:hip:navatala_graph_pack_u32",
  "abi-r1:hip:navatala_graph_pack_u32",
  "abi-r1:hip:navatala_graph_pack_u32",
  4,
  kAbiArgs_hip_navatala_graph_pack_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_unpack_f32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_unpack_f32 = {
  1,
  "navatala_graph_unpack_f32",
  "hip",
  "navatala_graph_unpack_f32",
  "kernel:hip:navatala_graph_unpack_f32",
  "abi-r1:hip:navatala_graph_unpack_f32",
  "abi-r1:hip:navatala_graph_unpack_f32",
  4,
  kAbiArgs_hip_navatala_graph_unpack_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_unpack_f64[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_unpack_f64 = {
  1,
  "navatala_graph_unpack_f64",
  "hip",
  "navatala_graph_unpack_f64",
  "kernel:hip:navatala_graph_unpack_f64",
  "abi-r1:hip:navatala_graph_unpack_f64",
  "abi-r1:hip:navatala_graph_unpack_f64",
  4,
  kAbiArgs_hip_navatala_graph_unpack_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_unpack_i32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_unpack_i32 = {
  1,
  "navatala_graph_unpack_i32",
  "hip",
  "navatala_graph_unpack_i32",
  "kernel:hip:navatala_graph_unpack_i32",
  "abi-r1:hip:navatala_graph_unpack_i32",
  "abi-r1:hip:navatala_graph_unpack_i32",
  4,
  kAbiArgs_hip_navatala_graph_unpack_i32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_unpack_u32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_unpack_u32 = {
  1,
  "navatala_graph_unpack_u32",
  "hip",
  "navatala_graph_unpack_u32",
  "kernel:hip:navatala_graph_unpack_u32",
  "abi-r1:hip:navatala_graph_unpack_u32",
  "abi-r1:hip:navatala_graph_unpack_u32",
  4,
  kAbiArgs_hip_navatala_graph_unpack_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_pack_f32x3[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_pack_f32x3 = {
  1,
  "navatala_graph_pack_f32x3",
  "hip",
  "navatala_graph_pack_f32x3",
  "kernel:hip:navatala_graph_pack_f32x3",
  "abi-r1:hip:navatala_graph_pack_f32x3",
  "abi-r1:hip:navatala_graph_pack_f32x3",
  5,
  kAbiArgs_hip_navatala_graph_pack_f32x3
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_unpack_f32x3[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_unpack_f32x3 = {
  1,
  "navatala_graph_unpack_f32x3",
  "hip",
  "navatala_graph_unpack_f32x3",
  "kernel:hip:navatala_graph_unpack_f32x3",
  "abi-r1:hip:navatala_graph_unpack_f32x3",
  "abi-r1:hip:navatala_graph_unpack_f32x3",
  5,
  kAbiArgs_hip_navatala_graph_unpack_f32x3
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_pack_f32x6[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_pack_f32x6 = {
  1,
  "navatala_graph_pack_f32x6",
  "hip",
  "navatala_graph_pack_f32x6",
  "kernel:hip:navatala_graph_pack_f32x6",
  "abi-r1:hip:navatala_graph_pack_f32x6",
  "abi-r1:hip:navatala_graph_pack_f32x6",
  5,
  kAbiArgs_hip_navatala_graph_pack_f32x6
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_unpack_f32x6[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_unpack_f32x6 = {
  1,
  "navatala_graph_unpack_f32x6",
  "hip",
  "navatala_graph_unpack_f32x6",
  "kernel:hip:navatala_graph_unpack_f32x6",
  "abi-r1:hip:navatala_graph_unpack_f32x6",
  "abi-r1:hip:navatala_graph_unpack_f32x6",
  5,
  kAbiArgs_hip_navatala_graph_unpack_f32x6
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_pack_f32x9[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_pack_f32x9 = {
  1,
  "navatala_graph_pack_f32x9",
  "hip",
  "navatala_graph_pack_f32x9",
  "kernel:hip:navatala_graph_pack_f32x9",
  "abi-r1:hip:navatala_graph_pack_f32x9",
  "abi-r1:hip:navatala_graph_pack_f32x9",
  5,
  kAbiArgs_hip_navatala_graph_pack_f32x9
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_unpack_f32x9[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_unpack_f32x9 = {
  1,
  "navatala_graph_unpack_f32x9",
  "hip",
  "navatala_graph_unpack_f32x9",
  "kernel:hip:navatala_graph_unpack_f32x9",
  "abi-r1:hip:navatala_graph_unpack_f32x9",
  "abi-r1:hip:navatala_graph_unpack_f32x9",
  5,
  kAbiArgs_hip_navatala_graph_unpack_f32x9
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_pack_f64x3[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_pack_f64x3 = {
  1,
  "navatala_graph_pack_f64x3",
  "hip",
  "navatala_graph_pack_f64x3",
  "kernel:hip:navatala_graph_pack_f64x3",
  "abi-r1:hip:navatala_graph_pack_f64x3",
  "abi-r1:hip:navatala_graph_pack_f64x3",
  5,
  kAbiArgs_hip_navatala_graph_pack_f64x3
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_unpack_f64x3[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_unpack_f64x3 = {
  1,
  "navatala_graph_unpack_f64x3",
  "hip",
  "navatala_graph_unpack_f64x3",
  "kernel:hip:navatala_graph_unpack_f64x3",
  "abi-r1:hip:navatala_graph_unpack_f64x3",
  "abi-r1:hip:navatala_graph_unpack_f64x3",
  5,
  kAbiArgs_hip_navatala_graph_unpack_f64x3
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_pack_f64x6[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_pack_f64x6 = {
  1,
  "navatala_graph_pack_f64x6",
  "hip",
  "navatala_graph_pack_f64x6",
  "kernel:hip:navatala_graph_pack_f64x6",
  "abi-r1:hip:navatala_graph_pack_f64x6",
  "abi-r1:hip:navatala_graph_pack_f64x6",
  5,
  kAbiArgs_hip_navatala_graph_pack_f64x6
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_unpack_f64x6[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_unpack_f64x6 = {
  1,
  "navatala_graph_unpack_f64x6",
  "hip",
  "navatala_graph_unpack_f64x6",
  "kernel:hip:navatala_graph_unpack_f64x6",
  "abi-r1:hip:navatala_graph_unpack_f64x6",
  "abi-r1:hip:navatala_graph_unpack_f64x6",
  5,
  kAbiArgs_hip_navatala_graph_unpack_f64x6
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_pack_f64x9[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_pack_f64x9 = {
  1,
  "navatala_graph_pack_f64x9",
  "hip",
  "navatala_graph_pack_f64x9",
  "kernel:hip:navatala_graph_pack_f64x9",
  "abi-r1:hip:navatala_graph_pack_f64x9",
  "abi-r1:hip:navatala_graph_pack_f64x9",
  5,
  kAbiArgs_hip_navatala_graph_pack_f64x9
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_unpack_f64x9[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_unpack_f64x9 = {
  1,
  "navatala_graph_unpack_f64x9",
  "hip",
  "navatala_graph_unpack_f64x9",
  "kernel:hip:navatala_graph_unpack_f64x9",
  "abi-r1:hip:navatala_graph_unpack_f64x9",
  "abi-r1:hip:navatala_graph_unpack_f64x9",
  5,
  kAbiArgs_hip_navatala_graph_unpack_f64x9
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_degree_out_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_degree_out_u32 = {
  1,
  "navatala_graph_degree_out_u32",
  "hip",
  "navatala_graph_degree_out_u32",
  "kernel:hip:navatala_graph_degree_out_u32",
  "abi-r1:hip:navatala_graph_degree_out_u32",
  "abi-r1:hip:navatala_graph_degree_out_u32",
  3,
  kAbiArgs_hip_navatala_graph_degree_out_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_degree_in_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_degree_in_u32 = {
  1,
  "navatala_graph_degree_in_u32",
  "hip",
  "navatala_graph_degree_in_u32",
  "kernel:hip:navatala_graph_degree_in_u32",
  "abi-r1:hip:navatala_graph_degree_in_u32",
  "abi-r1:hip:navatala_graph_degree_in_u32",
  3,
  kAbiArgs_hip_navatala_graph_degree_in_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_spmv_weighted_f32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_spmv_weighted_f32 = {
  1,
  "navatala_graph_spmv_weighted_f32",
  "hip",
  "navatala_graph_spmv_weighted_f32",
  "kernel:hip:navatala_graph_spmv_weighted_f32",
  "abi-r1:hip:navatala_graph_spmv_weighted_f32",
  "abi-r1:hip:navatala_graph_spmv_weighted_f32",
  6,
  kAbiArgs_hip_navatala_graph_spmv_weighted_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_spmv_weighted_subgroup_f32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_spmv_weighted_subgroup_f32 = {
  1,
  "navatala_graph_spmv_weighted_subgroup_f32",
  "hip",
  "navatala_graph_spmv_weighted_subgroup_f32",
  "kernel:hip:navatala_graph_spmv_weighted_subgroup_f32",
  "abi-r1:hip:navatala_graph_spmv_weighted_subgroup_f32",
  "abi-r1:hip:navatala_graph_spmv_weighted_subgroup_f32",
  6,
  kAbiArgs_hip_navatala_graph_spmv_weighted_subgroup_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_spmv_unweighted_f32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_spmv_unweighted_f32 = {
  1,
  "navatala_graph_spmv_unweighted_f32",
  "hip",
  "navatala_graph_spmv_unweighted_f32",
  "kernel:hip:navatala_graph_spmv_unweighted_f32",
  "abi-r1:hip:navatala_graph_spmv_unweighted_f32",
  "abi-r1:hip:navatala_graph_spmv_unweighted_f32",
  5,
  kAbiArgs_hip_navatala_graph_spmv_unweighted_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_spmv_unweighted_subgroup_f32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_spmv_unweighted_subgroup_f32 = {
  1,
  "navatala_graph_spmv_unweighted_subgroup_f32",
  "hip",
  "navatala_graph_spmv_unweighted_subgroup_f32",
  "kernel:hip:navatala_graph_spmv_unweighted_subgroup_f32",
  "abi-r1:hip:navatala_graph_spmv_unweighted_subgroup_f32",
  "abi-r1:hip:navatala_graph_spmv_unweighted_subgroup_f32",
  5,
  kAbiArgs_hip_navatala_graph_spmv_unweighted_subgroup_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_axpy2_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "z", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_axpy2_f32 = {
  1,
  "navatala_graph_axpy2_f32",
  "hip",
  "navatala_graph_axpy2_f32",
  "kernel:hip:navatala_graph_axpy2_f32",
  "abi-r1:hip:navatala_graph_axpy2_f32",
  "abi-r1:hip:navatala_graph_axpy2_f32",
  6,
  kAbiArgs_hip_navatala_graph_axpy2_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_scale_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_scale_f32 = {
  1,
  "navatala_graph_scale_f32",
  "hip",
  "navatala_graph_scale_f32",
  "kernel:hip:navatala_graph_scale_f32",
  "abi-r1:hip:navatala_graph_scale_f32",
  "abi-r1:hip:navatala_graph_scale_f32",
  4,
  kAbiArgs_hip_navatala_graph_scale_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_scale_add_bias_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "bias", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_scale_add_bias_f32 = {
  1,
  "navatala_graph_scale_add_bias_f32",
  "hip",
  "navatala_graph_scale_add_bias_f32",
  "kernel:hip:navatala_graph_scale_add_bias_f32",
  "abi-r1:hip:navatala_graph_scale_add_bias_f32",
  "abi-r1:hip:navatala_graph_scale_add_bias_f32",
  5,
  kAbiArgs_hip_navatala_graph_scale_add_bias_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_l1norm_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_l1norm_f32 = {
  1,
  "navatala_graph_l1norm_f32",
  "hip",
  "navatala_graph_l1norm_f32",
  "kernel:hip:navatala_graph_l1norm_f32",
  "abi-r1:hip:navatala_graph_l1norm_f32",
  "abi-r1:hip:navatala_graph_l1norm_f32",
  3,
  kAbiArgs_hip_navatala_graph_l1norm_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_recip_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_recip_f32 = {
  1,
  "navatala_graph_recip_f32",
  "hip",
  "navatala_graph_recip_f32",
  "kernel:hip:navatala_graph_recip_f32",
  "abi-r1:hip:navatala_graph_recip_f32",
  "abi-r1:hip:navatala_graph_recip_f32",
  2,
  kAbiArgs_hip_navatala_graph_recip_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_l1_diff_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_l1_diff_f32 = {
  1,
  "navatala_graph_l1_diff_f32",
  "hip",
  "navatala_graph_l1_diff_f32",
  "kernel:hip:navatala_graph_l1_diff_f32",
  "abi-r1:hip:navatala_graph_l1_diff_f32",
  "abi-r1:hip:navatala_graph_l1_diff_f32",
  4,
  kAbiArgs_hip_navatala_graph_l1_diff_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_count_diff_u32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_count_diff_u32 = {
  1,
  "navatala_graph_count_diff_u32",
  "hip",
  "navatala_graph_count_diff_u32",
  "kernel:hip:navatala_graph_count_diff_u32",
  "abi-r1:hip:navatala_graph_count_diff_u32",
  "abi-r1:hip:navatala_graph_count_diff_u32",
  4,
  kAbiArgs_hip_navatala_graph_count_diff_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_copy_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_copy_f32 = {
  1,
  "navatala_graph_copy_f32",
  "hip",
  "navatala_graph_copy_f32",
  "kernel:hip:navatala_graph_copy_f32",
  "abi-r1:hip:navatala_graph_copy_f32",
  "abi-r1:hip:navatala_graph_copy_f32",
  3,
  kAbiArgs_hip_navatala_graph_copy_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_copy_u32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_copy_u32 = {
  1,
  "navatala_graph_copy_u32",
  "hip",
  "navatala_graph_copy_u32",
  "kernel:hip:navatala_graph_copy_u32",
  "abi-r1:hip:navatala_graph_copy_u32",
  "abi-r1:hip:navatala_graph_copy_u32",
  3,
  kAbiArgs_hip_navatala_graph_copy_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_relax_vertex_pull_f32[] = {
  { "inOffsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "inSrcs", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "inWeights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dist", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "inf", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_relax_vertex_pull_f32 = {
  1,
  "navatala_graph_relax_vertex_pull_f32",
  "hip",
  "navatala_graph_relax_vertex_pull_f32",
  "kernel:hip:navatala_graph_relax_vertex_pull_f32",
  "abi-r1:hip:navatala_graph_relax_vertex_pull_f32",
  "abi-r1:hip:navatala_graph_relax_vertex_pull_f32",
  7,
  kAbiArgs_hip_navatala_graph_relax_vertex_pull_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_label_propagate_edges_u32[] = {
  { "srcs", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dsts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "labels", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numEdges", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labels", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_label_propagate_edges_u32 = {
  1,
  "navatala_graph_label_propagate_edges_u32",
  "hip",
  "navatala_graph_label_propagate_edges_u32",
  "kernel:hip:navatala_graph_label_propagate_edges_u32",
  "abi-r1:hip:navatala_graph_label_propagate_edges_u32",
  "abi-r1:hip:navatala_graph_label_propagate_edges_u32",
  5,
  kAbiArgs_hip_navatala_graph_label_propagate_edges_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_symmetrize_reverse_u32[] = {
  { "srcs", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dsts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numEdges", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outSrcs", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outDsts", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_symmetrize_reverse_u32 = {
  1,
  "navatala_graph_symmetrize_reverse_u32",
  "hip",
  "navatala_graph_symmetrize_reverse_u32",
  "kernel:hip:navatala_graph_symmetrize_reverse_u32",
  "abi-r1:hip:navatala_graph_symmetrize_reverse_u32",
  "abi-r1:hip:navatala_graph_symmetrize_reverse_u32",
  5,
  kAbiArgs_hip_navatala_graph_symmetrize_reverse_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_apply_renumber_u32[] = {
  { "mapping", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "newIds", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_apply_renumber_u32 = {
  1,
  "navatala_graph_apply_renumber_u32",
  "hip",
  "navatala_graph_apply_renumber_u32",
  "kernel:hip:navatala_graph_apply_renumber_u32",
  "abi-r1:hip:navatala_graph_apply_renumber_u32",
  "abi-r1:hip:navatala_graph_apply_renumber_u32",
  3,
  kAbiArgs_hip_navatala_graph_apply_renumber_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_peel_below_threshold_u32[] = {
  { "active", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "threshold", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numVertices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "active", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_peel_below_threshold_u32 = {
  1,
  "navatala_graph_peel_below_threshold_u32",
  "hip",
  "navatala_graph_peel_below_threshold_u32",
  "kernel:hip:navatala_graph_peel_below_threshold_u32",
  "abi-r1:hip:navatala_graph_peel_below_threshold_u32",
  "abi-r1:hip:navatala_graph_peel_below_threshold_u32",
  5,
  kAbiArgs_hip_navatala_graph_peel_below_threshold_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_pair_intersection_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numPairs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "inter", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_pair_intersection_u32 = {
  1,
  "navatala_graph_pair_intersection_u32",
  "hip",
  "navatala_graph_pair_intersection_u32",
  "kernel:hip:navatala_graph_pair_intersection_u32",
  "abi-r1:hip:navatala_graph_pair_intersection_u32",
  "abi-r1:hip:navatala_graph_pair_intersection_u32",
  6,
  kAbiArgs_hip_navatala_graph_pair_intersection_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_jaccard_f32[] = {
  { "inter", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numPairs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coeff", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_jaccard_f32 = {
  1,
  "navatala_graph_jaccard_f32",
  "hip",
  "navatala_graph_jaccard_f32",
  "kernel:hip:navatala_graph_jaccard_f32",
  "abi-r1:hip:navatala_graph_jaccard_f32",
  "abi-r1:hip:navatala_graph_jaccard_f32",
  6,
  kAbiArgs_hip_navatala_graph_jaccard_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_overlap_f32[] = {
  { "inter", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numPairs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coeff", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_overlap_f32 = {
  1,
  "navatala_graph_overlap_f32",
  "hip",
  "navatala_graph_overlap_f32",
  "kernel:hip:navatala_graph_overlap_f32",
  "abi-r1:hip:navatala_graph_overlap_f32",
  "abi-r1:hip:navatala_graph_overlap_f32",
  6,
  kAbiArgs_hip_navatala_graph_overlap_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_sorensen_f32[] = {
  { "inter", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numPairs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coeff", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_sorensen_f32 = {
  1,
  "navatala_graph_sorensen_f32",
  "hip",
  "navatala_graph_sorensen_f32",
  "kernel:hip:navatala_graph_sorensen_f32",
  "abi-r1:hip:navatala_graph_sorensen_f32",
  "abi-r1:hip:navatala_graph_sorensen_f32",
  6,
  kAbiArgs_hip_navatala_graph_sorensen_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_cosine_f32[] = {
  { "inter", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numPairs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coeff", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_cosine_f32 = {
  1,
  "navatala_graph_cosine_f32",
  "hip",
  "navatala_graph_cosine_f32",
  "kernel:hip:navatala_graph_cosine_f32",
  "abi-r1:hip:navatala_graph_cosine_f32",
  "abi-r1:hip:navatala_graph_cosine_f32",
  6,
  kAbiArgs_hip_navatala_graph_cosine_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_uniform_random_walk_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "startVertices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "maxLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numWalks", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "walks", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_uniform_random_walk_u32 = {
  1,
  "navatala_graph_uniform_random_walk_u32",
  "hip",
  "navatala_graph_uniform_random_walk_u32",
  "kernel:hip:navatala_graph_uniform_random_walk_u32",
  "abi-r1:hip:navatala_graph_uniform_random_walk_u32",
  "abi-r1:hip:navatala_graph_uniform_random_walk_u32",
  7,
  kAbiArgs_hip_navatala_graph_uniform_random_walk_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_out_degree_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_out_degree_u32 = {
  1,
  "navatala_graph_out_degree_u32",
  "hip",
  "navatala_graph_out_degree_u32",
  "kernel:hip:navatala_graph_out_degree_u32",
  "abi-r1:hip:navatala_graph_out_degree_u32",
  "abi-r1:hip:navatala_graph_out_degree_u32",
  3,
  kAbiArgs_hip_navatala_graph_out_degree_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_in_degree_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_in_degree_u32 = {
  1,
  "navatala_graph_in_degree_u32",
  "hip",
  "navatala_graph_in_degree_u32",
  "kernel:hip:navatala_graph_in_degree_u32",
  "abi-r1:hip:navatala_graph_in_degree_u32",
  "abi-r1:hip:navatala_graph_in_degree_u32",
  3,
  kAbiArgs_hip_navatala_graph_in_degree_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_rbf_affinity_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sigma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "W", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_rbf_affinity_f32 = {
  1,
  "navatala_graph_rbf_affinity_f32",
  "hip",
  "navatala_graph_rbf_affinity_f32",
  "kernel:hip:navatala_graph_rbf_affinity_f32",
  "abi-r1:hip:navatala_graph_rbf_affinity_f32",
  "abi-r1:hip:navatala_graph_rbf_affinity_f32",
  5,
  kAbiArgs_hip_navatala_graph_rbf_affinity_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_rbf_affinity_f64[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sigma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "W", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_rbf_affinity_f64 = {
  1,
  "navatala_graph_rbf_affinity_f64",
  "hip",
  "navatala_graph_rbf_affinity_f64",
  "kernel:hip:navatala_graph_rbf_affinity_f64",
  "abi-r1:hip:navatala_graph_rbf_affinity_f64",
  "abi-r1:hip:navatala_graph_rbf_affinity_f64",
  5,
  kAbiArgs_hip_navatala_graph_rbf_affinity_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_compute_degree_matrix_f32[] = {
  { "W", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "D", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_compute_degree_matrix_f32 = {
  1,
  "navatala_graph_compute_degree_matrix_f32",
  "hip",
  "navatala_graph_compute_degree_matrix_f32",
  "kernel:hip:navatala_graph_compute_degree_matrix_f32",
  "abi-r1:hip:navatala_graph_compute_degree_matrix_f32",
  "abi-r1:hip:navatala_graph_compute_degree_matrix_f32",
  3,
  kAbiArgs_hip_navatala_graph_compute_degree_matrix_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_compute_degree_matrix_f64[] = {
  { "W", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "D", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_compute_degree_matrix_f64 = {
  1,
  "navatala_graph_compute_degree_matrix_f64",
  "hip",
  "navatala_graph_compute_degree_matrix_f64",
  "kernel:hip:navatala_graph_compute_degree_matrix_f64",
  "abi-r1:hip:navatala_graph_compute_degree_matrix_f64",
  "abi-r1:hip:navatala_graph_compute_degree_matrix_f64",
  3,
  kAbiArgs_hip_navatala_graph_compute_degree_matrix_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_compute_laplacian_f32[] = {
  { "W", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "D", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "L", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_compute_laplacian_f32 = {
  1,
  "navatala_graph_compute_laplacian_f32",
  "hip",
  "navatala_graph_compute_laplacian_f32",
  "kernel:hip:navatala_graph_compute_laplacian_f32",
  "abi-r1:hip:navatala_graph_compute_laplacian_f32",
  "abi-r1:hip:navatala_graph_compute_laplacian_f32",
  4,
  kAbiArgs_hip_navatala_graph_compute_laplacian_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_compute_laplacian_f64[] = {
  { "W", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "D", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "L", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_compute_laplacian_f64 = {
  1,
  "navatala_graph_compute_laplacian_f64",
  "hip",
  "navatala_graph_compute_laplacian_f64",
  "kernel:hip:navatala_graph_compute_laplacian_f64",
  "abi-r1:hip:navatala_graph_compute_laplacian_f64",
  "abi-r1:hip:navatala_graph_compute_laplacian_f64",
  4,
  kAbiArgs_hip_navatala_graph_compute_laplacian_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_normalized_laplacian_f32[] = {
  { "W", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "D", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "L_sym", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_normalized_laplacian_f32 = {
  1,
  "navatala_graph_normalized_laplacian_f32",
  "hip",
  "navatala_graph_normalized_laplacian_f32",
  "kernel:hip:navatala_graph_normalized_laplacian_f32",
  "abi-r1:hip:navatala_graph_normalized_laplacian_f32",
  "abi-r1:hip:navatala_graph_normalized_laplacian_f32",
  4,
  kAbiArgs_hip_navatala_graph_normalized_laplacian_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_normalized_laplacian_f64[] = {
  { "W", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "D", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "L_sym", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_normalized_laplacian_f64 = {
  1,
  "navatala_graph_normalized_laplacian_f64",
  "hip",
  "navatala_graph_normalized_laplacian_f64",
  "kernel:hip:navatala_graph_normalized_laplacian_f64",
  "abi-r1:hip:navatala_graph_normalized_laplacian_f64",
  "abi-r1:hip:navatala_graph_normalized_laplacian_f64",
  4,
  kAbiArgs_hip_navatala_graph_normalized_laplacian_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_normalize_eigenvector_rows_f32[] = {
  { "V", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "V_norm", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_normalize_eigenvector_rows_f32 = {
  1,
  "navatala_graph_normalize_eigenvector_rows_f32",
  "hip",
  "navatala_graph_normalize_eigenvector_rows_f32",
  "kernel:hip:navatala_graph_normalize_eigenvector_rows_f32",
  "abi-r1:hip:navatala_graph_normalize_eigenvector_rows_f32",
  "abi-r1:hip:navatala_graph_normalize_eigenvector_rows_f32",
  4,
  kAbiArgs_hip_navatala_graph_normalize_eigenvector_rows_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_normalize_eigenvector_rows_f64[] = {
  { "V", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "V_norm", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_normalize_eigenvector_rows_f64 = {
  1,
  "navatala_graph_normalize_eigenvector_rows_f64",
  "hip",
  "navatala_graph_normalize_eigenvector_rows_f64",
  "kernel:hip:navatala_graph_normalize_eigenvector_rows_f64",
  "abi-r1:hip:navatala_graph_normalize_eigenvector_rows_f64",
  "abi-r1:hip:navatala_graph_normalize_eigenvector_rows_f64",
  4,
  kAbiArgs_hip_navatala_graph_normalize_eigenvector_rows_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_spectral_embedding_f32[] = {
  { "eigenvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "embedding", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_spectral_embedding_f32 = {
  1,
  "navatala_graph_spectral_embedding_f32",
  "hip",
  "navatala_graph_spectral_embedding_f32",
  "kernel:hip:navatala_graph_spectral_embedding_f32",
  "abi-r1:hip:navatala_graph_spectral_embedding_f32",
  "abi-r1:hip:navatala_graph_spectral_embedding_f32",
  4,
  kAbiArgs_hip_navatala_graph_spectral_embedding_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_spectral_embedding_f64[] = {
  { "eigenvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "embedding", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_spectral_embedding_f64 = {
  1,
  "navatala_graph_spectral_embedding_f64",
  "hip",
  "navatala_graph_spectral_embedding_f64",
  "kernel:hip:navatala_graph_spectral_embedding_f64",
  "abi-r1:hip:navatala_graph_spectral_embedding_f64",
  "abi-r1:hip:navatala_graph_spectral_embedding_f64",
  4,
  kAbiArgs_hip_navatala_graph_spectral_embedding_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_rmat_generator_f32[] = {
  { "random", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "randomWeights", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "a", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "b", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "scale", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numEdges", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcNodes", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dstNodes", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeCount", 12, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_rmat_generator_f32 = {
  1,
  "navatala_graph_rmat_generator_f32",
  "hip",
  "navatala_graph_rmat_generator_f32",
  "kernel:hip:navatala_graph_rmat_generator_f32",
  "abi-r1:hip:navatala_graph_rmat_generator_f32",
  "abi-r1:hip:navatala_graph_rmat_generator_f32",
  13,
  kAbiArgs_hip_navatala_graph_rmat_generator_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_rmat_generator_f64[] = {
  { "random", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "randomWeights", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "a", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "b", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "c", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "d", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "n", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "scale", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numEdges", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcNodes", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dstNodes", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeCount", 12, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_rmat_generator_f64 = {
  1,
  "navatala_graph_rmat_generator_f64",
  "hip",
  "navatala_graph_rmat_generator_f64",
  "kernel:hip:navatala_graph_rmat_generator_f64",
  "abi-r1:hip:navatala_graph_rmat_generator_f64",
  "abi-r1:hip:navatala_graph_rmat_generator_f64",
  13,
  kAbiArgs_hip_navatala_graph_rmat_generator_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_erdos_renyi_f32[] = {
  { "random", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "randomWeights", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "p", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcNodes", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dstNodes", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeCount", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_erdos_renyi_f32 = {
  1,
  "navatala_graph_erdos_renyi_f32",
  "hip",
  "navatala_graph_erdos_renyi_f32",
  "kernel:hip:navatala_graph_erdos_renyi_f32",
  "abi-r1:hip:navatala_graph_erdos_renyi_f32",
  "abi-r1:hip:navatala_graph_erdos_renyi_f32",
  8,
  kAbiArgs_hip_navatala_graph_erdos_renyi_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_erdos_renyi_f64[] = {
  { "random", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "randomWeights", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "p", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcNodes", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dstNodes", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeCount", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_erdos_renyi_f64 = {
  1,
  "navatala_graph_erdos_renyi_f64",
  "hip",
  "navatala_graph_erdos_renyi_f64",
  "kernel:hip:navatala_graph_erdos_renyi_f64",
  "abi-r1:hip:navatala_graph_erdos_renyi_f64",
  "abi-r1:hip:navatala_graph_erdos_renyi_f64",
  8,
  kAbiArgs_hip_navatala_graph_erdos_renyi_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_barabasi_albert_f32[] = {
  { "cumDegree", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "random", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "randomWeights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "newNode", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalDegree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numExistingNodes", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcNodes", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dstNodes", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeCount", 10, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_barabasi_albert_f32 = {
  1,
  "navatala_graph_barabasi_albert_f32",
  "hip",
  "navatala_graph_barabasi_albert_f32",
  "kernel:hip:navatala_graph_barabasi_albert_f32",
  "abi-r1:hip:navatala_graph_barabasi_albert_f32",
  "abi-r1:hip:navatala_graph_barabasi_albert_f32",
  11,
  kAbiArgs_hip_navatala_graph_barabasi_albert_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_barabasi_albert_f64[] = {
  { "cumDegree", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "random", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "randomWeights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "newNode", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalDegree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numExistingNodes", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcNodes", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dstNodes", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeCount", 10, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_barabasi_albert_f64 = {
  1,
  "navatala_graph_barabasi_albert_f64",
  "hip",
  "navatala_graph_barabasi_albert_f64",
  "kernel:hip:navatala_graph_barabasi_albert_f64",
  "abi-r1:hip:navatala_graph_barabasi_albert_f64",
  "abi-r1:hip:navatala_graph_barabasi_albert_f64",
  11,
  kAbiArgs_hip_navatala_graph_barabasi_albert_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_compute_modularity_f32[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusters", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClusters", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalEdgeWeight", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "modularity", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_compute_modularity_f32 = {
  1,
  "navatala_graph_compute_modularity_f32",
  "hip",
  "navatala_graph_compute_modularity_f32",
  "kernel:hip:navatala_graph_compute_modularity_f32",
  "abi-r1:hip:navatala_graph_compute_modularity_f32",
  "abi-r1:hip:navatala_graph_compute_modularity_f32",
  8,
  kAbiArgs_hip_navatala_graph_compute_modularity_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_compute_modularity_f64[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusters", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClusters", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalEdgeWeight", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "modularity", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_compute_modularity_f64 = {
  1,
  "navatala_graph_compute_modularity_f64",
  "hip",
  "navatala_graph_compute_modularity_f64",
  "kernel:hip:navatala_graph_compute_modularity_f64",
  "abi-r1:hip:navatala_graph_compute_modularity_f64",
  "abi-r1:hip:navatala_graph_compute_modularity_f64",
  8,
  kAbiArgs_hip_navatala_graph_compute_modularity_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_modularity_matrix_vec_f32[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "totalEdgeWeight", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dotDX", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_modularity_matrix_vec_f32 = {
  1,
  "navatala_graph_modularity_matrix_vec_f32",
  "hip",
  "navatala_graph_modularity_matrix_vec_f32",
  "kernel:hip:navatala_graph_modularity_matrix_vec_f32",
  "abi-r1:hip:navatala_graph_modularity_matrix_vec_f32",
  "abi-r1:hip:navatala_graph_modularity_matrix_vec_f32",
  9,
  kAbiArgs_hip_navatala_graph_modularity_matrix_vec_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_modularity_matrix_vec_f64[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "totalEdgeWeight", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "n", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dotDX", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "y", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_modularity_matrix_vec_f64 = {
  1,
  "navatala_graph_modularity_matrix_vec_f64",
  "hip",
  "navatala_graph_modularity_matrix_vec_f64",
  "kernel:hip:navatala_graph_modularity_matrix_vec_f64",
  "abi-r1:hip:navatala_graph_modularity_matrix_vec_f64",
  "abi-r1:hip:navatala_graph_modularity_matrix_vec_f64",
  9,
  kAbiArgs_hip_navatala_graph_modularity_matrix_vec_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_construct_indicator_f32[] = {
  { "clusters", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusterIndex", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indicator", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_construct_indicator_f32 = {
  1,
  "navatala_graph_construct_indicator_f32",
  "hip",
  "navatala_graph_construct_indicator_f32",
  "kernel:hip:navatala_graph_construct_indicator_f32",
  "abi-r1:hip:navatala_graph_construct_indicator_f32",
  "abi-r1:hip:navatala_graph_construct_indicator_f32",
  4,
  kAbiArgs_hip_navatala_graph_construct_indicator_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_construct_indicator_f64[] = {
  { "clusters", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusterIndex", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indicator", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_construct_indicator_f64 = {
  1,
  "navatala_graph_construct_indicator_f64",
  "hip",
  "navatala_graph_construct_indicator_f64",
  "kernel:hip:navatala_graph_construct_indicator_f64",
  "abi-r1:hip:navatala_graph_construct_indicator_f64",
  "abi-r1:hip:navatala_graph_construct_indicator_f64",
  4,
  kAbiArgs_hip_navatala_graph_construct_indicator_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_compute_edge_cut_f32[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusters", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "edgeCut", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_compute_edge_cut_f32 = {
  1,
  "navatala_graph_compute_edge_cut_f32",
  "hip",
  "navatala_graph_compute_edge_cut_f32",
  "kernel:hip:navatala_graph_compute_edge_cut_f32",
  "abi-r1:hip:navatala_graph_compute_edge_cut_f32",
  "abi-r1:hip:navatala_graph_compute_edge_cut_f32",
  6,
  kAbiArgs_hip_navatala_graph_compute_edge_cut_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_compute_edge_cut_f64[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusters", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "edgeCut", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_compute_edge_cut_f64 = {
  1,
  "navatala_graph_compute_edge_cut_f64",
  "hip",
  "navatala_graph_compute_edge_cut_f64",
  "kernel:hip:navatala_graph_compute_edge_cut_f64",
  "abi-r1:hip:navatala_graph_compute_edge_cut_f64",
  "abi-r1:hip:navatala_graph_compute_edge_cut_f64",
  6,
  kAbiArgs_hip_navatala_graph_compute_edge_cut_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_compute_partition_cost_f32[] = {
  { "edgeCutPerCluster", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusterSizes", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numClusters", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cost", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_compute_partition_cost_f32 = {
  1,
  "navatala_graph_compute_partition_cost_f32",
  "hip",
  "navatala_graph_compute_partition_cost_f32",
  "kernel:hip:navatala_graph_compute_partition_cost_f32",
  "abi-r1:hip:navatala_graph_compute_partition_cost_f32",
  "abi-r1:hip:navatala_graph_compute_partition_cost_f32",
  4,
  kAbiArgs_hip_navatala_graph_compute_partition_cost_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_compute_partition_cost_f64[] = {
  { "edgeCutPerCluster", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusterSizes", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numClusters", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cost", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_compute_partition_cost_f64 = {
  1,
  "navatala_graph_compute_partition_cost_f64",
  "hip",
  "navatala_graph_compute_partition_cost_f64",
  "kernel:hip:navatala_graph_compute_partition_cost_f64",
  "abi-r1:hip:navatala_graph_compute_partition_cost_f64",
  "abi-r1:hip:navatala_graph_compute_partition_cost_f64",
  4,
  kAbiArgs_hip_navatala_graph_compute_partition_cost_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_analyze_partition_f32[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusters", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClusters", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "edgeCut", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cost", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_analyze_partition_f32 = {
  1,
  "navatala_graph_analyze_partition_f32",
  "hip",
  "navatala_graph_analyze_partition_f32",
  "kernel:hip:navatala_graph_analyze_partition_f32",
  "abi-r1:hip:navatala_graph_analyze_partition_f32",
  "abi-r1:hip:navatala_graph_analyze_partition_f32",
  8,
  kAbiArgs_hip_navatala_graph_analyze_partition_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_analyze_partition_f64[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusters", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClusters", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "edgeCut", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "cost", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_analyze_partition_f64 = {
  1,
  "navatala_graph_analyze_partition_f64",
  "hip",
  "navatala_graph_analyze_partition_f64",
  "kernel:hip:navatala_graph_analyze_partition_f64",
  "abi-r1:hip:navatala_graph_analyze_partition_f64",
  "abi-r1:hip:navatala_graph_analyze_partition_f64",
  8,
  kAbiArgs_hip_navatala_graph_analyze_partition_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_transform_eigen_matrix_f32[] = {
  { "eigenvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "transformed", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_transform_eigen_matrix_f32 = {
  1,
  "navatala_graph_transform_eigen_matrix_f32",
  "hip",
  "navatala_graph_transform_eigen_matrix_f32",
  "kernel:hip:navatala_graph_transform_eigen_matrix_f32",
  "abi-r1:hip:navatala_graph_transform_eigen_matrix_f32",
  "abi-r1:hip:navatala_graph_transform_eigen_matrix_f32",
  4,
  kAbiArgs_hip_navatala_graph_transform_eigen_matrix_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_transform_eigen_matrix_f64[] = {
  { "eigenvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "transformed", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_transform_eigen_matrix_f64 = {
  1,
  "navatala_graph_transform_eigen_matrix_f64",
  "hip",
  "navatala_graph_transform_eigen_matrix_f64",
  "kernel:hip:navatala_graph_transform_eigen_matrix_f64",
  "abi-r1:hip:navatala_graph_transform_eigen_matrix_f64",
  "abi-r1:hip:navatala_graph_transform_eigen_matrix_f64",
  4,
  kAbiArgs_hip_navatala_graph_transform_eigen_matrix_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_compute_cluster_sizes_f32[] = {
  { "clusters", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClusters", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sizes", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_compute_cluster_sizes_f32 = {
  1,
  "navatala_graph_compute_cluster_sizes_f32",
  "hip",
  "navatala_graph_compute_cluster_sizes_f32",
  "kernel:hip:navatala_graph_compute_cluster_sizes_f32",
  "abi-r1:hip:navatala_graph_compute_cluster_sizes_f32",
  "abi-r1:hip:navatala_graph_compute_cluster_sizes_f32",
  4,
  kAbiArgs_hip_navatala_graph_compute_cluster_sizes_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_compute_cluster_sizes_f64[] = {
  { "clusters", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClusters", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sizes", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_compute_cluster_sizes_f64 = {
  1,
  "navatala_graph_compute_cluster_sizes_f64",
  "hip",
  "navatala_graph_compute_cluster_sizes_f64",
  "kernel:hip:navatala_graph_compute_cluster_sizes_f64",
  "abi-r1:hip:navatala_graph_compute_cluster_sizes_f64",
  "abi-r1:hip:navatala_graph_compute_cluster_sizes_f64",
  4,
  kAbiArgs_hip_navatala_graph_compute_cluster_sizes_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_laplacian_from_adjacency_f32[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "laplacianValues", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degree", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_laplacian_from_adjacency_f32 = {
  1,
  "navatala_graph_laplacian_from_adjacency_f32",
  "hip",
  "navatala_graph_laplacian_from_adjacency_f32",
  "kernel:hip:navatala_graph_laplacian_from_adjacency_f32",
  "abi-r1:hip:navatala_graph_laplacian_from_adjacency_f32",
  "abi-r1:hip:navatala_graph_laplacian_from_adjacency_f32",
  6,
  kAbiArgs_hip_navatala_graph_laplacian_from_adjacency_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_graph_laplacian_from_adjacency_f64[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "laplacianValues", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degree", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_graph_laplacian_from_adjacency_f64 = {
  1,
  "navatala_graph_laplacian_from_adjacency_f64",
  "hip",
  "navatala_graph_laplacian_from_adjacency_f64",
  "kernel:hip:navatala_graph_laplacian_from_adjacency_f64",
  "abi-r1:hip:navatala_graph_laplacian_from_adjacency_f64",
  "abi-r1:hip:navatala_graph_laplacian_from_adjacency_f64",
  6,
  kAbiArgs_hip_navatala_graph_laplacian_from_adjacency_f64
};

bool tryGetKernelAbiManifest_hip_graph(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "hip" && kernelName == "navatala_graph_pack_f32") {
    out = &kAbiManifest_hip_navatala_graph_pack_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_f64") {
    out = &kAbiManifest_hip_navatala_graph_pack_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_i32") {
    out = &kAbiManifest_hip_navatala_graph_pack_i32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_u32") {
    out = &kAbiManifest_hip_navatala_graph_pack_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_f32") {
    out = &kAbiManifest_hip_navatala_graph_unpack_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_f64") {
    out = &kAbiManifest_hip_navatala_graph_unpack_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_i32") {
    out = &kAbiManifest_hip_navatala_graph_unpack_i32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_u32") {
    out = &kAbiManifest_hip_navatala_graph_unpack_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_f32x3") {
    out = &kAbiManifest_hip_navatala_graph_pack_f32x3;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_f32x3") {
    out = &kAbiManifest_hip_navatala_graph_unpack_f32x3;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_f32x6") {
    out = &kAbiManifest_hip_navatala_graph_pack_f32x6;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_f32x6") {
    out = &kAbiManifest_hip_navatala_graph_unpack_f32x6;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_f32x9") {
    out = &kAbiManifest_hip_navatala_graph_pack_f32x9;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_f32x9") {
    out = &kAbiManifest_hip_navatala_graph_unpack_f32x9;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_f64x3") {
    out = &kAbiManifest_hip_navatala_graph_pack_f64x3;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_f64x3") {
    out = &kAbiManifest_hip_navatala_graph_unpack_f64x3;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_f64x6") {
    out = &kAbiManifest_hip_navatala_graph_pack_f64x6;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_f64x6") {
    out = &kAbiManifest_hip_navatala_graph_unpack_f64x6;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_f64x9") {
    out = &kAbiManifest_hip_navatala_graph_pack_f64x9;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_f64x9") {
    out = &kAbiManifest_hip_navatala_graph_unpack_f64x9;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_degree_out_u32") {
    out = &kAbiManifest_hip_navatala_graph_degree_out_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_degree_in_u32") {
    out = &kAbiManifest_hip_navatala_graph_degree_in_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_spmv_weighted_f32") {
    out = &kAbiManifest_hip_navatala_graph_spmv_weighted_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_spmv_weighted_subgroup_f32") {
    out = &kAbiManifest_hip_navatala_graph_spmv_weighted_subgroup_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_spmv_unweighted_f32") {
    out = &kAbiManifest_hip_navatala_graph_spmv_unweighted_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_spmv_unweighted_subgroup_f32") {
    out = &kAbiManifest_hip_navatala_graph_spmv_unweighted_subgroup_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_axpy2_f32") {
    out = &kAbiManifest_hip_navatala_graph_axpy2_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_scale_f32") {
    out = &kAbiManifest_hip_navatala_graph_scale_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_scale_add_bias_f32") {
    out = &kAbiManifest_hip_navatala_graph_scale_add_bias_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_l1norm_f32") {
    out = &kAbiManifest_hip_navatala_graph_l1norm_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_recip_f32") {
    out = &kAbiManifest_hip_navatala_graph_recip_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_l1_diff_f32") {
    out = &kAbiManifest_hip_navatala_graph_l1_diff_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_count_diff_u32") {
    out = &kAbiManifest_hip_navatala_graph_count_diff_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_copy_f32") {
    out = &kAbiManifest_hip_navatala_graph_copy_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_copy_u32") {
    out = &kAbiManifest_hip_navatala_graph_copy_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_relax_vertex_pull_f32") {
    out = &kAbiManifest_hip_navatala_graph_relax_vertex_pull_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_label_propagate_edges_u32") {
    out = &kAbiManifest_hip_navatala_graph_label_propagate_edges_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_symmetrize_reverse_u32") {
    out = &kAbiManifest_hip_navatala_graph_symmetrize_reverse_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_apply_renumber_u32") {
    out = &kAbiManifest_hip_navatala_graph_apply_renumber_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_peel_below_threshold_u32") {
    out = &kAbiManifest_hip_navatala_graph_peel_below_threshold_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pair_intersection_u32") {
    out = &kAbiManifest_hip_navatala_graph_pair_intersection_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_jaccard_f32") {
    out = &kAbiManifest_hip_navatala_graph_jaccard_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_overlap_f32") {
    out = &kAbiManifest_hip_navatala_graph_overlap_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_sorensen_f32") {
    out = &kAbiManifest_hip_navatala_graph_sorensen_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_cosine_f32") {
    out = &kAbiManifest_hip_navatala_graph_cosine_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_uniform_random_walk_u32") {
    out = &kAbiManifest_hip_navatala_graph_uniform_random_walk_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_out_degree_u32") {
    out = &kAbiManifest_hip_navatala_graph_out_degree_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_in_degree_u32") {
    out = &kAbiManifest_hip_navatala_graph_in_degree_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_rbf_affinity_f32") {
    out = &kAbiManifest_hip_navatala_graph_rbf_affinity_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_rbf_affinity_f64") {
    out = &kAbiManifest_hip_navatala_graph_rbf_affinity_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_degree_matrix_f32") {
    out = &kAbiManifest_hip_navatala_graph_compute_degree_matrix_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_degree_matrix_f64") {
    out = &kAbiManifest_hip_navatala_graph_compute_degree_matrix_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_laplacian_f32") {
    out = &kAbiManifest_hip_navatala_graph_compute_laplacian_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_laplacian_f64") {
    out = &kAbiManifest_hip_navatala_graph_compute_laplacian_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_normalized_laplacian_f32") {
    out = &kAbiManifest_hip_navatala_graph_normalized_laplacian_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_normalized_laplacian_f64") {
    out = &kAbiManifest_hip_navatala_graph_normalized_laplacian_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_normalize_eigenvector_rows_f32") {
    out = &kAbiManifest_hip_navatala_graph_normalize_eigenvector_rows_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_normalize_eigenvector_rows_f64") {
    out = &kAbiManifest_hip_navatala_graph_normalize_eigenvector_rows_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_spectral_embedding_f32") {
    out = &kAbiManifest_hip_navatala_graph_spectral_embedding_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_spectral_embedding_f64") {
    out = &kAbiManifest_hip_navatala_graph_spectral_embedding_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_rmat_generator_f32") {
    out = &kAbiManifest_hip_navatala_graph_rmat_generator_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_rmat_generator_f64") {
    out = &kAbiManifest_hip_navatala_graph_rmat_generator_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_erdos_renyi_f32") {
    out = &kAbiManifest_hip_navatala_graph_erdos_renyi_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_erdos_renyi_f64") {
    out = &kAbiManifest_hip_navatala_graph_erdos_renyi_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_barabasi_albert_f32") {
    out = &kAbiManifest_hip_navatala_graph_barabasi_albert_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_barabasi_albert_f64") {
    out = &kAbiManifest_hip_navatala_graph_barabasi_albert_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_modularity_f32") {
    out = &kAbiManifest_hip_navatala_graph_compute_modularity_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_modularity_f64") {
    out = &kAbiManifest_hip_navatala_graph_compute_modularity_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_modularity_matrix_vec_f32") {
    out = &kAbiManifest_hip_navatala_graph_modularity_matrix_vec_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_modularity_matrix_vec_f64") {
    out = &kAbiManifest_hip_navatala_graph_modularity_matrix_vec_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_construct_indicator_f32") {
    out = &kAbiManifest_hip_navatala_graph_construct_indicator_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_construct_indicator_f64") {
    out = &kAbiManifest_hip_navatala_graph_construct_indicator_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_edge_cut_f32") {
    out = &kAbiManifest_hip_navatala_graph_compute_edge_cut_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_edge_cut_f64") {
    out = &kAbiManifest_hip_navatala_graph_compute_edge_cut_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_partition_cost_f32") {
    out = &kAbiManifest_hip_navatala_graph_compute_partition_cost_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_partition_cost_f64") {
    out = &kAbiManifest_hip_navatala_graph_compute_partition_cost_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_analyze_partition_f32") {
    out = &kAbiManifest_hip_navatala_graph_analyze_partition_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_analyze_partition_f64") {
    out = &kAbiManifest_hip_navatala_graph_analyze_partition_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_transform_eigen_matrix_f32") {
    out = &kAbiManifest_hip_navatala_graph_transform_eigen_matrix_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_transform_eigen_matrix_f64") {
    out = &kAbiManifest_hip_navatala_graph_transform_eigen_matrix_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_cluster_sizes_f32") {
    out = &kAbiManifest_hip_navatala_graph_compute_cluster_sizes_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_cluster_sizes_f64") {
    out = &kAbiManifest_hip_navatala_graph_compute_cluster_sizes_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_laplacian_from_adjacency_f32") {
    out = &kAbiManifest_hip_navatala_graph_laplacian_from_adjacency_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_laplacian_from_adjacency_f64") {
    out = &kAbiManifest_hip_navatala_graph_laplacian_from_adjacency_f64;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_hip_graph(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "hip" && kernelName == "navatala_graph_pack_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_pack_f32";
    std::string_view sv(k_hip_navatala_graph_pack_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_pack_f64";
    std::string_view sv(k_hip_navatala_graph_pack_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_pack_i32";
    std::string_view sv(k_hip_navatala_graph_pack_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_pack_u32";
    std::string_view sv(k_hip_navatala_graph_pack_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_unpack_f32";
    std::string_view sv(k_hip_navatala_graph_unpack_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_unpack_f64";
    std::string_view sv(k_hip_navatala_graph_unpack_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_unpack_i32";
    std::string_view sv(k_hip_navatala_graph_unpack_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_unpack_u32";
    std::string_view sv(k_hip_navatala_graph_unpack_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_f32x3") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_pack_f32x3";
    std::string_view sv(k_hip_navatala_graph_pack_f32x3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_f32x3") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_unpack_f32x3";
    std::string_view sv(k_hip_navatala_graph_unpack_f32x3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_f32x6") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_pack_f32x6";
    std::string_view sv(k_hip_navatala_graph_pack_f32x6);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_f32x6") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_unpack_f32x6";
    std::string_view sv(k_hip_navatala_graph_unpack_f32x6);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_f32x9") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_pack_f32x9";
    std::string_view sv(k_hip_navatala_graph_pack_f32x9);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_f32x9") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_unpack_f32x9";
    std::string_view sv(k_hip_navatala_graph_unpack_f32x9);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_f64x3") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_pack_f64x3";
    std::string_view sv(k_hip_navatala_graph_pack_f64x3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_f64x3") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_unpack_f64x3";
    std::string_view sv(k_hip_navatala_graph_unpack_f64x3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_f64x6") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_pack_f64x6";
    std::string_view sv(k_hip_navatala_graph_pack_f64x6);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_f64x6") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_unpack_f64x6";
    std::string_view sv(k_hip_navatala_graph_unpack_f64x6);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pack_f64x9") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_pack_f64x9";
    std::string_view sv(k_hip_navatala_graph_pack_f64x9);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_unpack_f64x9") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_unpack_f64x9";
    std::string_view sv(k_hip_navatala_graph_unpack_f64x9);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_degree_out_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_degree_out_u32";
    std::string_view sv(k_hip_navatala_graph_degree_out_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_degree_in_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_degree_in_u32";
    std::string_view sv(k_hip_navatala_graph_degree_in_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_spmv_weighted_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_spmv_weighted_f32";
    std::string_view sv(k_hip_navatala_graph_spmv_weighted_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_spmv_weighted_subgroup_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_spmv_weighted_subgroup_f32";
    std::string_view sv(k_hip_navatala_graph_spmv_weighted_subgroup_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_spmv_unweighted_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_spmv_unweighted_f32";
    std::string_view sv(k_hip_navatala_graph_spmv_unweighted_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_spmv_unweighted_subgroup_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_spmv_unweighted_subgroup_f32";
    std::string_view sv(k_hip_navatala_graph_spmv_unweighted_subgroup_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_axpy2_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_axpy2_f32";
    std::string_view sv(k_hip_navatala_graph_axpy2_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_scale_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_scale_f32";
    std::string_view sv(k_hip_navatala_graph_scale_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_scale_add_bias_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_scale_add_bias_f32";
    std::string_view sv(k_hip_navatala_graph_scale_add_bias_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_l1norm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_l1norm_f32";
    std::string_view sv(k_hip_navatala_graph_l1norm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_recip_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_recip_f32";
    std::string_view sv(k_hip_navatala_graph_recip_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_l1_diff_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_l1_diff_f32";
    std::string_view sv(k_hip_navatala_graph_l1_diff_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_count_diff_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_count_diff_u32";
    std::string_view sv(k_hip_navatala_graph_count_diff_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_copy_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_copy_f32";
    std::string_view sv(k_hip_navatala_graph_copy_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_copy_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_copy_u32";
    std::string_view sv(k_hip_navatala_graph_copy_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_relax_vertex_pull_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_relax_vertex_pull_f32";
    std::string_view sv(k_hip_navatala_graph_relax_vertex_pull_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_label_propagate_edges_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_label_propagate_edges_u32";
    std::string_view sv(k_hip_navatala_graph_label_propagate_edges_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_symmetrize_reverse_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_symmetrize_reverse_u32";
    std::string_view sv(k_hip_navatala_graph_symmetrize_reverse_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_apply_renumber_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_apply_renumber_u32";
    std::string_view sv(k_hip_navatala_graph_apply_renumber_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_peel_below_threshold_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_peel_below_threshold_u32";
    std::string_view sv(k_hip_navatala_graph_peel_below_threshold_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_pair_intersection_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_pair_intersection_u32";
    std::string_view sv(k_hip_navatala_graph_pair_intersection_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_jaccard_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_jaccard_f32";
    std::string_view sv(k_hip_navatala_graph_jaccard_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_overlap_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_overlap_f32";
    std::string_view sv(k_hip_navatala_graph_overlap_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_sorensen_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_sorensen_f32";
    std::string_view sv(k_hip_navatala_graph_sorensen_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_cosine_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_cosine_f32";
    std::string_view sv(k_hip_navatala_graph_cosine_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_uniform_random_walk_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_uniform_random_walk_u32";
    std::string_view sv(k_hip_navatala_graph_uniform_random_walk_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_out_degree_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_out_degree_u32";
    std::string_view sv(k_hip_navatala_graph_out_degree_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_in_degree_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_in_degree_u32";
    std::string_view sv(k_hip_navatala_graph_in_degree_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_rbf_affinity_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_rbf_affinity_f32";
    std::string_view sv(k_hip_navatala_graph_rbf_affinity_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_rbf_affinity_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_rbf_affinity_f64";
    std::string_view sv(k_hip_navatala_graph_rbf_affinity_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_degree_matrix_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_compute_degree_matrix_f32";
    std::string_view sv(k_hip_navatala_graph_compute_degree_matrix_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_degree_matrix_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_compute_degree_matrix_f64";
    std::string_view sv(k_hip_navatala_graph_compute_degree_matrix_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_laplacian_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_compute_laplacian_f32";
    std::string_view sv(k_hip_navatala_graph_compute_laplacian_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_laplacian_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_compute_laplacian_f64";
    std::string_view sv(k_hip_navatala_graph_compute_laplacian_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_normalized_laplacian_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_normalized_laplacian_f32";
    std::string_view sv(k_hip_navatala_graph_normalized_laplacian_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_normalized_laplacian_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_normalized_laplacian_f64";
    std::string_view sv(k_hip_navatala_graph_normalized_laplacian_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_normalize_eigenvector_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_normalize_eigenvector_rows_f32";
    std::string_view sv(k_hip_navatala_graph_normalize_eigenvector_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_normalize_eigenvector_rows_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_normalize_eigenvector_rows_f64";
    std::string_view sv(k_hip_navatala_graph_normalize_eigenvector_rows_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_spectral_embedding_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_spectral_embedding_f32";
    std::string_view sv(k_hip_navatala_graph_spectral_embedding_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_spectral_embedding_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_spectral_embedding_f64";
    std::string_view sv(k_hip_navatala_graph_spectral_embedding_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_rmat_generator_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_rmat_generator_f32";
    std::string_view sv(k_hip_navatala_graph_rmat_generator_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_rmat_generator_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_rmat_generator_f64";
    std::string_view sv(k_hip_navatala_graph_rmat_generator_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_erdos_renyi_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_erdos_renyi_f32";
    std::string_view sv(k_hip_navatala_graph_erdos_renyi_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_erdos_renyi_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_erdos_renyi_f64";
    std::string_view sv(k_hip_navatala_graph_erdos_renyi_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_barabasi_albert_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_barabasi_albert_f32";
    std::string_view sv(k_hip_navatala_graph_barabasi_albert_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_barabasi_albert_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_barabasi_albert_f64";
    std::string_view sv(k_hip_navatala_graph_barabasi_albert_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_modularity_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_compute_modularity_f32";
    std::string_view sv(k_hip_navatala_graph_compute_modularity_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_modularity_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_compute_modularity_f64";
    std::string_view sv(k_hip_navatala_graph_compute_modularity_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_modularity_matrix_vec_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_modularity_matrix_vec_f32";
    std::string_view sv(k_hip_navatala_graph_modularity_matrix_vec_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_modularity_matrix_vec_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_modularity_matrix_vec_f64";
    std::string_view sv(k_hip_navatala_graph_modularity_matrix_vec_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_construct_indicator_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_construct_indicator_f32";
    std::string_view sv(k_hip_navatala_graph_construct_indicator_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_construct_indicator_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_construct_indicator_f64";
    std::string_view sv(k_hip_navatala_graph_construct_indicator_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_edge_cut_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_compute_edge_cut_f32";
    std::string_view sv(k_hip_navatala_graph_compute_edge_cut_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_edge_cut_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_compute_edge_cut_f64";
    std::string_view sv(k_hip_navatala_graph_compute_edge_cut_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_partition_cost_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_compute_partition_cost_f32";
    std::string_view sv(k_hip_navatala_graph_compute_partition_cost_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_partition_cost_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_compute_partition_cost_f64";
    std::string_view sv(k_hip_navatala_graph_compute_partition_cost_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_analyze_partition_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_analyze_partition_f32";
    std::string_view sv(k_hip_navatala_graph_analyze_partition_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_analyze_partition_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_analyze_partition_f64";
    std::string_view sv(k_hip_navatala_graph_analyze_partition_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_transform_eigen_matrix_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_transform_eigen_matrix_f32";
    std::string_view sv(k_hip_navatala_graph_transform_eigen_matrix_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_transform_eigen_matrix_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_transform_eigen_matrix_f64";
    std::string_view sv(k_hip_navatala_graph_transform_eigen_matrix_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_cluster_sizes_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_compute_cluster_sizes_f32";
    std::string_view sv(k_hip_navatala_graph_compute_cluster_sizes_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_compute_cluster_sizes_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_compute_cluster_sizes_f64";
    std::string_view sv(k_hip_navatala_graph_compute_cluster_sizes_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_laplacian_from_adjacency_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_laplacian_from_adjacency_f32";
    std::string_view sv(k_hip_navatala_graph_laplacian_from_adjacency_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_graph_laplacian_from_adjacency_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_graph_laplacian_from_adjacency_f64";
    std::string_view sv(k_hip_navatala_graph_laplacian_from_adjacency_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

