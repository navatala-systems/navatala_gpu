// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_graph` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_graph_pack_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_pack_f32(device const int* indices [[buffer(0)]], device const float* src [[buffer(1)]], device const int* count [[buffer(2)]], device float* dst [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  if (i < count[0]) {
    dst[i] = src[indices[((int)(int(__gid.x)))]];
  }
}

)kernel";
const char* k_metal_navatala_graph_pack_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_pack_i32(device const int* indices [[buffer(0)]], device const int* src [[buffer(1)]], device const int* count [[buffer(2)]], device int* dst [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  if (i < count[0]) {
    dst[i] = src[indices[((int)(int(__gid.x)))]];
  }
}

)kernel";
const char* k_metal_navatala_graph_pack_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_pack_u32(device const int* indices [[buffer(0)]], device const uint* src [[buffer(1)]], device const int* count [[buffer(2)]], device uint* dst [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  if (i < count[0]) {
    dst[i] = src[indices[((int)(int(__gid.x)))]];
  }
}

)kernel";
const char* k_metal_navatala_graph_unpack_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_unpack_f32(device const int* indices [[buffer(0)]], device const float* src [[buffer(1)]], device const int* count [[buffer(2)]], device float* dst [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  if (i < count[0]) {
    dst[indices[((int)(int(__gid.x)))]] = src[((int)(int(__gid.x)))];
  }
}

)kernel";
const char* k_metal_navatala_graph_unpack_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_unpack_i32(device const int* indices [[buffer(0)]], device const int* src [[buffer(1)]], device const int* count [[buffer(2)]], device int* dst [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  if (i < count[0]) {
    dst[indices[((int)(int(__gid.x)))]] = src[((int)(int(__gid.x)))];
  }
}

)kernel";
const char* k_metal_navatala_graph_unpack_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_unpack_u32(device const int* indices [[buffer(0)]], device const uint* src [[buffer(1)]], device const int* count [[buffer(2)]], device uint* dst [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  if (i < count[0]) {
    dst[indices[((int)(int(__gid.x)))]] = src[((int)(int(__gid.x)))];
  }
}

)kernel";
const char* k_metal_navatala_graph_pack_f32x3 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_pack_f32x3(device const int* indices [[buffer(0)]], device const float* src [[buffer(1)]], device const int* count [[buffer(2)]], device const int* n [[buffer(3)]], device float* dst [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(3); ++c) {
      int dstLin = ((c * count[0]) + i);
      int srcLin = ((c * n[0]) + indices[((int)(int(__gid.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_unpack_f32x3 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_unpack_f32x3(device const int* indices [[buffer(0)]], device const float* src [[buffer(1)]], device const int* count [[buffer(2)]], device const int* n [[buffer(3)]], device float* dst [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(3); ++c) {
      int srcLin = ((c * count[0]) + i);
      int dstLin = ((c * n[0]) + indices[((int)(int(__gid.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_pack_f32x6 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_pack_f32x6(device const int* indices [[buffer(0)]], device const float* src [[buffer(1)]], device const int* count [[buffer(2)]], device const int* n [[buffer(3)]], device float* dst [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(6); ++c) {
      int dstLin = ((c * count[0]) + i);
      int srcLin = ((c * n[0]) + indices[((int)(int(__gid.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_unpack_f32x6 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_unpack_f32x6(device const int* indices [[buffer(0)]], device const float* src [[buffer(1)]], device const int* count [[buffer(2)]], device const int* n [[buffer(3)]], device float* dst [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(6); ++c) {
      int srcLin = ((c * count[0]) + i);
      int dstLin = ((c * n[0]) + indices[((int)(int(__gid.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_pack_f32x9 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_pack_f32x9(device const int* indices [[buffer(0)]], device const float* src [[buffer(1)]], device const int* count [[buffer(2)]], device const int* n [[buffer(3)]], device float* dst [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(9); ++c) {
      int dstLin = ((c * count[0]) + i);
      int srcLin = ((c * n[0]) + indices[((int)(int(__gid.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_unpack_f32x9 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_unpack_f32x9(device const int* indices [[buffer(0)]], device const float* src [[buffer(1)]], device const int* count [[buffer(2)]], device const int* n [[buffer(3)]], device float* dst [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = ((int)(int(__gid.x)));
  if (i < count[0]) {
    for (int c = 0; c < (int)(9); ++c) {
      int srcLin = ((c * count[0]) + i);
      int dstLin = ((c * n[0]) + indices[((int)(int(__gid.x)))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_degree_out_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_degree_out_u32(device const uint* offsets [[buffer(0)]], device const uint* numVertices [[buffer(1)]], device uint* degrees [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numV = numVertices[0];
  if (gid < numV) {
    uint hi = offsets[(gid + 1u)];
    uint lo = offsets[gid];
    uint deg = (hi - lo);
    degrees[gid] = deg;
  }
}

)kernel";
const char* k_metal_navatala_graph_degree_in_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_degree_in_u32(device const uint* offsets [[buffer(0)]], device const uint* numVertices [[buffer(1)]], device uint* degrees [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numV = numVertices[0];
  if (gid < numV) {
    uint hi = offsets[(gid + 1u)];
    uint lo = offsets[gid];
    uint deg = (hi - lo);
    degrees[gid] = deg;
  }
}

)kernel";
const char* k_metal_navatala_graph_spmv_weighted_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_spmv_weighted_f32(device const uint* offsets [[buffer(0)]], device const uint* indices [[buffer(1)]], device const float* weights [[buffer(2)]], device const float* x [[buffer(3)]], device const uint* numVertices [[buffer(4)]], device float* y [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numV = numVertices[0];
  if (gid < numV) {
    uint base = offsets[gid];
    uint endv = offsets[(gid + 1u)];
    uint rowlen = (endv - base);
    float acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(rowlen); ++k) {
      uint eidx = (base + ((uint)(k)));
      uint col = indices[eidx];
      float w = weights[eidx];
      float xv = x[col];
      acc = (acc + (w * xv));
    }
    y[gid] = acc;
  }
}

)kernel";
const char* k_metal_navatala_graph_spmv_weighted_subgroup_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_spmv_weighted_subgroup_f32(device const uint* offsets [[buffer(0)]], device const uint* indices [[buffer(1)]], device const float* weights [[buffer(2)]], device const float* x [[buffer(3)]], device const uint* numVertices [[buffer(4)]], device float* y [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lane = ((uint)(int(__lane)));
  uint subgroupSize = ((uint)(int(__simd_size)));
  uint row = (gid / subgroupSize);
  uint numV = numVertices[0];
  if (row < numV) {
    uint base = offsets[row];
    uint endv = offsets[(row + 1u)];
    uint rowlen = (endv - base);
    uint rowIters = ((rowlen + (subgroupSize - 1u)) / subgroupSize);
    float laneAcc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(rowIters); ++k) {
      uint rel = ((((uint)(k)) * subgroupSize) + lane);
      if (rel < rowlen) {
        uint eidx = (base + rel);
        uint col = indices[eidx];
        float w = weights[eidx];
        float xv = x[col];
        laneAcc = (laneAcc + (w * xv));
      }
    }
    float rowSum = simd_sum(laneAcc);
    if (lane == 0u) {
      y[row] = rowSum;
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_spmv_unweighted_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_spmv_unweighted_f32(device const uint* offsets [[buffer(0)]], device const uint* indices [[buffer(1)]], device const float* x [[buffer(2)]], device const uint* numVertices [[buffer(3)]], device float* y [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numV = numVertices[0];
  if (gid < numV) {
    uint base = offsets[gid];
    uint endv = offsets[(gid + 1u)];
    uint rowlen = (endv - base);
    float acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(rowlen); ++k) {
      uint eidx = (base + ((uint)(k)));
      uint col = indices[eidx];
      float xv = x[col];
      acc = (acc + xv);
    }
    y[gid] = acc;
  }
}

)kernel";
const char* k_metal_navatala_graph_spmv_unweighted_subgroup_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_spmv_unweighted_subgroup_f32(device const uint* offsets [[buffer(0)]], device const uint* indices [[buffer(1)]], device const float* x [[buffer(2)]], device const uint* numVertices [[buffer(3)]], device float* y [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lane = ((uint)(int(__lane)));
  uint subgroupSize = ((uint)(int(__simd_size)));
  uint row = (gid / subgroupSize);
  uint numV = numVertices[0];
  if (row < numV) {
    uint base = offsets[row];
    uint endv = offsets[(row + 1u)];
    uint rowlen = (endv - base);
    uint rowIters = ((rowlen + (subgroupSize - 1u)) / subgroupSize);
    float laneAcc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(rowIters); ++k) {
      uint rel = ((((uint)(k)) * subgroupSize) + lane);
      if (rel < rowlen) {
        uint eidx = (base + rel);
        uint col = indices[eidx];
        float xv = x[col];
        laneAcc = (laneAcc + xv);
      }
    }
    float rowSum = simd_sum(laneAcc);
    if (lane == 0u) {
      y[row] = rowSum;
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_axpy2_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_axpy2_f32(device const float* x [[buffer(0)]], device const float* z [[buffer(1)]], device const float* alpha [[buffer(2)]], device const float* beta [[buffer(3)]], device const uint* count [[buffer(4)]], device float* y [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
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
const char* k_metal_navatala_graph_scale_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_scale_f32(device const float* x [[buffer(0)]], device const float* alpha [[buffer(1)]], device const uint* count [[buffer(2)]], device float* y [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float alphaV = alpha[0];
    float xv = x[gid];
    y[gid] = (alphaV * xv);
  }
}

)kernel";
const char* k_metal_navatala_graph_scale_add_bias_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_scale_add_bias_f32(device const float* x [[buffer(0)]], device const float* alpha [[buffer(1)]], device const float* bias [[buffer(2)]], device const uint* count [[buffer(3)]], device float* y [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float alphaV = alpha[0];
    float biasV = bias[0];
    float xv = x[gid];
    y[gid] = ((alphaV * xv) + biasV);
  }
}

)kernel";
const char* k_metal_navatala_graph_l1norm_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_l1norm_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if (idx < countVal) {
      float val = abs(_input[idx]);
      gsAcc = (gsAcc + val);
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    result[0] = sdata[0];
  }
}

)kernel";
const char* k_metal_navatala_graph_recip_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_recip_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  if (gid == 0u) {
    float xv = x[0];
    if (xv == as_type<float>(0x00000000u)) {
      y[0] = as_type<float>(0x00000000u);
    } else {
      y[0] = (as_type<float>(0x3f800000u) / xv);
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_l1_diff_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_l1_diff_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* result [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if (idx < countVal) {
      float av = a[idx];
      float bv = b[idx];
      float d = abs((av - bv));
      gsAcc = (gsAcc + d);
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    result[0] = sdata[0];
  }
}

)kernel";
const char* k_metal_navatala_graph_count_diff_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_count_diff_u32(device const uint* a [[buffer(0)]], device const uint* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* result [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if (idx < countVal) {
      uint av = a[idx];
      uint bv = b[idx];
      float diff = (((av == bv)) ? (as_type<float>(0x00000000u)) : (as_type<float>(0x3f800000u)));
      gsAcc = (gsAcc + diff);
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    result[0] = sdata[0];
  }
}

)kernel";
const char* k_metal_navatala_graph_copy_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_copy_f32(device const float* x [[buffer(0)]], device const uint* count [[buffer(1)]], device float* y [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float xv = x[gid];
    y[gid] = xv;
  }
}

)kernel";
const char* k_metal_navatala_graph_copy_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_copy_u32(device const uint* x [[buffer(0)]], device const uint* count [[buffer(1)]], device uint* y [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    uint xv = x[gid];
    y[gid] = xv;
  }
}

)kernel";
const char* k_metal_navatala_graph_relax_vertex_pull_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_relax_vertex_pull_f32(device const uint* inOffsets [[buffer(0)]], device const uint* inSrcs [[buffer(1)]], device const float* inWeights [[buffer(2)]], device float* dist [[buffer(3)]], device const uint* numVertices [[buffer(4)]], device const float* inf [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numV = numVertices[0];
  if (gid < numV) {
    uint base = inOffsets[gid];
    uint endv = inOffsets[(gid + 1u)];
    uint deg = (endv - base);
    float infV = inf[0];
    float best = dist[gid];
    for (int k = 0; k < (int)(deg); ++k) {
      uint eidx = (base + ((uint)(k)));
      uint u = inSrcs[eidx];
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
const char* k_metal_navatala_graph_label_propagate_edges_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_graph_label_propagate_edges_u32(device const uint* srcs [[buffer(0)]], device const uint* dsts [[buffer(1)]], device atomic_uint* labels [[buffer(2)]], device const uint* numEdges [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numE = numEdges[0];
  if (gid < numE) {
    uint u = srcs[gid];
    uint v = dsts[gid];
    uint lu = labels[u];
    uint lv = labels[v];
    uint prevLv = atomic_fetch_min_explicit(((device atomic_uint*)(&(labels[v]))), lu, memory_order_relaxed);
    uint prevLu = atomic_fetch_min_explicit(((device atomic_uint*)(&(labels[u]))), lv, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_graph_symmetrize_reverse_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_symmetrize_reverse_u32(device const uint* srcs [[buffer(0)]], device const uint* dsts [[buffer(1)]], device const uint* numEdges [[buffer(2)]], device uint* outSrcs [[buffer(3)]], device uint* outDsts [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numE = numEdges[0];
  if (gid < numE) {
    uint u = srcs[gid];
    uint v = dsts[gid];
    outSrcs[gid] = v;
    outDsts[gid] = u;
  }
}

)kernel";
const char* k_metal_navatala_graph_apply_renumber_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_apply_renumber_u32(device const uint* mapping [[buffer(0)]], device const uint* numVertices [[buffer(1)]], device uint* newIds [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numV = numVertices[0];
  if (gid < numV) {
    newIds[gid] = mapping[gid];
  }
}

)kernel";
const char* k_metal_navatala_graph_peel_below_threshold_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_peel_below_threshold_u32(device uint* _active [[buffer(0)]], device const uint* degrees [[buffer(1)]], device const uint* threshold [[buffer(2)]], device const uint* numVertices [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numV = numVertices[0];
  if (gid < numV) {
    uint deg = degrees[gid];
    uint k = threshold[0];
    if (deg < k) {
      _active[gid] = 0u;
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_pair_intersection_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_pair_intersection_u32(device const uint* offsets [[buffer(0)]], device const uint* indices [[buffer(1)]], device const uint* pairsA [[buffer(2)]], device const uint* pairsB [[buffer(3)]], device const uint* numPairs [[buffer(4)]], device uint* inter [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numP = numPairs[0];
  if (gid < numP) {
    uint a = pairsA[gid];
    uint b = pairsB[gid];
    uint baseU = offsets[a];
    uint endU = offsets[(a + 1u)];
    uint baseV = offsets[b];
    uint endV = offsets[(b + 1u)];
    uint degU = (endU - baseU);
    uint interAccum = 0u;
    uint loAccum = 0u;
    uint hiAccum = 0u;
    for (int k = 0; k < (int)(degU); ++k) {
      uint eidx = (baseU + ((uint)(k)));
      uint w = indices[eidx];
      loAccum = baseV;
      hiAccum = endV;
      for (int _bs = 0; _bs < (int)(32u); ++_bs) {
        uint lo = loAccum;
        uint hi = hiAccum;
        if (lo < hi) {
          uint mid = ((lo + hi) / 2u);
          uint midval = indices[mid];
          bool goRight = (midval < w);
          uint newLo = ((goRight) ? ((mid + 1u)) : (lo));
          uint newHi = ((goRight) ? (hi) : (mid));
          loAccum = newLo;
          hiAccum = newHi;
        }
      }
      uint lb = loAccum;
      if (lb < endV) {
        uint cand = indices[lb];
        if (cand == w) {
          interAccum = (interAccum + 1u);
        }
      }
    }
    inter[gid] = interAccum;
  }
}

)kernel";
const char* k_metal_navatala_graph_jaccard_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_jaccard_f32(device const uint* inter [[buffer(0)]], device const uint* degrees [[buffer(1)]], device const uint* pairsA [[buffer(2)]], device const uint* pairsB [[buffer(3)]], device const uint* numPairs [[buffer(4)]], device float* coeff [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numP = numPairs[0];
  if (gid < numP) {
    uint a = pairsA[gid];
    uint b = pairsB[gid];
    uint iU = inter[gid];
    uint dAU = degrees[a];
    uint dBU = degrees[b];
    float iF = ((float)(iU));
    float dAF = ((float)(dAU));
    float dBF = ((float)(dBU));
    coeff[gid] = (((((dAF + dBF) - iF) > as_type<float>(0x00000000u))) ? ((iF / ((dAF + dBF) - iF))) : (as_type<float>(0x00000000u)));
  }
}

)kernel";
const char* k_metal_navatala_graph_overlap_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_overlap_f32(device const uint* inter [[buffer(0)]], device const uint* degrees [[buffer(1)]], device const uint* pairsA [[buffer(2)]], device const uint* pairsB [[buffer(3)]], device const uint* numPairs [[buffer(4)]], device float* coeff [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numP = numPairs[0];
  if (gid < numP) {
    uint a = pairsA[gid];
    uint b = pairsB[gid];
    uint iU = inter[gid];
    uint dAU = degrees[a];
    uint dBU = degrees[b];
    float iF = ((float)(iU));
    float dAF = ((float)(dAU));
    float dBF = ((float)(dBU));
    coeff[gid] = ((((((dAF < dBF)) ? (dAF) : (dBF)) > as_type<float>(0x00000000u))) ? ((iF / (((dAF < dBF)) ? (dAF) : (dBF)))) : (as_type<float>(0x00000000u)));
  }
}

)kernel";
const char* k_metal_navatala_graph_sorensen_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_sorensen_f32(device const uint* inter [[buffer(0)]], device const uint* degrees [[buffer(1)]], device const uint* pairsA [[buffer(2)]], device const uint* pairsB [[buffer(3)]], device const uint* numPairs [[buffer(4)]], device float* coeff [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numP = numPairs[0];
  if (gid < numP) {
    uint a = pairsA[gid];
    uint b = pairsB[gid];
    uint iU = inter[gid];
    uint dAU = degrees[a];
    uint dBU = degrees[b];
    float iF = ((float)(iU));
    float dAF = ((float)(dAU));
    float dBF = ((float)(dBU));
    coeff[gid] = ((((dAF + dBF) > as_type<float>(0x00000000u))) ? (((as_type<float>(0x40000000u) * iF) / (dAF + dBF))) : (as_type<float>(0x00000000u)));
  }
}

)kernel";
const char* k_metal_navatala_graph_cosine_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_cosine_f32(device const uint* inter [[buffer(0)]], device const uint* degrees [[buffer(1)]], device const uint* pairsA [[buffer(2)]], device const uint* pairsB [[buffer(3)]], device const uint* numPairs [[buffer(4)]], device float* coeff [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numP = numPairs[0];
  if (gid < numP) {
    uint a = pairsA[gid];
    uint b = pairsB[gid];
    uint iU = inter[gid];
    uint dAU = degrees[a];
    uint dBU = degrees[b];
    float iF = ((float)(iU));
    float dAF = ((float)(dAU));
    float dBF = ((float)(dBU));
    coeff[gid] = (((sqrt((dAF * dBF)) > as_type<float>(0x00000000u))) ? ((iF / sqrt((dAF * dBF)))) : (as_type<float>(0x00000000u)));
  }
}

)kernel";
const char* k_metal_navatala_graph_uniform_random_walk_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_uniform_random_walk_u32(device const uint* offsets [[buffer(0)]], device const uint* indices [[buffer(1)]], device const uint* startVertices [[buffer(2)]], device const uint* maxLen [[buffer(3)]], device const uint* numWalks [[buffer(4)]], device const uint* seed [[buffer(5)]], device uint* walks [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numW = numWalks[0];
  if (gid < numW) {
    uint maxL = maxLen[0];
    uint seedV = seed[0];
    uint rowBase = (gid * maxL);
    uint curAccum = startVertices[gid];
    uint rngAccum = (seedV + gid);
    for (int s = 0; s < (int)(maxL); ++s) {
      uint cur = curAccum;
      uint pos = (rowBase + ((uint)(s)));
      walks[pos] = cur;
      uint off = offsets[cur];
      uint off1 = offsets[(cur + 1u)];
      uint deg = (off1 - off);
      uint rng = rngAccum;
      uint rngNext = ((rng * 1103515245u) + 12345u);
      rngAccum = rngNext;
      if (deg > 0u) {
        uint pick = (rngNext % deg);
        uint nbr = indices[(off + pick)];
        curAccum = nbr;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_out_degree_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_out_degree_u32(device const uint* offsets [[buffer(0)]], device const uint* numVertices [[buffer(1)]], device uint* degrees [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numV = numVertices[0];
  if (gid < numV) {
    uint hi = offsets[(gid + 1u)];
    uint lo = offsets[gid];
    uint deg = (hi - lo);
    degrees[gid] = deg;
  }
}

)kernel";
const char* k_metal_navatala_graph_in_degree_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_in_degree_u32(device const uint* offsets [[buffer(0)]], device const uint* numVertices [[buffer(1)]], device uint* degrees [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numV = numVertices[0];
  if (gid < numV) {
    uint hi = offsets[(gid + 1u)];
    uint lo = offsets[gid];
    uint deg = (hi - lo);
    degrees[gid] = deg;
  }
}

)kernel";
const char* k_metal_navatala_graph_rbf_affinity_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_rbf_affinity_f32(device const float* data [[buffer(0)]], device const float* sigma [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* d [[buffer(3)]], device float* W [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint dVal = d[0];
  float sigmaVal = sigma[0];
  uint totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / nVal);
    uint j = (gid % nVal);
    bool isDiag = (i == j);
    if (isDiag) {
      W[gid] = as_type<float>(0x00000000u);
    } else {
      float distSq = as_type<float>(0x00000000u);
      uint rowI = (i * dVal);
      uint rowJ = (j * dVal);
      uint idxI = rowI;
      uint idxJ = rowJ;
      float xi = data[idxI];
      float xj = data[idxJ];
      float diff = (xi - xj);
      float diffSq = (diff * diff);
      float sigmaSq = (sigmaVal * sigmaVal);
      float twoSigmaSq = (as_type<float>(0x40000000u) * sigmaSq);
      float exponent = ((as_type<float>(0x00000000u) - diffSq) / twoSigmaSq);
      float affinity = exp(exponent);
      W[gid] = affinity;
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_compute_degree_matrix_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_compute_degree_matrix_f32(device const float* W [[buffer(0)]], device const uint* n [[buffer(1)]], device float* D [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint wgid = ((uint)(int(__tgid.x)));
  threadgroup float sdata[256];
  uint nVal = n[0];
  uint rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    uint rowBase = (rowIdx * nVal);
    bool colInBounds = (lid < nVal);
    uint elemIdx = (rowBase + lid);
    if (colInBounds) {
      float wVal = W[elemIdx];
      sdata[lid] = wVal;
    } else {
      sdata[lid] = as_type<float>(0x00000000u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint degF32RedStride = 128u;
    for (int degF32RedStep = 0; degF32RedStep < (int)(8); ++degF32RedStep) {
      uint degF32Stride = degF32RedStride;
      if (lid < degF32Stride) {
        float other = sdata[(lid + degF32Stride)];
        float mine = sdata[lid];
        float sum = (mine + other);
        sdata[lid] = sum;
      }
      uint degF32StrideToHalve = degF32RedStride;
      uint degF32NextStride = (degF32StrideToHalve >> 1u);
      degF32RedStride = degF32NextStride;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
      float degreeVal = sdata[0];
      D[rowIdx] = degreeVal;
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_compute_laplacian_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_compute_laplacian_f32(device const float* W [[buffer(0)]], device const float* D [[buffer(1)]], device const uint* n [[buffer(2)]], device float* L [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / nVal);
    uint j = (gid % nVal);
    bool isDiag = (i == j);
    if (isDiag) {
      float dVal = D[i];
      L[gid] = dVal;
    } else {
      float wVal = W[gid];
      float negW = (as_type<float>(0x00000000u) - wVal);
      L[gid] = negW;
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_normalized_laplacian_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_normalized_laplacian_f32(device const float* W [[buffer(0)]], device const float* D [[buffer(1)]], device const uint* n [[buffer(2)]], device float* L_sym [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / nVal);
    uint j = (gid % nVal);
    float di = D[i];
    float dj = D[j];
    float eps = as_type<float>(0x2edbe6ffu);
    bool diZero = (di < eps);
    bool djZero = (dj < eps);
    bool isolated = (diZero || djZero);
    bool isDiag = (i == j);
    if (isolated) {
      if (isDiag) {
        L_sym[gid] = as_type<float>(0x3f800000u);
      } else {
        L_sym[gid] = as_type<float>(0x00000000u);
      }
    } else {
      float wVal = W[gid];
      float didjProd = (di * dj);
      float sqrtDiDj = sqrt(didjProd);
      float normalized = (wVal / sqrtDiDj);
      if (isDiag) {
        float lSymVal = (as_type<float>(0x3f800000u) - normalized);
        L_sym[gid] = lSymVal;
      } else {
        float lSymVal = (as_type<float>(0x00000000u) - normalized);
        L_sym[gid] = lSymVal;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_normalize_eigenvector_rows_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_normalize_eigenvector_rows_f32(device const float* V [[buffer(0)]], device const uint* n [[buffer(1)]], device const uint* k [[buffer(2)]], device float* V_norm [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint wgid = ((uint)(int(__tgid.x)));
  threadgroup float sdata[256];
  uint nVal = n[0];
  uint kVal = k[0];
  uint rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    uint rowBase = (rowIdx * kVal);
    bool colInBounds = (lid < kVal);
    uint elemIdx = (rowBase + lid);
    if (colInBounds) {
      float vVal = V[elemIdx];
      float vSq = (vVal * vVal);
      sdata[lid] = vSq;
    } else {
      sdata[lid] = as_type<float>(0x00000000u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint normF32RedStride = 128u;
    for (int normF32RedStep = 0; normF32RedStep < (int)(8); ++normF32RedStep) {
      uint normF32Stride = normF32RedStride;
      if (lid < normF32Stride) {
        float other = sdata[(lid + normF32Stride)];
        float mine = sdata[lid];
        float sum = (mine + other);
        sdata[lid] = sum;
      }
      uint normF32StrideToHalve = normF32RedStride;
      uint normF32NextStride = (normF32StrideToHalve >> 1u);
      normF32RedStride = normF32NextStride;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
      float sumSq = sdata[0];
      float normVal = sqrt(sumSq);
      sdata[0] = normVal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (colInBounds) {
      float norm = sdata[0];
      float vVal2 = V[elemIdx];
      float eps = as_type<float>(0x2edbe6ffu);
      bool normZero = (norm < eps);
      if (normZero) {
        V_norm[elemIdx] = as_type<float>(0x00000000u);
      } else {
        float normalized = (vVal2 / norm);
        V_norm[elemIdx] = normalized;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_spectral_embedding_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_spectral_embedding_f32(device const float* eigenvectors [[buffer(0)]], device const uint* n [[buffer(1)]], device const uint* k [[buffer(2)]], device float* embedding [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint kVal = k[0];
  uint totalElems = (nVal * kVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint row = (gid / kVal);
    uint col = (gid % kVal);
    uint srcIdx = ((col * nVal) + row);
    float eigVal = eigenvectors[srcIdx];
    embedding[gid] = eigVal;
  }
}

)kernel";
const char* k_metal_navatala_graph_rmat_generator_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_graph_rmat_generator_f32(device const float* random [[buffer(0)]], device const float* randomWeights [[buffer(1)]], device const float* a [[buffer(2)]], device const float* b [[buffer(3)]], device const float* c [[buffer(4)]], device const float* d [[buffer(5)]], device const uint* n [[buffer(6)]], device const uint* scale [[buffer(7)]], device const uint* numEdges [[buffer(8)]], device uint* srcNodes [[buffer(9)]], device uint* dstNodes [[buffer(10)]], device float* weights [[buffer(11)]], device atomic_uint* edgeCount [[buffer(12)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numE = numEdges[0u];
  bool inBounds = (gid < numE);
  if (inBounds) {
    float probA = a[0u];
    float probB = b[0u];
    float probC = c[0u];
    uint scaleVal = scale[0u];
    float ab = (probA + probB);
    float abc = (ab + probC);
    uint srcAccum = 0u;
    uint dstAccum = 0u;
    for (int level = 0; level < (int)(scaleVal); ++level) {
      uint levelU32 = ((uint)(level));
      uint randIdx = ((gid * scaleVal) + levelU32);
      float r = random[randIdx];
      uint bitPos = ((scaleVal - levelU32) - 1u);
      uint bitMask = (1u << bitPos);
      bool inQ01 = (r >= probA);
      bool inQ10 = (r >= ab);
      bool inQ11 = (r >= abc);
      uint srcBit = ((inQ10) ? (bitMask) : (0u));
      uint currentSrc = srcAccum;
      uint newSrc = (currentSrc | srcBit);
      srcAccum = newSrc;
      bool dstBitCond = ((inQ01 && (!inQ10)) || inQ11);
      uint dstBit = ((dstBitCond) ? (bitMask) : (0u));
      uint currentDst = dstAccum;
      uint newDst = (currentDst | dstBit);
      dstAccum = newDst;
    }
    uint src = srcAccum;
    uint dst = dstAccum;
    float w = randomWeights[gid];
    uint slot = atomic_fetch_add_explicit(((device atomic_uint*)(&(edgeCount[0u]))), 1u, memory_order_relaxed);
    srcNodes[slot] = src;
    dstNodes[slot] = dst;
    weights[slot] = w;
  }
}

)kernel";
const char* k_metal_navatala_graph_erdos_renyi_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_graph_erdos_renyi_f32(device const float* random [[buffer(0)]], device const float* randomWeights [[buffer(1)]], device const float* p [[buffer(2)]], device const uint* n [[buffer(3)]], device uint* srcNodes [[buffer(4)]], device uint* dstNodes [[buffer(5)]], device float* weights [[buffer(6)]], device atomic_uint* edgeCount [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0u];
  uint nMinus1 = (numN - 1u);
  uint totalPairs = ((numN * nMinus1) / 2u);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    float prob = p[0u];
    float gidF = ((float)(gid));
    float discriminant = (as_type<float>(0x3f800000u) + (as_type<float>(0x41000000u) * gidF));
    float sqrtDisc = sqrt(discriminant);
    float iFloat = ((as_type<float>(0x3f800000u) + sqrtDisc) / as_type<float>(0x40000000u));
    uint i = ((uint)(floor(iFloat)));
    uint iTriangle = ((i * (i - 1u)) / 2u);
    uint j = (gid - iTriangle);
    float r = random[gid];
    bool includeEdge = (r < prob);
    if (includeEdge) {
      float w = randomWeights[gid];
      uint slot = atomic_fetch_add_explicit(((device atomic_uint*)(&(edgeCount[0u]))), 1u, memory_order_relaxed);
      srcNodes[slot] = j;
      dstNodes[slot] = i;
      weights[slot] = w;
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_barabasi_albert_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_graph_barabasi_albert_f32(device const uint* cumDegree [[buffer(0)]], device const float* random [[buffer(1)]], device const float* randomWeights [[buffer(2)]], device const uint* newNode [[buffer(3)]], device const uint* totalDegree [[buffer(4)]], device const uint* m [[buffer(5)]], device const uint* numExistingNodes [[buffer(6)]], device uint* srcNodes [[buffer(7)]], device uint* dstNodes [[buffer(8)]], device float* weights [[buffer(9)]], device atomic_uint* edgeCount [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numM = m[0u];
  bool inBounds = (gid < numM);
  if (inBounds) {
    uint newNodeIdx = newNode[0u];
    uint totalDeg = totalDegree[0u];
    uint numExisting = numExistingNodes[0u];
    float r = random[gid];
    float totalDegF = ((float)(totalDeg));
    float targetDegF = (r * totalDegF);
    uint targetDeg = ((uint)(targetDegF));
    uint loAccum = 0u;
    uint hiAccum = numExisting;
    for (int _ = 0; _ < (int)(32u); ++_) {
      uint lo = loAccum;
      uint hi = hiAccum;
      bool stillSearching = (lo < hi);
      if (stillSearching) {
        uint mid = ((lo + hi) / 2u);
        uint cumVal = cumDegree[mid];
        bool goRight = (cumVal <= targetDeg);
        uint newLo = ((goRight) ? ((mid + 1u)) : (lo));
        uint newHi = ((goRight) ? (hi) : (mid));
        loAccum = newLo;
        hiAccum = newHi;
      }
    }
    uint finalLo = loAccum;
    uint targetNode = (((finalLo > 0u)) ? ((finalLo - 1u)) : (0u));
    float w = randomWeights[gid];
    uint slot = atomic_fetch_add_explicit(((device atomic_uint*)(&(edgeCount[0u]))), 1u, memory_order_relaxed);
    srcNodes[slot] = newNodeIdx;
    dstNodes[slot] = targetNode;
    weights[slot] = w;
  }
}

)kernel";
const char* k_metal_navatala_graph_compute_modularity_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_graph_compute_modularity_f32(device const float* adjacencyValues [[buffer(0)]], device const uint* rowPtr [[buffer(1)]], device const uint* colInd [[buffer(2)]], device const uint* clusters [[buffer(3)]], device const uint* n [[buffer(4)]], device const uint* numClusters [[buffer(5)]], device const float* totalEdgeWeight [[buffer(6)]], device float* modularity [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  float totalWeight = totalEdgeWeight[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = clusters[gid];
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + 1u);
    uint rowEnd = rowPtr[gidPlus1];
    float degreeAccum = as_type<float>(0x00000000u);
    float intraWeightAccum = as_type<float>(0x00000000u);
    for (int idx = 0; idx < (int)(rowEnd); ++idx) {
      uint idxU32 = ((uint)(idx));
      bool idxInRange = (idxU32 >= rowStart);
      if (idxInRange) {
        float edgeWeight = adjacencyValues[idxU32];
        uint neighbor = colInd[idxU32];
        uint neighborCluster = clusters[neighbor];
        float currentDegree = degreeAccum;
        float newDegree = (currentDegree + edgeWeight);
        degreeAccum = newDegree;
        bool sameCluster = (neighborCluster == myCluster);
        float weightToAdd = ((sameCluster) ? (edgeWeight) : (as_type<float>(0x00000000u)));
        float currentIntraWeight = intraWeightAccum;
        float newIntraWeight = (currentIntraWeight + weightToAdd);
        intraWeightAccum = newIntraWeight;
      }
    }
    float degree = degreeAccum;
    float intraWeight = intraWeightAccum;
    float invTotalWeight = (as_type<float>(0x3f800000u) / totalWeight);
    float eContrib = (intraWeight * invTotalWeight);
    float degreeNorm = (degree * invTotalWeight);
    float aContrib = (degreeNorm * degreeNorm);
    float contribution = (eContrib - aContrib);
    atomic_fetch_add_explicit((device atomic_float*)(&modularity[0]), contribution, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_graph_modularity_matrix_vec_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_modularity_matrix_vec_f32(device const float* adjacencyValues [[buffer(0)]], device const uint* rowPtr [[buffer(1)]], device const uint* colInd [[buffer(2)]], device const float* x [[buffer(3)]], device const float* degree [[buffer(4)]], device const float* totalEdgeWeight [[buffer(5)]], device const uint* n [[buffer(6)]], device const float* dotDX [[buffer(7)]], device float* y [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  float totalWeight = totalEdgeWeight[0];
  float dotDXVal = dotDX[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + 1u);
    uint rowEnd = rowPtr[gidPlus1];
    float axValAccum = as_type<float>(0x00000000u);
    for (int idx = 0; idx < (int)(rowEnd); ++idx) {
      uint idxU32 = ((uint)(idx));
      bool idxInRange = (idxU32 >= rowStart);
      if (idxInRange) {
        float aij = adjacencyValues[idxU32];
        uint j = colInd[idxU32];
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
const char* k_metal_navatala_graph_construct_indicator_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_construct_indicator_f32(device const uint* clusters [[buffer(0)]], device const uint* clusterIndex [[buffer(1)]], device const uint* n [[buffer(2)]], device float* indicator [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint targetCluster = clusterIndex[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint nodeCluster = clusters[gid];
    bool isTarget = (nodeCluster == targetCluster);
    float indicatorVal = ((isTarget) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    indicator[gid] = indicatorVal;
  }
}

)kernel";
const char* k_metal_navatala_graph_compute_edge_cut_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_graph_compute_edge_cut_f32(device const float* adjacencyValues [[buffer(0)]], device const uint* rowPtr [[buffer(1)]], device const uint* colInd [[buffer(2)]], device const uint* clusters [[buffer(3)]], device const uint* n [[buffer(4)]], device float* edgeCut [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = clusters[gid];
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + 1u);
    uint rowEnd = rowPtr[gidPlus1];
    float localCut = as_type<float>(0x00000000u);
    for (int edgeIdx = 0; edgeIdx < (int)(rowEnd); ++edgeIdx) {
      uint edgeIdxU = ((uint)(edgeIdx));
      bool edgeInRange = (edgeIdxU >= rowStart);
      if (edgeInRange) {
        uint neighbor = colInd[edgeIdxU];
        uint neighborCluster = clusters[neighbor];
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
    bool hasContribution = (finalLocalCut > as_type<float>(0x00000000u));
    if (hasContribution) {
      atomic_fetch_add_explicit((device atomic_float*)(&edgeCut[0]), finalLocalCut, memory_order_relaxed);
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_compute_partition_cost_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_graph_compute_partition_cost_f32(device const float* edgeCutPerCluster [[buffer(0)]], device const uint* clusterSizes [[buffer(1)]], device const uint* numClusters [[buffer(2)]], device float* cost [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numClustersVal = numClusters[0];
  bool inBounds = (gid < numClustersVal);
  if (inBounds) {
    float clusterEdgeCut = edgeCutPerCluster[gid];
    uint clusterSize = clusterSizes[gid];
    bool sizeIsZero = (clusterSize == 0u);
    float clusterSizeF = ((float)(clusterSize));
    float normalizedCost = ((sizeIsZero) ? (as_type<float>(0x00000000u)) : ((clusterEdgeCut / clusterSizeF)));
    bool hasContribution = (normalizedCost > as_type<float>(0x00000000u));
    if (hasContribution) {
      atomic_fetch_add_explicit((device atomic_float*)(&cost[0]), normalizedCost, memory_order_relaxed);
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_analyze_partition_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_graph_analyze_partition_f32(device const float* adjacencyValues [[buffer(0)]], device const uint* rowPtr [[buffer(1)]], device const uint* colInd [[buffer(2)]], device const uint* clusters [[buffer(3)]], device const uint* n [[buffer(4)]], device const uint* numClusters [[buffer(5)]], device float* edgeCut [[buffer(6)]], device float* cost [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint numClustersVal = numClusters[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = clusters[gid];
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + 1u);
    uint rowEnd = rowPtr[gidPlus1];
    float nodeEdgeCut = as_type<float>(0x00000000u);
    for (int edgeIdx = 0; edgeIdx < (int)(rowEnd); ++edgeIdx) {
      uint edgeIdxU = ((uint)(edgeIdx));
      bool edgeInRange = (edgeIdxU >= rowStart);
      if (edgeInRange) {
        uint neighbor = colInd[edgeIdxU];
        uint neighborCluster = clusters[neighbor];
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
    bool hasEdgeCut = (finalNodeEdgeCut > as_type<float>(0x00000000u));
    if (hasEdgeCut) {
      atomic_fetch_add_explicit((device atomic_float*)(&edgeCut[0]), finalNodeEdgeCut, memory_order_relaxed);
    }
    float nValF = ((float)(nVal));
    float nodeCostContrib = (finalNodeEdgeCut / nValF);
    bool hasCostContrib = (nodeCostContrib > as_type<float>(0x00000000u));
    if (hasCostContrib) {
      atomic_fetch_add_explicit((device atomic_float*)(&cost[0]), nodeCostContrib, memory_order_relaxed);
    }
  }
}

)kernel";
const char* k_metal_navatala_graph_transform_eigen_matrix_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_transform_eigen_matrix_f32(device const float* eigenvectors [[buffer(0)]], device const uint* n [[buffer(1)]], device const uint* k [[buffer(2)]], device float* transformed [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint kVal = k[0];
  uint totalElems = (kVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint eigIdx = (gid / nVal);
    uint sampleIdx = (gid % nVal);
    uint srcIdx = ((sampleIdx * kVal) + eigIdx);
    float val = eigenvectors[srcIdx];
    transformed[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_graph_laplacian_from_adjacency_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_graph_laplacian_from_adjacency_f32(device const float* adjacencyValues [[buffer(0)]], device const uint* rowPtr [[buffer(1)]], device const uint* colInd [[buffer(2)]], device const uint* n [[buffer(3)]], device float* laplacianValues [[buffer(4)]], device float* degree [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint rowIdx = gid;
  bool inBounds = (rowIdx < nVal);
  if (inBounds) {
    uint rowStart = rowPtr[rowIdx];
    uint one32 = ((uint)(1));
    uint rowIdxPlusOne = (rowIdx + one32);
    uint rowEnd = rowPtr[rowIdxPlusOne];
    float degreeSumAccum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((rowEnd - rowStart)); ++j) {
      uint jU32 = ((uint)(j));
      uint absIdx = (rowStart + jU32);
      float adjVal = adjacencyValues[absIdx];
      float negAdjVal = (as_type<float>(0x00000000u) - adjVal);
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

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_pack_f32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_pack_f32 = {
  1,
  "navatala_graph_pack_f32",
  "metal",
  "navatala_graph_pack_f32",
  "kernel:metal:navatala_graph_pack_f32",
  "abi-r1:metal:navatala_graph_pack_f32",
  "abi-r1:metal:navatala_graph_pack_f32",
  4,
  kAbiArgs_metal_navatala_graph_pack_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_pack_i32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_pack_i32 = {
  1,
  "navatala_graph_pack_i32",
  "metal",
  "navatala_graph_pack_i32",
  "kernel:metal:navatala_graph_pack_i32",
  "abi-r1:metal:navatala_graph_pack_i32",
  "abi-r1:metal:navatala_graph_pack_i32",
  4,
  kAbiArgs_metal_navatala_graph_pack_i32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_pack_u32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_pack_u32 = {
  1,
  "navatala_graph_pack_u32",
  "metal",
  "navatala_graph_pack_u32",
  "kernel:metal:navatala_graph_pack_u32",
  "abi-r1:metal:navatala_graph_pack_u32",
  "abi-r1:metal:navatala_graph_pack_u32",
  4,
  kAbiArgs_metal_navatala_graph_pack_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_unpack_f32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_unpack_f32 = {
  1,
  "navatala_graph_unpack_f32",
  "metal",
  "navatala_graph_unpack_f32",
  "kernel:metal:navatala_graph_unpack_f32",
  "abi-r1:metal:navatala_graph_unpack_f32",
  "abi-r1:metal:navatala_graph_unpack_f32",
  4,
  kAbiArgs_metal_navatala_graph_unpack_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_unpack_i32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_unpack_i32 = {
  1,
  "navatala_graph_unpack_i32",
  "metal",
  "navatala_graph_unpack_i32",
  "kernel:metal:navatala_graph_unpack_i32",
  "abi-r1:metal:navatala_graph_unpack_i32",
  "abi-r1:metal:navatala_graph_unpack_i32",
  4,
  kAbiArgs_metal_navatala_graph_unpack_i32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_unpack_u32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_unpack_u32 = {
  1,
  "navatala_graph_unpack_u32",
  "metal",
  "navatala_graph_unpack_u32",
  "kernel:metal:navatala_graph_unpack_u32",
  "abi-r1:metal:navatala_graph_unpack_u32",
  "abi-r1:metal:navatala_graph_unpack_u32",
  4,
  kAbiArgs_metal_navatala_graph_unpack_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_pack_f32x3[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_pack_f32x3 = {
  1,
  "navatala_graph_pack_f32x3",
  "metal",
  "navatala_graph_pack_f32x3",
  "kernel:metal:navatala_graph_pack_f32x3",
  "abi-r1:metal:navatala_graph_pack_f32x3",
  "abi-r1:metal:navatala_graph_pack_f32x3",
  5,
  kAbiArgs_metal_navatala_graph_pack_f32x3
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_unpack_f32x3[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_unpack_f32x3 = {
  1,
  "navatala_graph_unpack_f32x3",
  "metal",
  "navatala_graph_unpack_f32x3",
  "kernel:metal:navatala_graph_unpack_f32x3",
  "abi-r1:metal:navatala_graph_unpack_f32x3",
  "abi-r1:metal:navatala_graph_unpack_f32x3",
  5,
  kAbiArgs_metal_navatala_graph_unpack_f32x3
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_pack_f32x6[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_pack_f32x6 = {
  1,
  "navatala_graph_pack_f32x6",
  "metal",
  "navatala_graph_pack_f32x6",
  "kernel:metal:navatala_graph_pack_f32x6",
  "abi-r1:metal:navatala_graph_pack_f32x6",
  "abi-r1:metal:navatala_graph_pack_f32x6",
  5,
  kAbiArgs_metal_navatala_graph_pack_f32x6
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_unpack_f32x6[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_unpack_f32x6 = {
  1,
  "navatala_graph_unpack_f32x6",
  "metal",
  "navatala_graph_unpack_f32x6",
  "kernel:metal:navatala_graph_unpack_f32x6",
  "abi-r1:metal:navatala_graph_unpack_f32x6",
  "abi-r1:metal:navatala_graph_unpack_f32x6",
  5,
  kAbiArgs_metal_navatala_graph_unpack_f32x6
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_pack_f32x9[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_pack_f32x9 = {
  1,
  "navatala_graph_pack_f32x9",
  "metal",
  "navatala_graph_pack_f32x9",
  "kernel:metal:navatala_graph_pack_f32x9",
  "abi-r1:metal:navatala_graph_pack_f32x9",
  "abi-r1:metal:navatala_graph_pack_f32x9",
  5,
  kAbiArgs_metal_navatala_graph_pack_f32x9
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_unpack_f32x9[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_unpack_f32x9 = {
  1,
  "navatala_graph_unpack_f32x9",
  "metal",
  "navatala_graph_unpack_f32x9",
  "kernel:metal:navatala_graph_unpack_f32x9",
  "abi-r1:metal:navatala_graph_unpack_f32x9",
  "abi-r1:metal:navatala_graph_unpack_f32x9",
  5,
  kAbiArgs_metal_navatala_graph_unpack_f32x9
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_degree_out_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_degree_out_u32 = {
  1,
  "navatala_graph_degree_out_u32",
  "metal",
  "navatala_graph_degree_out_u32",
  "kernel:metal:navatala_graph_degree_out_u32",
  "abi-r1:metal:navatala_graph_degree_out_u32",
  "abi-r1:metal:navatala_graph_degree_out_u32",
  3,
  kAbiArgs_metal_navatala_graph_degree_out_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_degree_in_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_degree_in_u32 = {
  1,
  "navatala_graph_degree_in_u32",
  "metal",
  "navatala_graph_degree_in_u32",
  "kernel:metal:navatala_graph_degree_in_u32",
  "abi-r1:metal:navatala_graph_degree_in_u32",
  "abi-r1:metal:navatala_graph_degree_in_u32",
  3,
  kAbiArgs_metal_navatala_graph_degree_in_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_spmv_weighted_f32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_spmv_weighted_f32 = {
  1,
  "navatala_graph_spmv_weighted_f32",
  "metal",
  "navatala_graph_spmv_weighted_f32",
  "kernel:metal:navatala_graph_spmv_weighted_f32",
  "abi-r1:metal:navatala_graph_spmv_weighted_f32",
  "abi-r1:metal:navatala_graph_spmv_weighted_f32",
  6,
  kAbiArgs_metal_navatala_graph_spmv_weighted_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_spmv_weighted_subgroup_f32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_spmv_weighted_subgroup_f32 = {
  1,
  "navatala_graph_spmv_weighted_subgroup_f32",
  "metal",
  "navatala_graph_spmv_weighted_subgroup_f32",
  "kernel:metal:navatala_graph_spmv_weighted_subgroup_f32",
  "abi-r1:metal:navatala_graph_spmv_weighted_subgroup_f32",
  "abi-r1:metal:navatala_graph_spmv_weighted_subgroup_f32",
  6,
  kAbiArgs_metal_navatala_graph_spmv_weighted_subgroup_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_spmv_unweighted_f32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_spmv_unweighted_f32 = {
  1,
  "navatala_graph_spmv_unweighted_f32",
  "metal",
  "navatala_graph_spmv_unweighted_f32",
  "kernel:metal:navatala_graph_spmv_unweighted_f32",
  "abi-r1:metal:navatala_graph_spmv_unweighted_f32",
  "abi-r1:metal:navatala_graph_spmv_unweighted_f32",
  5,
  kAbiArgs_metal_navatala_graph_spmv_unweighted_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_spmv_unweighted_subgroup_f32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_spmv_unweighted_subgroup_f32 = {
  1,
  "navatala_graph_spmv_unweighted_subgroup_f32",
  "metal",
  "navatala_graph_spmv_unweighted_subgroup_f32",
  "kernel:metal:navatala_graph_spmv_unweighted_subgroup_f32",
  "abi-r1:metal:navatala_graph_spmv_unweighted_subgroup_f32",
  "abi-r1:metal:navatala_graph_spmv_unweighted_subgroup_f32",
  5,
  kAbiArgs_metal_navatala_graph_spmv_unweighted_subgroup_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_axpy2_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "z", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_axpy2_f32 = {
  1,
  "navatala_graph_axpy2_f32",
  "metal",
  "navatala_graph_axpy2_f32",
  "kernel:metal:navatala_graph_axpy2_f32",
  "abi-r1:metal:navatala_graph_axpy2_f32",
  "abi-r1:metal:navatala_graph_axpy2_f32",
  6,
  kAbiArgs_metal_navatala_graph_axpy2_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_scale_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_scale_f32 = {
  1,
  "navatala_graph_scale_f32",
  "metal",
  "navatala_graph_scale_f32",
  "kernel:metal:navatala_graph_scale_f32",
  "abi-r1:metal:navatala_graph_scale_f32",
  "abi-r1:metal:navatala_graph_scale_f32",
  4,
  kAbiArgs_metal_navatala_graph_scale_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_scale_add_bias_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "bias", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_scale_add_bias_f32 = {
  1,
  "navatala_graph_scale_add_bias_f32",
  "metal",
  "navatala_graph_scale_add_bias_f32",
  "kernel:metal:navatala_graph_scale_add_bias_f32",
  "abi-r1:metal:navatala_graph_scale_add_bias_f32",
  "abi-r1:metal:navatala_graph_scale_add_bias_f32",
  5,
  kAbiArgs_metal_navatala_graph_scale_add_bias_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_l1norm_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_l1norm_f32 = {
  1,
  "navatala_graph_l1norm_f32",
  "metal",
  "navatala_graph_l1norm_f32",
  "kernel:metal:navatala_graph_l1norm_f32",
  "abi-r1:metal:navatala_graph_l1norm_f32",
  "abi-r1:metal:navatala_graph_l1norm_f32",
  3,
  kAbiArgs_metal_navatala_graph_l1norm_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_recip_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_recip_f32 = {
  1,
  "navatala_graph_recip_f32",
  "metal",
  "navatala_graph_recip_f32",
  "kernel:metal:navatala_graph_recip_f32",
  "abi-r1:metal:navatala_graph_recip_f32",
  "abi-r1:metal:navatala_graph_recip_f32",
  2,
  kAbiArgs_metal_navatala_graph_recip_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_l1_diff_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_l1_diff_f32 = {
  1,
  "navatala_graph_l1_diff_f32",
  "metal",
  "navatala_graph_l1_diff_f32",
  "kernel:metal:navatala_graph_l1_diff_f32",
  "abi-r1:metal:navatala_graph_l1_diff_f32",
  "abi-r1:metal:navatala_graph_l1_diff_f32",
  4,
  kAbiArgs_metal_navatala_graph_l1_diff_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_count_diff_u32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_count_diff_u32 = {
  1,
  "navatala_graph_count_diff_u32",
  "metal",
  "navatala_graph_count_diff_u32",
  "kernel:metal:navatala_graph_count_diff_u32",
  "abi-r1:metal:navatala_graph_count_diff_u32",
  "abi-r1:metal:navatala_graph_count_diff_u32",
  4,
  kAbiArgs_metal_navatala_graph_count_diff_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_copy_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_copy_f32 = {
  1,
  "navatala_graph_copy_f32",
  "metal",
  "navatala_graph_copy_f32",
  "kernel:metal:navatala_graph_copy_f32",
  "abi-r1:metal:navatala_graph_copy_f32",
  "abi-r1:metal:navatala_graph_copy_f32",
  3,
  kAbiArgs_metal_navatala_graph_copy_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_copy_u32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_copy_u32 = {
  1,
  "navatala_graph_copy_u32",
  "metal",
  "navatala_graph_copy_u32",
  "kernel:metal:navatala_graph_copy_u32",
  "abi-r1:metal:navatala_graph_copy_u32",
  "abi-r1:metal:navatala_graph_copy_u32",
  3,
  kAbiArgs_metal_navatala_graph_copy_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_relax_vertex_pull_f32[] = {
  { "inOffsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "inSrcs", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "inWeights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dist", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "inf", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_relax_vertex_pull_f32 = {
  1,
  "navatala_graph_relax_vertex_pull_f32",
  "metal",
  "navatala_graph_relax_vertex_pull_f32",
  "kernel:metal:navatala_graph_relax_vertex_pull_f32",
  "abi-r1:metal:navatala_graph_relax_vertex_pull_f32",
  "abi-r1:metal:navatala_graph_relax_vertex_pull_f32",
  7,
  kAbiArgs_metal_navatala_graph_relax_vertex_pull_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_label_propagate_edges_u32[] = {
  { "srcs", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dsts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "labels", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numEdges", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labels", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_label_propagate_edges_u32 = {
  1,
  "navatala_graph_label_propagate_edges_u32",
  "metal",
  "navatala_graph_label_propagate_edges_u32",
  "kernel:metal:navatala_graph_label_propagate_edges_u32",
  "abi-r1:metal:navatala_graph_label_propagate_edges_u32",
  "abi-r1:metal:navatala_graph_label_propagate_edges_u32",
  5,
  kAbiArgs_metal_navatala_graph_label_propagate_edges_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_symmetrize_reverse_u32[] = {
  { "srcs", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dsts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numEdges", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outSrcs", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outDsts", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_symmetrize_reverse_u32 = {
  1,
  "navatala_graph_symmetrize_reverse_u32",
  "metal",
  "navatala_graph_symmetrize_reverse_u32",
  "kernel:metal:navatala_graph_symmetrize_reverse_u32",
  "abi-r1:metal:navatala_graph_symmetrize_reverse_u32",
  "abi-r1:metal:navatala_graph_symmetrize_reverse_u32",
  5,
  kAbiArgs_metal_navatala_graph_symmetrize_reverse_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_apply_renumber_u32[] = {
  { "mapping", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "newIds", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_apply_renumber_u32 = {
  1,
  "navatala_graph_apply_renumber_u32",
  "metal",
  "navatala_graph_apply_renumber_u32",
  "kernel:metal:navatala_graph_apply_renumber_u32",
  "abi-r1:metal:navatala_graph_apply_renumber_u32",
  "abi-r1:metal:navatala_graph_apply_renumber_u32",
  3,
  kAbiArgs_metal_navatala_graph_apply_renumber_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_peel_below_threshold_u32[] = {
  { "active", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "threshold", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numVertices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "active", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_peel_below_threshold_u32 = {
  1,
  "navatala_graph_peel_below_threshold_u32",
  "metal",
  "navatala_graph_peel_below_threshold_u32",
  "kernel:metal:navatala_graph_peel_below_threshold_u32",
  "abi-r1:metal:navatala_graph_peel_below_threshold_u32",
  "abi-r1:metal:navatala_graph_peel_below_threshold_u32",
  5,
  kAbiArgs_metal_navatala_graph_peel_below_threshold_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_pair_intersection_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numPairs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "inter", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_pair_intersection_u32 = {
  1,
  "navatala_graph_pair_intersection_u32",
  "metal",
  "navatala_graph_pair_intersection_u32",
  "kernel:metal:navatala_graph_pair_intersection_u32",
  "abi-r1:metal:navatala_graph_pair_intersection_u32",
  "abi-r1:metal:navatala_graph_pair_intersection_u32",
  6,
  kAbiArgs_metal_navatala_graph_pair_intersection_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_jaccard_f32[] = {
  { "inter", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numPairs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coeff", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_jaccard_f32 = {
  1,
  "navatala_graph_jaccard_f32",
  "metal",
  "navatala_graph_jaccard_f32",
  "kernel:metal:navatala_graph_jaccard_f32",
  "abi-r1:metal:navatala_graph_jaccard_f32",
  "abi-r1:metal:navatala_graph_jaccard_f32",
  6,
  kAbiArgs_metal_navatala_graph_jaccard_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_overlap_f32[] = {
  { "inter", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numPairs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coeff", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_overlap_f32 = {
  1,
  "navatala_graph_overlap_f32",
  "metal",
  "navatala_graph_overlap_f32",
  "kernel:metal:navatala_graph_overlap_f32",
  "abi-r1:metal:navatala_graph_overlap_f32",
  "abi-r1:metal:navatala_graph_overlap_f32",
  6,
  kAbiArgs_metal_navatala_graph_overlap_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_sorensen_f32[] = {
  { "inter", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numPairs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coeff", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_sorensen_f32 = {
  1,
  "navatala_graph_sorensen_f32",
  "metal",
  "navatala_graph_sorensen_f32",
  "kernel:metal:navatala_graph_sorensen_f32",
  "abi-r1:metal:navatala_graph_sorensen_f32",
  "abi-r1:metal:navatala_graph_sorensen_f32",
  6,
  kAbiArgs_metal_navatala_graph_sorensen_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_cosine_f32[] = {
  { "inter", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numPairs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coeff", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_cosine_f32 = {
  1,
  "navatala_graph_cosine_f32",
  "metal",
  "navatala_graph_cosine_f32",
  "kernel:metal:navatala_graph_cosine_f32",
  "abi-r1:metal:navatala_graph_cosine_f32",
  "abi-r1:metal:navatala_graph_cosine_f32",
  6,
  kAbiArgs_metal_navatala_graph_cosine_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_uniform_random_walk_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "startVertices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "maxLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numWalks", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "walks", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_uniform_random_walk_u32 = {
  1,
  "navatala_graph_uniform_random_walk_u32",
  "metal",
  "navatala_graph_uniform_random_walk_u32",
  "kernel:metal:navatala_graph_uniform_random_walk_u32",
  "abi-r1:metal:navatala_graph_uniform_random_walk_u32",
  "abi-r1:metal:navatala_graph_uniform_random_walk_u32",
  7,
  kAbiArgs_metal_navatala_graph_uniform_random_walk_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_out_degree_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_out_degree_u32 = {
  1,
  "navatala_graph_out_degree_u32",
  "metal",
  "navatala_graph_out_degree_u32",
  "kernel:metal:navatala_graph_out_degree_u32",
  "abi-r1:metal:navatala_graph_out_degree_u32",
  "abi-r1:metal:navatala_graph_out_degree_u32",
  3,
  kAbiArgs_metal_navatala_graph_out_degree_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_in_degree_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_in_degree_u32 = {
  1,
  "navatala_graph_in_degree_u32",
  "metal",
  "navatala_graph_in_degree_u32",
  "kernel:metal:navatala_graph_in_degree_u32",
  "abi-r1:metal:navatala_graph_in_degree_u32",
  "abi-r1:metal:navatala_graph_in_degree_u32",
  3,
  kAbiArgs_metal_navatala_graph_in_degree_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_rbf_affinity_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sigma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "W", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_rbf_affinity_f32 = {
  1,
  "navatala_graph_rbf_affinity_f32",
  "metal",
  "navatala_graph_rbf_affinity_f32",
  "kernel:metal:navatala_graph_rbf_affinity_f32",
  "abi-r1:metal:navatala_graph_rbf_affinity_f32",
  "abi-r1:metal:navatala_graph_rbf_affinity_f32",
  5,
  kAbiArgs_metal_navatala_graph_rbf_affinity_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_compute_degree_matrix_f32[] = {
  { "W", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "D", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_compute_degree_matrix_f32 = {
  1,
  "navatala_graph_compute_degree_matrix_f32",
  "metal",
  "navatala_graph_compute_degree_matrix_f32",
  "kernel:metal:navatala_graph_compute_degree_matrix_f32",
  "abi-r1:metal:navatala_graph_compute_degree_matrix_f32",
  "abi-r1:metal:navatala_graph_compute_degree_matrix_f32",
  3,
  kAbiArgs_metal_navatala_graph_compute_degree_matrix_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_compute_laplacian_f32[] = {
  { "W", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "D", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "L", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_compute_laplacian_f32 = {
  1,
  "navatala_graph_compute_laplacian_f32",
  "metal",
  "navatala_graph_compute_laplacian_f32",
  "kernel:metal:navatala_graph_compute_laplacian_f32",
  "abi-r1:metal:navatala_graph_compute_laplacian_f32",
  "abi-r1:metal:navatala_graph_compute_laplacian_f32",
  4,
  kAbiArgs_metal_navatala_graph_compute_laplacian_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_normalized_laplacian_f32[] = {
  { "W", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "D", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "L_sym", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_normalized_laplacian_f32 = {
  1,
  "navatala_graph_normalized_laplacian_f32",
  "metal",
  "navatala_graph_normalized_laplacian_f32",
  "kernel:metal:navatala_graph_normalized_laplacian_f32",
  "abi-r1:metal:navatala_graph_normalized_laplacian_f32",
  "abi-r1:metal:navatala_graph_normalized_laplacian_f32",
  4,
  kAbiArgs_metal_navatala_graph_normalized_laplacian_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_normalize_eigenvector_rows_f32[] = {
  { "V", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "V_norm", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_normalize_eigenvector_rows_f32 = {
  1,
  "navatala_graph_normalize_eigenvector_rows_f32",
  "metal",
  "navatala_graph_normalize_eigenvector_rows_f32",
  "kernel:metal:navatala_graph_normalize_eigenvector_rows_f32",
  "abi-r1:metal:navatala_graph_normalize_eigenvector_rows_f32",
  "abi-r1:metal:navatala_graph_normalize_eigenvector_rows_f32",
  4,
  kAbiArgs_metal_navatala_graph_normalize_eigenvector_rows_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_spectral_embedding_f32[] = {
  { "eigenvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "embedding", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_spectral_embedding_f32 = {
  1,
  "navatala_graph_spectral_embedding_f32",
  "metal",
  "navatala_graph_spectral_embedding_f32",
  "kernel:metal:navatala_graph_spectral_embedding_f32",
  "abi-r1:metal:navatala_graph_spectral_embedding_f32",
  "abi-r1:metal:navatala_graph_spectral_embedding_f32",
  4,
  kAbiArgs_metal_navatala_graph_spectral_embedding_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_rmat_generator_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_rmat_generator_f32 = {
  1,
  "navatala_graph_rmat_generator_f32",
  "metal",
  "navatala_graph_rmat_generator_f32",
  "kernel:metal:navatala_graph_rmat_generator_f32",
  "abi-r1:metal:navatala_graph_rmat_generator_f32",
  "abi-r1:metal:navatala_graph_rmat_generator_f32",
  13,
  kAbiArgs_metal_navatala_graph_rmat_generator_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_erdos_renyi_f32[] = {
  { "random", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "randomWeights", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "p", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcNodes", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dstNodes", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeCount", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_erdos_renyi_f32 = {
  1,
  "navatala_graph_erdos_renyi_f32",
  "metal",
  "navatala_graph_erdos_renyi_f32",
  "kernel:metal:navatala_graph_erdos_renyi_f32",
  "abi-r1:metal:navatala_graph_erdos_renyi_f32",
  "abi-r1:metal:navatala_graph_erdos_renyi_f32",
  8,
  kAbiArgs_metal_navatala_graph_erdos_renyi_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_barabasi_albert_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_barabasi_albert_f32 = {
  1,
  "navatala_graph_barabasi_albert_f32",
  "metal",
  "navatala_graph_barabasi_albert_f32",
  "kernel:metal:navatala_graph_barabasi_albert_f32",
  "abi-r1:metal:navatala_graph_barabasi_albert_f32",
  "abi-r1:metal:navatala_graph_barabasi_albert_f32",
  11,
  kAbiArgs_metal_navatala_graph_barabasi_albert_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_compute_modularity_f32[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusters", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClusters", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalEdgeWeight", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "modularity", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_compute_modularity_f32 = {
  1,
  "navatala_graph_compute_modularity_f32",
  "metal",
  "navatala_graph_compute_modularity_f32",
  "kernel:metal:navatala_graph_compute_modularity_f32",
  "abi-r1:metal:navatala_graph_compute_modularity_f32",
  "abi-r1:metal:navatala_graph_compute_modularity_f32",
  8,
  kAbiArgs_metal_navatala_graph_compute_modularity_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_modularity_matrix_vec_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_modularity_matrix_vec_f32 = {
  1,
  "navatala_graph_modularity_matrix_vec_f32",
  "metal",
  "navatala_graph_modularity_matrix_vec_f32",
  "kernel:metal:navatala_graph_modularity_matrix_vec_f32",
  "abi-r1:metal:navatala_graph_modularity_matrix_vec_f32",
  "abi-r1:metal:navatala_graph_modularity_matrix_vec_f32",
  9,
  kAbiArgs_metal_navatala_graph_modularity_matrix_vec_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_construct_indicator_f32[] = {
  { "clusters", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusterIndex", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indicator", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_construct_indicator_f32 = {
  1,
  "navatala_graph_construct_indicator_f32",
  "metal",
  "navatala_graph_construct_indicator_f32",
  "kernel:metal:navatala_graph_construct_indicator_f32",
  "abi-r1:metal:navatala_graph_construct_indicator_f32",
  "abi-r1:metal:navatala_graph_construct_indicator_f32",
  4,
  kAbiArgs_metal_navatala_graph_construct_indicator_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_compute_edge_cut_f32[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusters", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "edgeCut", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_compute_edge_cut_f32 = {
  1,
  "navatala_graph_compute_edge_cut_f32",
  "metal",
  "navatala_graph_compute_edge_cut_f32",
  "kernel:metal:navatala_graph_compute_edge_cut_f32",
  "abi-r1:metal:navatala_graph_compute_edge_cut_f32",
  "abi-r1:metal:navatala_graph_compute_edge_cut_f32",
  6,
  kAbiArgs_metal_navatala_graph_compute_edge_cut_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_compute_partition_cost_f32[] = {
  { "edgeCutPerCluster", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusterSizes", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numClusters", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cost", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_compute_partition_cost_f32 = {
  1,
  "navatala_graph_compute_partition_cost_f32",
  "metal",
  "navatala_graph_compute_partition_cost_f32",
  "kernel:metal:navatala_graph_compute_partition_cost_f32",
  "abi-r1:metal:navatala_graph_compute_partition_cost_f32",
  "abi-r1:metal:navatala_graph_compute_partition_cost_f32",
  4,
  kAbiArgs_metal_navatala_graph_compute_partition_cost_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_analyze_partition_f32[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusters", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClusters", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "edgeCut", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cost", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_analyze_partition_f32 = {
  1,
  "navatala_graph_analyze_partition_f32",
  "metal",
  "navatala_graph_analyze_partition_f32",
  "kernel:metal:navatala_graph_analyze_partition_f32",
  "abi-r1:metal:navatala_graph_analyze_partition_f32",
  "abi-r1:metal:navatala_graph_analyze_partition_f32",
  8,
  kAbiArgs_metal_navatala_graph_analyze_partition_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_transform_eigen_matrix_f32[] = {
  { "eigenvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "transformed", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_transform_eigen_matrix_f32 = {
  1,
  "navatala_graph_transform_eigen_matrix_f32",
  "metal",
  "navatala_graph_transform_eigen_matrix_f32",
  "kernel:metal:navatala_graph_transform_eigen_matrix_f32",
  "abi-r1:metal:navatala_graph_transform_eigen_matrix_f32",
  "abi-r1:metal:navatala_graph_transform_eigen_matrix_f32",
  4,
  kAbiArgs_metal_navatala_graph_transform_eigen_matrix_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_graph_laplacian_from_adjacency_f32[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "laplacianValues", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degree", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_graph_laplacian_from_adjacency_f32 = {
  1,
  "navatala_graph_laplacian_from_adjacency_f32",
  "metal",
  "navatala_graph_laplacian_from_adjacency_f32",
  "kernel:metal:navatala_graph_laplacian_from_adjacency_f32",
  "abi-r1:metal:navatala_graph_laplacian_from_adjacency_f32",
  "abi-r1:metal:navatala_graph_laplacian_from_adjacency_f32",
  6,
  kAbiArgs_metal_navatala_graph_laplacian_from_adjacency_f32
};

bool tryGetKernelAbiManifest_metal_graph(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_graph_pack_f32") {
    out = &kAbiManifest_metal_navatala_graph_pack_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_pack_i32") {
    out = &kAbiManifest_metal_navatala_graph_pack_i32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_pack_u32") {
    out = &kAbiManifest_metal_navatala_graph_pack_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_unpack_f32") {
    out = &kAbiManifest_metal_navatala_graph_unpack_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_unpack_i32") {
    out = &kAbiManifest_metal_navatala_graph_unpack_i32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_unpack_u32") {
    out = &kAbiManifest_metal_navatala_graph_unpack_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_pack_f32x3") {
    out = &kAbiManifest_metal_navatala_graph_pack_f32x3;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_unpack_f32x3") {
    out = &kAbiManifest_metal_navatala_graph_unpack_f32x3;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_pack_f32x6") {
    out = &kAbiManifest_metal_navatala_graph_pack_f32x6;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_unpack_f32x6") {
    out = &kAbiManifest_metal_navatala_graph_unpack_f32x6;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_pack_f32x9") {
    out = &kAbiManifest_metal_navatala_graph_pack_f32x9;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_unpack_f32x9") {
    out = &kAbiManifest_metal_navatala_graph_unpack_f32x9;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_degree_out_u32") {
    out = &kAbiManifest_metal_navatala_graph_degree_out_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_degree_in_u32") {
    out = &kAbiManifest_metal_navatala_graph_degree_in_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_spmv_weighted_f32") {
    out = &kAbiManifest_metal_navatala_graph_spmv_weighted_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_spmv_weighted_subgroup_f32") {
    out = &kAbiManifest_metal_navatala_graph_spmv_weighted_subgroup_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_spmv_unweighted_f32") {
    out = &kAbiManifest_metal_navatala_graph_spmv_unweighted_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_spmv_unweighted_subgroup_f32") {
    out = &kAbiManifest_metal_navatala_graph_spmv_unweighted_subgroup_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_axpy2_f32") {
    out = &kAbiManifest_metal_navatala_graph_axpy2_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_scale_f32") {
    out = &kAbiManifest_metal_navatala_graph_scale_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_scale_add_bias_f32") {
    out = &kAbiManifest_metal_navatala_graph_scale_add_bias_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_l1norm_f32") {
    out = &kAbiManifest_metal_navatala_graph_l1norm_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_recip_f32") {
    out = &kAbiManifest_metal_navatala_graph_recip_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_l1_diff_f32") {
    out = &kAbiManifest_metal_navatala_graph_l1_diff_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_count_diff_u32") {
    out = &kAbiManifest_metal_navatala_graph_count_diff_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_copy_f32") {
    out = &kAbiManifest_metal_navatala_graph_copy_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_copy_u32") {
    out = &kAbiManifest_metal_navatala_graph_copy_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_relax_vertex_pull_f32") {
    out = &kAbiManifest_metal_navatala_graph_relax_vertex_pull_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_label_propagate_edges_u32") {
    out = &kAbiManifest_metal_navatala_graph_label_propagate_edges_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_symmetrize_reverse_u32") {
    out = &kAbiManifest_metal_navatala_graph_symmetrize_reverse_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_apply_renumber_u32") {
    out = &kAbiManifest_metal_navatala_graph_apply_renumber_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_peel_below_threshold_u32") {
    out = &kAbiManifest_metal_navatala_graph_peel_below_threshold_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_pair_intersection_u32") {
    out = &kAbiManifest_metal_navatala_graph_pair_intersection_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_jaccard_f32") {
    out = &kAbiManifest_metal_navatala_graph_jaccard_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_overlap_f32") {
    out = &kAbiManifest_metal_navatala_graph_overlap_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_sorensen_f32") {
    out = &kAbiManifest_metal_navatala_graph_sorensen_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_cosine_f32") {
    out = &kAbiManifest_metal_navatala_graph_cosine_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_uniform_random_walk_u32") {
    out = &kAbiManifest_metal_navatala_graph_uniform_random_walk_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_out_degree_u32") {
    out = &kAbiManifest_metal_navatala_graph_out_degree_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_in_degree_u32") {
    out = &kAbiManifest_metal_navatala_graph_in_degree_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_rbf_affinity_f32") {
    out = &kAbiManifest_metal_navatala_graph_rbf_affinity_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_compute_degree_matrix_f32") {
    out = &kAbiManifest_metal_navatala_graph_compute_degree_matrix_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_compute_laplacian_f32") {
    out = &kAbiManifest_metal_navatala_graph_compute_laplacian_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_normalized_laplacian_f32") {
    out = &kAbiManifest_metal_navatala_graph_normalized_laplacian_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_normalize_eigenvector_rows_f32") {
    out = &kAbiManifest_metal_navatala_graph_normalize_eigenvector_rows_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_spectral_embedding_f32") {
    out = &kAbiManifest_metal_navatala_graph_spectral_embedding_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_rmat_generator_f32") {
    out = &kAbiManifest_metal_navatala_graph_rmat_generator_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_erdos_renyi_f32") {
    out = &kAbiManifest_metal_navatala_graph_erdos_renyi_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_barabasi_albert_f32") {
    out = &kAbiManifest_metal_navatala_graph_barabasi_albert_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_compute_modularity_f32") {
    out = &kAbiManifest_metal_navatala_graph_compute_modularity_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_modularity_matrix_vec_f32") {
    out = &kAbiManifest_metal_navatala_graph_modularity_matrix_vec_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_construct_indicator_f32") {
    out = &kAbiManifest_metal_navatala_graph_construct_indicator_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_compute_edge_cut_f32") {
    out = &kAbiManifest_metal_navatala_graph_compute_edge_cut_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_compute_partition_cost_f32") {
    out = &kAbiManifest_metal_navatala_graph_compute_partition_cost_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_analyze_partition_f32") {
    out = &kAbiManifest_metal_navatala_graph_analyze_partition_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_transform_eigen_matrix_f32") {
    out = &kAbiManifest_metal_navatala_graph_transform_eigen_matrix_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_laplacian_from_adjacency_f32") {
    out = &kAbiManifest_metal_navatala_graph_laplacian_from_adjacency_f32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_graph(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_graph_pack_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_pack_f32";
    std::string_view sv(k_metal_navatala_graph_pack_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_pack_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_pack_i32";
    std::string_view sv(k_metal_navatala_graph_pack_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_pack_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_pack_u32";
    std::string_view sv(k_metal_navatala_graph_pack_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_unpack_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_unpack_f32";
    std::string_view sv(k_metal_navatala_graph_unpack_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_unpack_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_unpack_i32";
    std::string_view sv(k_metal_navatala_graph_unpack_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_unpack_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_unpack_u32";
    std::string_view sv(k_metal_navatala_graph_unpack_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_pack_f32x3") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_pack_f32x3";
    std::string_view sv(k_metal_navatala_graph_pack_f32x3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_unpack_f32x3") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_unpack_f32x3";
    std::string_view sv(k_metal_navatala_graph_unpack_f32x3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_pack_f32x6") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_pack_f32x6";
    std::string_view sv(k_metal_navatala_graph_pack_f32x6);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_unpack_f32x6") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_unpack_f32x6";
    std::string_view sv(k_metal_navatala_graph_unpack_f32x6);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_pack_f32x9") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_pack_f32x9";
    std::string_view sv(k_metal_navatala_graph_pack_f32x9);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_unpack_f32x9") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_unpack_f32x9";
    std::string_view sv(k_metal_navatala_graph_unpack_f32x9);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_degree_out_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_degree_out_u32";
    std::string_view sv(k_metal_navatala_graph_degree_out_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_degree_in_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_degree_in_u32";
    std::string_view sv(k_metal_navatala_graph_degree_in_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_spmv_weighted_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_spmv_weighted_f32";
    std::string_view sv(k_metal_navatala_graph_spmv_weighted_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_spmv_weighted_subgroup_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_spmv_weighted_subgroup_f32";
    std::string_view sv(k_metal_navatala_graph_spmv_weighted_subgroup_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_spmv_unweighted_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_spmv_unweighted_f32";
    std::string_view sv(k_metal_navatala_graph_spmv_unweighted_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_spmv_unweighted_subgroup_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_spmv_unweighted_subgroup_f32";
    std::string_view sv(k_metal_navatala_graph_spmv_unweighted_subgroup_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_axpy2_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_axpy2_f32";
    std::string_view sv(k_metal_navatala_graph_axpy2_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_scale_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_scale_f32";
    std::string_view sv(k_metal_navatala_graph_scale_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_scale_add_bias_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_scale_add_bias_f32";
    std::string_view sv(k_metal_navatala_graph_scale_add_bias_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_l1norm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_l1norm_f32";
    std::string_view sv(k_metal_navatala_graph_l1norm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_recip_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_recip_f32";
    std::string_view sv(k_metal_navatala_graph_recip_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_l1_diff_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_l1_diff_f32";
    std::string_view sv(k_metal_navatala_graph_l1_diff_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_count_diff_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_count_diff_u32";
    std::string_view sv(k_metal_navatala_graph_count_diff_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_copy_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_copy_f32";
    std::string_view sv(k_metal_navatala_graph_copy_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_copy_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_copy_u32";
    std::string_view sv(k_metal_navatala_graph_copy_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_relax_vertex_pull_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_relax_vertex_pull_f32";
    std::string_view sv(k_metal_navatala_graph_relax_vertex_pull_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_label_propagate_edges_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_label_propagate_edges_u32";
    std::string_view sv(k_metal_navatala_graph_label_propagate_edges_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_symmetrize_reverse_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_symmetrize_reverse_u32";
    std::string_view sv(k_metal_navatala_graph_symmetrize_reverse_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_apply_renumber_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_apply_renumber_u32";
    std::string_view sv(k_metal_navatala_graph_apply_renumber_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_peel_below_threshold_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_peel_below_threshold_u32";
    std::string_view sv(k_metal_navatala_graph_peel_below_threshold_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_pair_intersection_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_pair_intersection_u32";
    std::string_view sv(k_metal_navatala_graph_pair_intersection_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_jaccard_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_jaccard_f32";
    std::string_view sv(k_metal_navatala_graph_jaccard_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_overlap_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_overlap_f32";
    std::string_view sv(k_metal_navatala_graph_overlap_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_sorensen_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_sorensen_f32";
    std::string_view sv(k_metal_navatala_graph_sorensen_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_cosine_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_cosine_f32";
    std::string_view sv(k_metal_navatala_graph_cosine_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_uniform_random_walk_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_uniform_random_walk_u32";
    std::string_view sv(k_metal_navatala_graph_uniform_random_walk_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_out_degree_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_out_degree_u32";
    std::string_view sv(k_metal_navatala_graph_out_degree_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_in_degree_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_in_degree_u32";
    std::string_view sv(k_metal_navatala_graph_in_degree_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_rbf_affinity_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_rbf_affinity_f32";
    std::string_view sv(k_metal_navatala_graph_rbf_affinity_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_compute_degree_matrix_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_compute_degree_matrix_f32";
    std::string_view sv(k_metal_navatala_graph_compute_degree_matrix_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_compute_laplacian_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_compute_laplacian_f32";
    std::string_view sv(k_metal_navatala_graph_compute_laplacian_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_normalized_laplacian_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_normalized_laplacian_f32";
    std::string_view sv(k_metal_navatala_graph_normalized_laplacian_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_normalize_eigenvector_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_normalize_eigenvector_rows_f32";
    std::string_view sv(k_metal_navatala_graph_normalize_eigenvector_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_spectral_embedding_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_spectral_embedding_f32";
    std::string_view sv(k_metal_navatala_graph_spectral_embedding_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_rmat_generator_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_rmat_generator_f32";
    std::string_view sv(k_metal_navatala_graph_rmat_generator_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_erdos_renyi_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_erdos_renyi_f32";
    std::string_view sv(k_metal_navatala_graph_erdos_renyi_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_barabasi_albert_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_barabasi_albert_f32";
    std::string_view sv(k_metal_navatala_graph_barabasi_albert_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_compute_modularity_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_compute_modularity_f32";
    std::string_view sv(k_metal_navatala_graph_compute_modularity_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_modularity_matrix_vec_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_modularity_matrix_vec_f32";
    std::string_view sv(k_metal_navatala_graph_modularity_matrix_vec_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_construct_indicator_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_construct_indicator_f32";
    std::string_view sv(k_metal_navatala_graph_construct_indicator_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_compute_edge_cut_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_compute_edge_cut_f32";
    std::string_view sv(k_metal_navatala_graph_compute_edge_cut_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_compute_partition_cost_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_compute_partition_cost_f32";
    std::string_view sv(k_metal_navatala_graph_compute_partition_cost_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_analyze_partition_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_analyze_partition_f32";
    std::string_view sv(k_metal_navatala_graph_analyze_partition_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_transform_eigen_matrix_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_transform_eigen_matrix_f32";
    std::string_view sv(k_metal_navatala_graph_transform_eigen_matrix_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_graph_laplacian_from_adjacency_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_graph_laplacian_from_adjacency_f32";
    std::string_view sv(k_metal_navatala_graph_laplacian_from_adjacency_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

