// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `opencl_graph` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_opencl_navatala_graph_pack_f32 = R"kernel(
__kernel void navatala_graph_pack_f32(__global const int* indices, __global const float* src, __global const int* count, __global float* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    dst[i] = src[indices[((int)((int)(get_global_id(0))))]];
  }
}

)kernel";
const char* k_opencl_navatala_graph_pack_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_pack_f64(__global const int* indices, __global const double* src, __global const int* count, __global double* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    dst[i] = src[indices[((int)((int)(get_global_id(0))))]];
  }
}

)kernel";
const char* k_opencl_navatala_graph_pack_i32 = R"kernel(
__kernel void navatala_graph_pack_i32(__global const int* indices, __global const int* src, __global const int* count, __global int* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    dst[i] = src[indices[((int)((int)(get_global_id(0))))]];
  }
}

)kernel";
const char* k_opencl_navatala_graph_pack_u32 = R"kernel(
__kernel void navatala_graph_pack_u32(__global const int* indices, __global const uint* src, __global const int* count, __global uint* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    dst[i] = src[indices[((int)((int)(get_global_id(0))))]];
  }
}

)kernel";
const char* k_opencl_navatala_graph_unpack_f32 = R"kernel(
__kernel void navatala_graph_unpack_f32(__global const int* indices, __global const float* src, __global const int* count, __global float* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    dst[indices[((int)((int)(get_global_id(0))))]] = src[((int)((int)(get_global_id(0))))];
  }
}

)kernel";
const char* k_opencl_navatala_graph_unpack_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_unpack_f64(__global const int* indices, __global const double* src, __global const int* count, __global double* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    dst[indices[((int)((int)(get_global_id(0))))]] = src[((int)((int)(get_global_id(0))))];
  }
}

)kernel";
const char* k_opencl_navatala_graph_unpack_i32 = R"kernel(
__kernel void navatala_graph_unpack_i32(__global const int* indices, __global const int* src, __global const int* count, __global int* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    dst[indices[((int)((int)(get_global_id(0))))]] = src[((int)((int)(get_global_id(0))))];
  }
}

)kernel";
const char* k_opencl_navatala_graph_unpack_u32 = R"kernel(
__kernel void navatala_graph_unpack_u32(__global const int* indices, __global const uint* src, __global const int* count, __global uint* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    dst[indices[((int)((int)(get_global_id(0))))]] = src[((int)((int)(get_global_id(0))))];
  }
}

)kernel";
const char* k_opencl_navatala_graph_pack_f32x3 = R"kernel(
__kernel void navatala_graph_pack_f32x3(__global const int* indices, __global const float* src, __global const int* count, __global const int* n, __global float* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    for (int c = 0; c < (int)(3); ++c) {
      int dstLin = ((c * count[0]) + i);
      int srcLin = ((c * n[0]) + indices[((int)((int)(get_global_id(0))))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_unpack_f32x3 = R"kernel(
__kernel void navatala_graph_unpack_f32x3(__global const int* indices, __global const float* src, __global const int* count, __global const int* n, __global float* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    for (int c = 0; c < (int)(3); ++c) {
      int srcLin = ((c * count[0]) + i);
      int dstLin = ((c * n[0]) + indices[((int)((int)(get_global_id(0))))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_pack_f32x6 = R"kernel(
__kernel void navatala_graph_pack_f32x6(__global const int* indices, __global const float* src, __global const int* count, __global const int* n, __global float* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    for (int c = 0; c < (int)(6); ++c) {
      int dstLin = ((c * count[0]) + i);
      int srcLin = ((c * n[0]) + indices[((int)((int)(get_global_id(0))))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_unpack_f32x6 = R"kernel(
__kernel void navatala_graph_unpack_f32x6(__global const int* indices, __global const float* src, __global const int* count, __global const int* n, __global float* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    for (int c = 0; c < (int)(6); ++c) {
      int srcLin = ((c * count[0]) + i);
      int dstLin = ((c * n[0]) + indices[((int)((int)(get_global_id(0))))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_pack_f32x9 = R"kernel(
__kernel void navatala_graph_pack_f32x9(__global const int* indices, __global const float* src, __global const int* count, __global const int* n, __global float* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    for (int c = 0; c < (int)(9); ++c) {
      int dstLin = ((c * count[0]) + i);
      int srcLin = ((c * n[0]) + indices[((int)((int)(get_global_id(0))))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_unpack_f32x9 = R"kernel(
__kernel void navatala_graph_unpack_f32x9(__global const int* indices, __global const float* src, __global const int* count, __global const int* n, __global float* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    for (int c = 0; c < (int)(9); ++c) {
      int srcLin = ((c * count[0]) + i);
      int dstLin = ((c * n[0]) + indices[((int)((int)(get_global_id(0))))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_pack_f64x3 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_pack_f64x3(__global const int* indices, __global const double* src, __global const int* count, __global const int* n, __global double* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    for (int c = 0; c < (int)(3); ++c) {
      int dstLin = ((c * count[0]) + i);
      int srcLin = ((c * n[0]) + indices[((int)((int)(get_global_id(0))))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_unpack_f64x3 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_unpack_f64x3(__global const int* indices, __global const double* src, __global const int* count, __global const int* n, __global double* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    for (int c = 0; c < (int)(3); ++c) {
      int srcLin = ((c * count[0]) + i);
      int dstLin = ((c * n[0]) + indices[((int)((int)(get_global_id(0))))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_pack_f64x6 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_pack_f64x6(__global const int* indices, __global const double* src, __global const int* count, __global const int* n, __global double* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    for (int c = 0; c < (int)(6); ++c) {
      int dstLin = ((c * count[0]) + i);
      int srcLin = ((c * n[0]) + indices[((int)((int)(get_global_id(0))))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_unpack_f64x6 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_unpack_f64x6(__global const int* indices, __global const double* src, __global const int* count, __global const int* n, __global double* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    for (int c = 0; c < (int)(6); ++c) {
      int srcLin = ((c * count[0]) + i);
      int dstLin = ((c * n[0]) + indices[((int)((int)(get_global_id(0))))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_pack_f64x9 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_pack_f64x9(__global const int* indices, __global const double* src, __global const int* count, __global const int* n, __global double* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    for (int c = 0; c < (int)(9); ++c) {
      int dstLin = ((c * count[0]) + i);
      int srcLin = ((c * n[0]) + indices[((int)((int)(get_global_id(0))))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_unpack_f64x9 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_unpack_f64x9(__global const int* indices, __global const double* src, __global const int* count, __global const int* n, __global double* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if ((i < count[0])) {
    for (int c = 0; c < (int)(9); ++c) {
      int srcLin = ((c * count[0]) + i);
      int dstLin = ((c * n[0]) + indices[((int)((int)(get_global_id(0))))]);
      dst[dstLin] = src[srcLin];
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_degree_out_u32 = R"kernel(
__kernel void navatala_graph_degree_out_u32(__global const uint* offsets, __global const uint* numVertices, __global uint* degrees) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numV = numVertices[0];
  if ((gid < numV)) {
    uint hi = offsets[(gid + (uint)(1u))];
    uint lo = offsets[gid];
    uint deg = (hi - lo);
    degrees[gid] = deg;
  }
}

)kernel";
const char* k_opencl_navatala_graph_degree_in_u32 = R"kernel(
__kernel void navatala_graph_degree_in_u32(__global const uint* offsets, __global const uint* numVertices, __global uint* degrees) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numV = numVertices[0];
  if ((gid < numV)) {
    uint hi = offsets[(gid + (uint)(1u))];
    uint lo = offsets[gid];
    uint deg = (hi - lo);
    degrees[gid] = deg;
  }
}

)kernel";
const char* k_opencl_navatala_graph_spmv_weighted_f32 = R"kernel(
__kernel void navatala_graph_spmv_weighted_f32(__global const uint* offsets, __global const uint* indices, __global const float* weights, __global const float* x, __global const uint* numVertices, __global float* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numV = numVertices[0];
  if ((gid < numV)) {
    uint base = offsets[gid];
    uint endv = offsets[(gid + (uint)(1u))];
    uint rowlen = (endv - base);
    float acc = as_float(0x00000000u);
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
const char* k_opencl_navatala_graph_spmv_weighted_subgroup_f32 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
__kernel void navatala_graph_spmv_weighted_subgroup_f32(__global const uint* offsets, __global const uint* indices, __global const float* weights, __global const float* x, __global const uint* numVertices, __global float* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lane = ((uint)((int)(get_sub_group_local_id())));
  uint subgroupSize = ((uint)((int)(get_sub_group_size())));
  uint row = (gid / subgroupSize);
  uint numV = numVertices[0];
  if ((row < numV)) {
    uint base = offsets[row];
    uint endv = offsets[(row + (uint)(1u))];
    uint rowlen = (endv - base);
    uint rowIters = ((rowlen + (subgroupSize - (uint)(1u))) / subgroupSize);
    float laneAcc = as_float(0x00000000u);
    for (int k = 0; k < (int)(rowIters); ++k) {
      uint rel = ((((uint)(k)) * subgroupSize) + lane);
      if ((rel < rowlen)) {
        uint eidx = (base + rel);
        uint col = indices[eidx];
        float w = weights[eidx];
        float xv = x[col];
        laneAcc = (laneAcc + (w * xv));
      }
    }
    float rowSum = sub_group_reduce_add(laneAcc);
    if ((lane == (uint)(0u))) {
      y[row] = rowSum;
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_spmv_unweighted_f32 = R"kernel(
__kernel void navatala_graph_spmv_unweighted_f32(__global const uint* offsets, __global const uint* indices, __global const float* x, __global const uint* numVertices, __global float* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numV = numVertices[0];
  if ((gid < numV)) {
    uint base = offsets[gid];
    uint endv = offsets[(gid + (uint)(1u))];
    uint rowlen = (endv - base);
    float acc = as_float(0x00000000u);
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
const char* k_opencl_navatala_graph_spmv_unweighted_subgroup_f32 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
__kernel void navatala_graph_spmv_unweighted_subgroup_f32(__global const uint* offsets, __global const uint* indices, __global const float* x, __global const uint* numVertices, __global float* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lane = ((uint)((int)(get_sub_group_local_id())));
  uint subgroupSize = ((uint)((int)(get_sub_group_size())));
  uint row = (gid / subgroupSize);
  uint numV = numVertices[0];
  if ((row < numV)) {
    uint base = offsets[row];
    uint endv = offsets[(row + (uint)(1u))];
    uint rowlen = (endv - base);
    uint rowIters = ((rowlen + (subgroupSize - (uint)(1u))) / subgroupSize);
    float laneAcc = as_float(0x00000000u);
    for (int k = 0; k < (int)(rowIters); ++k) {
      uint rel = ((((uint)(k)) * subgroupSize) + lane);
      if ((rel < rowlen)) {
        uint eidx = (base + rel);
        uint col = indices[eidx];
        float xv = x[col];
        laneAcc = (laneAcc + xv);
      }
    }
    float rowSum = sub_group_reduce_add(laneAcc);
    if ((lane == (uint)(0u))) {
      y[row] = rowSum;
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_axpy2_f32 = R"kernel(
__kernel void navatala_graph_axpy2_f32(__global const float* x, __global const float* z, __global const float* alpha, __global const float* beta, __global const uint* count, __global float* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if ((gid < countVal)) {
    float alphaV = alpha[0];
    float betaV = beta[0];
    float xv = x[gid];
    float zv = z[gid];
    float yv = ((alphaV * xv) + (betaV * zv));
    y[gid] = yv;
  }
}

)kernel";
const char* k_opencl_navatala_graph_scale_f32 = R"kernel(
__kernel void navatala_graph_scale_f32(__global const float* x, __global const float* alpha, __global const uint* count, __global float* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if ((gid < countVal)) {
    float alphaV = alpha[0];
    float xv = x[gid];
    y[gid] = (alphaV * xv);
  }
}

)kernel";
const char* k_opencl_navatala_graph_scale_add_bias_f32 = R"kernel(
__kernel void navatala_graph_scale_add_bias_f32(__global const float* x, __global const float* alpha, __global const float* bias, __global const uint* count, __global float* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if ((gid < countVal)) {
    float alphaV = alpha[0];
    float biasV = bias[0];
    float xv = x[gid];
    y[gid] = ((alphaV * xv) + biasV);
  }
}

)kernel";
const char* k_opencl_navatala_graph_l1norm_f32 = R"kernel(
__kernel void navatala_graph_l1norm_f32(__global const float* _input, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      float val = fabs(_input[idx]);
      gsAcc = (gsAcc + val);
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    result[0] = sdata[0];
  }
}

)kernel";
const char* k_opencl_navatala_graph_recip_f32 = R"kernel(
__kernel void navatala_graph_recip_f32(__global const float* x, __global float* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  if ((gid == (uint)(0u))) {
    float xv = x[0];
    if ((xv == as_float(0x00000000u))) {
      y[0] = as_float(0x00000000u);
    } else {
      y[0] = (as_float(0x3f800000u) / xv);
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_l1_diff_f32 = R"kernel(
__kernel void navatala_graph_l1_diff_f32(__global const float* a, __global const float* b, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      float av = a[idx];
      float bv = b[idx];
      float d = fabs((av - bv));
      gsAcc = (gsAcc + d);
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    result[0] = sdata[0];
  }
}

)kernel";
const char* k_opencl_navatala_graph_count_diff_u32 = R"kernel(
__kernel void navatala_graph_count_diff_u32(__global const uint* a, __global const uint* b, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      uint av = a[idx];
      uint bv = b[idx];
      float diff = (((av == bv)) ? (as_float(0x00000000u)) : (as_float(0x3f800000u)));
      gsAcc = (gsAcc + diff);
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    result[0] = sdata[0];
  }
}

)kernel";
const char* k_opencl_navatala_graph_copy_f32 = R"kernel(
__kernel void navatala_graph_copy_f32(__global const float* x, __global const uint* count, __global float* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if ((gid < countVal)) {
    float xv = x[gid];
    y[gid] = xv;
  }
}

)kernel";
const char* k_opencl_navatala_graph_copy_u32 = R"kernel(
__kernel void navatala_graph_copy_u32(__global const uint* x, __global const uint* count, __global uint* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if ((gid < countVal)) {
    uint xv = x[gid];
    y[gid] = xv;
  }
}

)kernel";
const char* k_opencl_navatala_graph_relax_vertex_pull_f32 = R"kernel(
__kernel void navatala_graph_relax_vertex_pull_f32(__global const uint* inOffsets, __global const uint* inSrcs, __global const float* inWeights, __global float* dist, __global const uint* numVertices, __global const float* inf) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numV = numVertices[0];
  if ((gid < numV)) {
    uint base = inOffsets[gid];
    uint endv = inOffsets[(gid + (uint)(1u))];
    uint deg = (endv - base);
    float infV = inf[0];
    float best = dist[gid];
    for (int k = 0; k < (int)(deg); ++k) {
      uint eidx = (base + ((uint)(k)));
      uint u = inSrcs[eidx];
      float w = inWeights[eidx];
      float du = dist[u];
      if ((du < infV)) {
        float cand = (du + w);
        float cur = best;
        best = (((cand < cur)) ? (cand) : (cur));
      }
    }
    dist[gid] = best;
  }
}

)kernel";
const char* k_opencl_navatala_graph_symmetrize_reverse_u32 = R"kernel(
__kernel void navatala_graph_symmetrize_reverse_u32(__global const uint* srcs, __global const uint* dsts, __global const uint* numEdges, __global uint* outSrcs, __global uint* outDsts) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numE = numEdges[0];
  if ((gid < numE)) {
    uint u = srcs[gid];
    uint v = dsts[gid];
    outSrcs[gid] = v;
    outDsts[gid] = u;
  }
}

)kernel";
const char* k_opencl_navatala_graph_apply_renumber_u32 = R"kernel(
__kernel void navatala_graph_apply_renumber_u32(__global const uint* mapping, __global const uint* numVertices, __global uint* newIds) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numV = numVertices[0];
  if ((gid < numV)) {
    newIds[gid] = mapping[gid];
  }
}

)kernel";
const char* k_opencl_navatala_graph_peel_below_threshold_u32 = R"kernel(
__kernel void navatala_graph_peel_below_threshold_u32(__global uint* _active, __global const uint* degrees, __global const uint* threshold, __global const uint* numVertices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numV = numVertices[0];
  if ((gid < numV)) {
    uint deg = degrees[gid];
    uint k = threshold[0];
    if ((deg < k)) {
      _active[gid] = (uint)(0u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_pair_intersection_u32 = R"kernel(
__kernel void navatala_graph_pair_intersection_u32(__global const uint* offsets, __global const uint* indices, __global const uint* pairsA, __global const uint* pairsB, __global const uint* numPairs, __global uint* inter) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numP = numPairs[0];
  if ((gid < numP)) {
    uint a = pairsA[gid];
    uint b = pairsB[gid];
    uint baseU = offsets[a];
    uint endU = offsets[(a + (uint)(1u))];
    uint baseV = offsets[b];
    uint endV = offsets[(b + (uint)(1u))];
    uint degU = (endU - baseU);
    uint interAccum = (uint)(0u);
    uint loAccum = (uint)(0u);
    uint hiAccum = (uint)(0u);
    for (int k = 0; k < (int)(degU); ++k) {
      uint eidx = (baseU + ((uint)(k)));
      uint w = indices[eidx];
      loAccum = baseV;
      hiAccum = endV;
      for (int _bs = 0; _bs < (int)((uint)(32u)); ++_bs) {
        uint lo = loAccum;
        uint hi = hiAccum;
        if ((lo < hi)) {
          uint mid = ((lo + hi) / (uint)(2u));
          uint midval = indices[mid];
          bool goRight = (midval < w);
          uint newLo = ((goRight) ? ((mid + (uint)(1u))) : (lo));
          uint newHi = ((goRight) ? (hi) : (mid));
          loAccum = newLo;
          hiAccum = newHi;
        }
      }
      uint lb = loAccum;
      if ((lb < endV)) {
        uint cand = indices[lb];
        if ((cand == w)) {
          interAccum = (interAccum + (uint)(1u));
        }
      }
    }
    inter[gid] = interAccum;
  }
}

)kernel";
const char* k_opencl_navatala_graph_jaccard_f32 = R"kernel(
__kernel void navatala_graph_jaccard_f32(__global const uint* inter, __global const uint* degrees, __global const uint* pairsA, __global const uint* pairsB, __global const uint* numPairs, __global float* coeff) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numP = numPairs[0];
  if ((gid < numP)) {
    uint a = pairsA[gid];
    uint b = pairsB[gid];
    uint iU = inter[gid];
    uint dAU = degrees[a];
    uint dBU = degrees[b];
    float iF = ((float)(iU));
    float dAF = ((float)(dAU));
    float dBF = ((float)(dBU));
    coeff[gid] = (((((dAF + dBF) - iF) > as_float(0x00000000u))) ? ((iF / ((dAF + dBF) - iF))) : (as_float(0x00000000u)));
  }
}

)kernel";
const char* k_opencl_navatala_graph_overlap_f32 = R"kernel(
__kernel void navatala_graph_overlap_f32(__global const uint* inter, __global const uint* degrees, __global const uint* pairsA, __global const uint* pairsB, __global const uint* numPairs, __global float* coeff) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numP = numPairs[0];
  if ((gid < numP)) {
    uint a = pairsA[gid];
    uint b = pairsB[gid];
    uint iU = inter[gid];
    uint dAU = degrees[a];
    uint dBU = degrees[b];
    float iF = ((float)(iU));
    float dAF = ((float)(dAU));
    float dBF = ((float)(dBU));
    coeff[gid] = ((((((dAF < dBF)) ? (dAF) : (dBF)) > as_float(0x00000000u))) ? ((iF / (((dAF < dBF)) ? (dAF) : (dBF)))) : (as_float(0x00000000u)));
  }
}

)kernel";
const char* k_opencl_navatala_graph_sorensen_f32 = R"kernel(
__kernel void navatala_graph_sorensen_f32(__global const uint* inter, __global const uint* degrees, __global const uint* pairsA, __global const uint* pairsB, __global const uint* numPairs, __global float* coeff) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numP = numPairs[0];
  if ((gid < numP)) {
    uint a = pairsA[gid];
    uint b = pairsB[gid];
    uint iU = inter[gid];
    uint dAU = degrees[a];
    uint dBU = degrees[b];
    float iF = ((float)(iU));
    float dAF = ((float)(dAU));
    float dBF = ((float)(dBU));
    coeff[gid] = ((((dAF + dBF) > as_float(0x00000000u))) ? (((as_float(0x40000000u) * iF) / (dAF + dBF))) : (as_float(0x00000000u)));
  }
}

)kernel";
const char* k_opencl_navatala_graph_cosine_f32 = R"kernel(
__kernel void navatala_graph_cosine_f32(__global const uint* inter, __global const uint* degrees, __global const uint* pairsA, __global const uint* pairsB, __global const uint* numPairs, __global float* coeff) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numP = numPairs[0];
  if ((gid < numP)) {
    uint a = pairsA[gid];
    uint b = pairsB[gid];
    uint iU = inter[gid];
    uint dAU = degrees[a];
    uint dBU = degrees[b];
    float iF = ((float)(iU));
    float dAF = ((float)(dAU));
    float dBF = ((float)(dBU));
    coeff[gid] = (((sqrt((dAF * dBF)) > as_float(0x00000000u))) ? ((iF / sqrt((dAF * dBF)))) : (as_float(0x00000000u)));
  }
}

)kernel";
const char* k_opencl_navatala_graph_uniform_random_walk_u32 = R"kernel(
__kernel void navatala_graph_uniform_random_walk_u32(__global const uint* offsets, __global const uint* indices, __global const uint* startVertices, __global const uint* maxLen, __global const uint* numWalks, __global const uint* seed, __global uint* walks) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numW = numWalks[0];
  if ((gid < numW)) {
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
      uint off1 = offsets[(cur + (uint)(1u))];
      uint deg = (off1 - off);
      uint rng = rngAccum;
      uint rngNext = ((rng * (uint)(1103515245u)) + (uint)(12345u));
      rngAccum = rngNext;
      if ((deg > (uint)(0u))) {
        uint pick = (rngNext % deg);
        uint nbr = indices[(off + pick)];
        curAccum = nbr;
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_out_degree_u32 = R"kernel(
__kernel void navatala_graph_out_degree_u32(__global const uint* offsets, __global const uint* numVertices, __global uint* degrees) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numV = numVertices[0];
  if ((gid < numV)) {
    uint hi = offsets[(gid + (uint)(1u))];
    uint lo = offsets[gid];
    uint deg = (hi - lo);
    degrees[gid] = deg;
  }
}

)kernel";
const char* k_opencl_navatala_graph_in_degree_u32 = R"kernel(
__kernel void navatala_graph_in_degree_u32(__global const uint* offsets, __global const uint* numVertices, __global uint* degrees) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numV = numVertices[0];
  if ((gid < numV)) {
    uint hi = offsets[(gid + (uint)(1u))];
    uint lo = offsets[gid];
    uint deg = (hi - lo);
    degrees[gid] = deg;
  }
}

)kernel";
const char* k_opencl_navatala_graph_rbf_affinity_f32 = R"kernel(
__kernel void navatala_graph_rbf_affinity_f32(__global const float* data, __global const float* sigma, __global const uint* n, __global const uint* d, __global float* W) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
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
      W[gid] = as_float(0x00000000u);
    } else {
      float distSq = as_float(0x00000000u);
      uint rowI = (i * dVal);
      uint rowJ = (j * dVal);
      uint idxI = rowI;
      uint idxJ = rowJ;
      float xi = data[idxI];
      float xj = data[idxJ];
      float diff = (xi - xj);
      float diffSq = (diff * diff);
      float sigmaSq = (sigmaVal * sigmaVal);
      float twoSigmaSq = (as_float(0x40000000u) * sigmaSq);
      float exponent = ((as_float(0x00000000u) - diffSq) / twoSigmaSq);
      float affinity = exp(exponent);
      W[gid] = affinity;
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_rbf_affinity_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_rbf_affinity_f64(__global const double* data, __global const double* sigma, __global const uint* n, __global const uint* d, __global double* W) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint dVal = d[0];
  double sigmaVal = sigma[0];
  uint totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / nVal);
    uint j = (gid % nVal);
    bool isDiag = (i == j);
    if (isDiag) {
      W[gid] = as_double(0x0000000000000000ul);
    } else {
      double distSq = as_double(0x0000000000000000ul);
      uint rowI = (i * dVal);
      uint rowJ = (j * dVal);
      uint idxI = rowI;
      uint idxJ = rowJ;
      double xi = data[idxI];
      double xj = data[idxJ];
      double diff = (xi - xj);
      double diffSq = (diff * diff);
      double sigmaSq = (sigmaVal * sigmaVal);
      double twoSigmaSq = (as_double(0x4000000000000000ul) * sigmaSq);
      double exponent = ((as_double(0x0000000000000000ul) - diffSq) / twoSigmaSq);
      double affinity = exp(exponent);
      W[gid] = affinity;
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_compute_degree_matrix_f32 = R"kernel(
__kernel void navatala_graph_compute_degree_matrix_f32(__global const float* W, __global const uint* n, __global float* D) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint wgid = ((uint)((int)(get_group_id(0))));
  __local float sdata[256];
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
      sdata[lid] = as_float(0x00000000u);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint degF32RedStride = (uint)(128u);
    for (int degF32RedStep = 0; degF32RedStep < (int)(8); ++degF32RedStep) {
      uint degF32Stride = degF32RedStride;
      if ((lid < degF32Stride)) {
        float other = sdata[(lid + degF32Stride)];
        float mine = sdata[lid];
        float sum = (mine + other);
        sdata[lid] = sum;
      }
      uint degF32StrideToHalve = degF32RedStride;
      uint degF32NextStride = (degF32StrideToHalve >> (uint)(1u));
      degF32RedStride = degF32NextStride;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if ((lid == (uint)(0u))) {
      float degreeVal = sdata[0];
      D[rowIdx] = degreeVal;
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_compute_degree_matrix_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_compute_degree_matrix_f64(__global const double* W, __global const uint* n, __global double* D) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint wgid = ((uint)((int)(get_group_id(0))));
  __local double sdata[256];
  uint nVal = n[0];
  uint rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    uint rowBase = (rowIdx * nVal);
    bool colInBounds = (lid < nVal);
    uint elemIdx = (rowBase + lid);
    if (colInBounds) {
      double wVal = W[elemIdx];
      sdata[lid] = wVal;
    } else {
      sdata[lid] = as_double(0x0000000000000000ul);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint degF64RedStride = (uint)(128u);
    for (int degF64RedStep = 0; degF64RedStep < (int)(8); ++degF64RedStep) {
      uint degF64Stride = degF64RedStride;
      if ((lid < degF64Stride)) {
        double other = sdata[(lid + degF64Stride)];
        double mine = sdata[lid];
        double sum = (mine + other);
        sdata[lid] = sum;
      }
      uint degF64StrideToHalve = degF64RedStride;
      uint degF64NextStride = (degF64StrideToHalve >> (uint)(1u));
      degF64RedStride = degF64NextStride;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if ((lid == (uint)(0u))) {
      double degreeVal = sdata[0];
      D[rowIdx] = degreeVal;
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_compute_laplacian_f32 = R"kernel(
__kernel void navatala_graph_compute_laplacian_f32(__global const float* W, __global const float* D, __global const uint* n, __global float* L) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
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
      float negW = (as_float(0x00000000u) - wVal);
      L[gid] = negW;
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_compute_laplacian_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_compute_laplacian_f64(__global const double* W, __global const double* D, __global const uint* n, __global double* L) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / nVal);
    uint j = (gid % nVal);
    bool isDiag = (i == j);
    if (isDiag) {
      double dVal = D[i];
      L[gid] = dVal;
    } else {
      double wVal = W[gid];
      double negW = (as_double(0x0000000000000000ul) - wVal);
      L[gid] = negW;
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_normalized_laplacian_f32 = R"kernel(
__kernel void navatala_graph_normalized_laplacian_f32(__global const float* W, __global const float* D, __global const uint* n, __global float* L_sym) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / nVal);
    uint j = (gid % nVal);
    float di = D[i];
    float dj = D[j];
    float eps = as_float(0x2edbe6ffu);
    bool diZero = (di < eps);
    bool djZero = (dj < eps);
    bool isolated = (diZero || djZero);
    bool isDiag = (i == j);
    if (isolated) {
      if (isDiag) {
        L_sym[gid] = as_float(0x3f800000u);
      } else {
        L_sym[gid] = as_float(0x00000000u);
      }
    } else {
      float wVal = W[gid];
      float didjProd = (di * dj);
      float sqrtDiDj = sqrt(didjProd);
      float normalized = (wVal / sqrtDiDj);
      if (isDiag) {
        float lSymVal = (as_float(0x3f800000u) - normalized);
        L_sym[gid] = lSymVal;
      } else {
        float lSymVal = (as_float(0x00000000u) - normalized);
        L_sym[gid] = lSymVal;
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_normalized_laplacian_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_normalized_laplacian_f64(__global const double* W, __global const double* D, __global const uint* n, __global double* L_sym) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / nVal);
    uint j = (gid % nVal);
    double di = D[i];
    double dj = D[j];
    double eps = as_double(0x3cd203af9ee75616ul);
    bool diZero = (di < eps);
    bool djZero = (dj < eps);
    bool isolated = (diZero || djZero);
    bool isDiag = (i == j);
    if (isolated) {
      if (isDiag) {
        L_sym[gid] = as_double(0x3ff0000000000000ul);
      } else {
        L_sym[gid] = as_double(0x0000000000000000ul);
      }
    } else {
      double wVal = W[gid];
      double didjProd = (di * dj);
      double sqrtDiDj = sqrt(didjProd);
      double normalized = (wVal / sqrtDiDj);
      if (isDiag) {
        double lSymVal = (as_double(0x3ff0000000000000ul) - normalized);
        L_sym[gid] = lSymVal;
      } else {
        double lSymVal = (as_double(0x0000000000000000ul) - normalized);
        L_sym[gid] = lSymVal;
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_normalize_eigenvector_rows_f32 = R"kernel(
__kernel void navatala_graph_normalize_eigenvector_rows_f32(__global const float* V, __global const uint* n, __global const uint* k, __global float* V_norm) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint wgid = ((uint)((int)(get_group_id(0))));
  __local float sdata[256];
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
      sdata[lid] = as_float(0x00000000u);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint normF32RedStride = (uint)(128u);
    for (int normF32RedStep = 0; normF32RedStep < (int)(8); ++normF32RedStep) {
      uint normF32Stride = normF32RedStride;
      if ((lid < normF32Stride)) {
        float other = sdata[(lid + normF32Stride)];
        float mine = sdata[lid];
        float sum = (mine + other);
        sdata[lid] = sum;
      }
      uint normF32StrideToHalve = normF32RedStride;
      uint normF32NextStride = (normF32StrideToHalve >> (uint)(1u));
      normF32RedStride = normF32NextStride;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if ((lid == (uint)(0u))) {
      float sumSq = sdata[0];
      float normVal = sqrt(sumSq);
      sdata[0] = normVal;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (colInBounds) {
      float norm = sdata[0];
      float vVal2 = V[elemIdx];
      float eps = as_float(0x2edbe6ffu);
      bool normZero = (norm < eps);
      if (normZero) {
        V_norm[elemIdx] = as_float(0x00000000u);
      } else {
        float normalized = (vVal2 / norm);
        V_norm[elemIdx] = normalized;
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_normalize_eigenvector_rows_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_normalize_eigenvector_rows_f64(__global const double* V, __global const uint* n, __global const uint* k, __global double* V_norm) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint wgid = ((uint)((int)(get_group_id(0))));
  __local double sdata[256];
  uint nVal = n[0];
  uint kVal = k[0];
  uint rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    uint rowBase = (rowIdx * kVal);
    bool colInBounds = (lid < kVal);
    uint elemIdx = (rowBase + lid);
    if (colInBounds) {
      double vVal = V[elemIdx];
      double vSq = (vVal * vVal);
      sdata[lid] = vSq;
    } else {
      sdata[lid] = as_double(0x0000000000000000ul);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint normF64RedStride = (uint)(128u);
    for (int normF64RedStep = 0; normF64RedStep < (int)(8); ++normF64RedStep) {
      uint normF64Stride = normF64RedStride;
      if ((lid < normF64Stride)) {
        double other = sdata[(lid + normF64Stride)];
        double mine = sdata[lid];
        double sum = (mine + other);
        sdata[lid] = sum;
      }
      uint normF64StrideToHalve = normF64RedStride;
      uint normF64NextStride = (normF64StrideToHalve >> (uint)(1u));
      normF64RedStride = normF64NextStride;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if ((lid == (uint)(0u))) {
      double sumSq = sdata[0];
      double normVal = sqrt(sumSq);
      sdata[0] = normVal;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (colInBounds) {
      double norm = sdata[0];
      double vVal2 = V[elemIdx];
      double eps = as_double(0x3cd203af9ee75616ul);
      bool normZero = (norm < eps);
      if (normZero) {
        V_norm[elemIdx] = as_double(0x0000000000000000ul);
      } else {
        double normalized = (vVal2 / norm);
        V_norm[elemIdx] = normalized;
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_spectral_embedding_f32 = R"kernel(
__kernel void navatala_graph_spectral_embedding_f32(__global const float* eigenvectors, __global const uint* n, __global const uint* k, __global float* embedding) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
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
const char* k_opencl_navatala_graph_spectral_embedding_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_spectral_embedding_f64(__global const double* eigenvectors, __global const uint* n, __global const uint* k, __global double* embedding) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint kVal = k[0];
  uint totalElems = (nVal * kVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint row = (gid / kVal);
    uint col = (gid % kVal);
    uint srcIdx = ((col * nVal) + row);
    double eigVal = eigenvectors[srcIdx];
    embedding[gid] = eigVal;
  }
}

)kernel";
const char* k_opencl_navatala_graph_compute_modularity_f32 = R"kernel(
__kernel void navatala_graph_compute_modularity_f32(__global const float* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const uint* clusters, __global const uint* n, __global const uint* numClusters, __global const float* totalEdgeWeight, __global float* modularity) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  float totalWeight = totalEdgeWeight[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = clusters[gid];
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + (uint)(1u));
    uint rowEnd = rowPtr[gidPlus1];
    float degreeAccum = as_float(0x00000000u);
    float intraWeightAccum = as_float(0x00000000u);
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
        float weightToAdd = ((sameCluster) ? (edgeWeight) : (as_float(0x00000000u)));
        float currentIntraWeight = intraWeightAccum;
        float newIntraWeight = (currentIntraWeight + weightToAdd);
        intraWeightAccum = newIntraWeight;
      }
    }
    float degree = degreeAccum;
    float intraWeight = intraWeightAccum;
    float invTotalWeight = (as_float(0x3f800000u) / totalWeight);
    float eContrib = (intraWeight * invTotalWeight);
    float degreeNorm = (degree * invTotalWeight);
    float aContrib = (degreeNorm * degreeNorm);
    float contribution = (eContrib - aContrib);
    atomic_add(&modularity[0], contribution);
  }
}

)kernel";
const char* k_opencl_navatala_graph_compute_modularity_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_compute_modularity_f64(__global const double* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const uint* clusters, __global const uint* n, __global const uint* numClusters, __global const double* totalEdgeWeight, __global double* modularity) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  double totalWeight = totalEdgeWeight[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = clusters[gid];
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + (uint)(1u));
    uint rowEnd = rowPtr[gidPlus1];
    double degreeAccum = as_double(0x0000000000000000ul);
    double intraWeightAccum = as_double(0x0000000000000000ul);
    for (int idx = 0; idx < (int)(rowEnd); ++idx) {
      uint idxU32 = ((uint)(idx));
      bool idxInRange = (idxU32 >= rowStart);
      if (idxInRange) {
        double edgeWeight = adjacencyValues[idxU32];
        uint neighbor = colInd[idxU32];
        uint neighborCluster = clusters[neighbor];
        double currentDegree = degreeAccum;
        double newDegree = (currentDegree + edgeWeight);
        degreeAccum = newDegree;
        bool sameCluster = (neighborCluster == myCluster);
        double weightToAdd = ((sameCluster) ? (edgeWeight) : (as_double(0x0000000000000000ul)));
        double currentIntraWeight = intraWeightAccum;
        double newIntraWeight = (currentIntraWeight + weightToAdd);
        intraWeightAccum = newIntraWeight;
      }
    }
    double degree = degreeAccum;
    double intraWeight = intraWeightAccum;
    double invTotalWeight = (as_double(0x3ff0000000000000ul) / totalWeight);
    double eContrib = (intraWeight * invTotalWeight);
    double degreeNorm = (degree * invTotalWeight);
    double aContrib = (degreeNorm * degreeNorm);
    double contribution = (eContrib - aContrib);
    atomic_add(&modularity[0], contribution);
  }
}

)kernel";
const char* k_opencl_navatala_graph_modularity_matrix_vec_f32 = R"kernel(
__kernel void navatala_graph_modularity_matrix_vec_f32(__global const float* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const float* x, __global const float* degree, __global const float* totalEdgeWeight, __global const uint* n, __global const float* dotDX, __global float* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  float totalWeight = totalEdgeWeight[0];
  float dotDXVal = dotDX[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + (uint)(1u));
    uint rowEnd = rowPtr[gidPlus1];
    float axValAccum = as_float(0x00000000u);
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
const char* k_opencl_navatala_graph_modularity_matrix_vec_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_modularity_matrix_vec_f64(__global const double* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const double* x, __global const double* degree, __global const double* totalEdgeWeight, __global const uint* n, __global const double* dotDX, __global double* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  double totalWeight = totalEdgeWeight[0];
  double dotDXVal = dotDX[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + (uint)(1u));
    uint rowEnd = rowPtr[gidPlus1];
    double axValAccum = as_double(0x0000000000000000ul);
    for (int idx = 0; idx < (int)(rowEnd); ++idx) {
      uint idxU32 = ((uint)(idx));
      bool idxInRange = (idxU32 >= rowStart);
      if (idxInRange) {
        double aij = adjacencyValues[idxU32];
        uint j = colInd[idxU32];
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
const char* k_opencl_navatala_graph_construct_indicator_f32 = R"kernel(
__kernel void navatala_graph_construct_indicator_f32(__global const uint* clusters, __global const uint* clusterIndex, __global const uint* n, __global float* indicator) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint targetCluster = clusterIndex[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint nodeCluster = clusters[gid];
    bool isTarget = (nodeCluster == targetCluster);
    float indicatorVal = ((isTarget) ? (as_float(0x3f800000u)) : (as_float(0x00000000u)));
    indicator[gid] = indicatorVal;
  }
}

)kernel";
const char* k_opencl_navatala_graph_construct_indicator_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_construct_indicator_f64(__global const uint* clusters, __global const uint* clusterIndex, __global const uint* n, __global double* indicator) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint targetCluster = clusterIndex[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint nodeCluster = clusters[gid];
    bool isTarget = (nodeCluster == targetCluster);
    double indicatorVal = ((isTarget) ? (as_double(0x3ff0000000000000ul)) : (as_double(0x0000000000000000ul)));
    indicator[gid] = indicatorVal;
  }
}

)kernel";
const char* k_opencl_navatala_graph_compute_edge_cut_f32 = R"kernel(
__kernel void navatala_graph_compute_edge_cut_f32(__global const float* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const uint* clusters, __global const uint* n, __global float* edgeCut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = clusters[gid];
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + (uint)(1u));
    uint rowEnd = rowPtr[gidPlus1];
    float localCut = as_float(0x00000000u);
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
    bool hasContribution = (finalLocalCut > as_float(0x00000000u));
    if (hasContribution) {
      atomic_add(&edgeCut[0], finalLocalCut);
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_compute_edge_cut_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_compute_edge_cut_f64(__global const double* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const uint* clusters, __global const uint* n, __global double* edgeCut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = clusters[gid];
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + (uint)(1u));
    uint rowEnd = rowPtr[gidPlus1];
    double localCut = as_double(0x0000000000000000ul);
    for (int edgeIdx = 0; edgeIdx < (int)(rowEnd); ++edgeIdx) {
      uint edgeIdxU = ((uint)(edgeIdx));
      bool edgeInRange = (edgeIdxU >= rowStart);
      if (edgeInRange) {
        uint neighbor = colInd[edgeIdxU];
        uint neighborCluster = clusters[neighbor];
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
    bool hasContribution = (finalLocalCut > as_double(0x0000000000000000ul));
    if (hasContribution) {
      atomic_add(&edgeCut[0], finalLocalCut);
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_compute_partition_cost_f32 = R"kernel(
__kernel void navatala_graph_compute_partition_cost_f32(__global const float* edgeCutPerCluster, __global const uint* clusterSizes, __global const uint* numClusters, __global float* cost) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numClustersVal = numClusters[0];
  bool inBounds = (gid < numClustersVal);
  if (inBounds) {
    float clusterEdgeCut = edgeCutPerCluster[gid];
    uint clusterSize = clusterSizes[gid];
    bool sizeIsZero = (clusterSize == (uint)(0u));
    float clusterSizeF = ((float)(clusterSize));
    float normalizedCost = ((sizeIsZero) ? (as_float(0x00000000u)) : ((clusterEdgeCut / clusterSizeF)));
    bool hasContribution = (normalizedCost > as_float(0x00000000u));
    if (hasContribution) {
      atomic_add(&cost[0], normalizedCost);
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_compute_partition_cost_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_compute_partition_cost_f64(__global const double* edgeCutPerCluster, __global const uint* clusterSizes, __global const uint* numClusters, __global double* cost) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numClustersVal = numClusters[0];
  bool inBounds = (gid < numClustersVal);
  if (inBounds) {
    double clusterEdgeCut = edgeCutPerCluster[gid];
    uint clusterSize = clusterSizes[gid];
    bool sizeIsZero = (clusterSize == (uint)(0u));
    double clusterSizeF = ((double)(clusterSize));
    double normalizedCost = ((sizeIsZero) ? (as_double(0x0000000000000000ul)) : ((clusterEdgeCut / clusterSizeF)));
    bool hasContribution = (normalizedCost > as_double(0x0000000000000000ul));
    if (hasContribution) {
      atomic_add(&cost[0], normalizedCost);
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_analyze_partition_f32 = R"kernel(
__kernel void navatala_graph_analyze_partition_f32(__global const float* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const uint* clusters, __global const uint* n, __global const uint* numClusters, __global float* edgeCut, __global float* cost) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint numClustersVal = numClusters[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = clusters[gid];
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + (uint)(1u));
    uint rowEnd = rowPtr[gidPlus1];
    float nodeEdgeCut = as_float(0x00000000u);
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
    bool hasEdgeCut = (finalNodeEdgeCut > as_float(0x00000000u));
    if (hasEdgeCut) {
      atomic_add(&edgeCut[0], finalNodeEdgeCut);
    }
    float nValF = ((float)(nVal));
    float nodeCostContrib = (finalNodeEdgeCut / nValF);
    bool hasCostContrib = (nodeCostContrib > as_float(0x00000000u));
    if (hasCostContrib) {
      atomic_add(&cost[0], nodeCostContrib);
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_analyze_partition_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_analyze_partition_f64(__global const double* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const uint* clusters, __global const uint* n, __global const uint* numClusters, __global double* edgeCut, __global double* cost) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint numClustersVal = numClusters[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = clusters[gid];
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + (uint)(1u));
    uint rowEnd = rowPtr[gidPlus1];
    double nodeEdgeCut = as_double(0x0000000000000000ul);
    for (int edgeIdx = 0; edgeIdx < (int)(rowEnd); ++edgeIdx) {
      uint edgeIdxU = ((uint)(edgeIdx));
      bool edgeInRange = (edgeIdxU >= rowStart);
      if (edgeInRange) {
        uint neighbor = colInd[edgeIdxU];
        uint neighborCluster = clusters[neighbor];
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
    bool hasEdgeCut = (finalNodeEdgeCut > as_double(0x0000000000000000ul));
    if (hasEdgeCut) {
      atomic_add(&edgeCut[0], finalNodeEdgeCut);
    }
    double nValF = ((double)(nVal));
    double nodeCostContrib = (finalNodeEdgeCut / nValF);
    bool hasCostContrib = (nodeCostContrib > as_double(0x0000000000000000ul));
    if (hasCostContrib) {
      atomic_add(&cost[0], nodeCostContrib);
    }
  }
}

)kernel";
const char* k_opencl_navatala_graph_transform_eigen_matrix_f32 = R"kernel(
__kernel void navatala_graph_transform_eigen_matrix_f32(__global const float* eigenvectors, __global const uint* n, __global const uint* k, __global float* transformed) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
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
const char* k_opencl_navatala_graph_transform_eigen_matrix_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_transform_eigen_matrix_f64(__global const double* eigenvectors, __global const uint* n, __global const uint* k, __global double* transformed) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint kVal = k[0];
  uint totalElems = (kVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint eigIdx = (gid / nVal);
    uint sampleIdx = (gid % nVal);
    uint srcIdx = ((sampleIdx * kVal) + eigIdx);
    double val = eigenvectors[srcIdx];
    transformed[gid] = val;
  }
}

)kernel";
const char* k_opencl_navatala_graph_laplacian_from_adjacency_f32 = R"kernel(
__kernel void navatala_graph_laplacian_from_adjacency_f32(__global const float* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const uint* n, __global float* laplacianValues, __global float* degree) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint rowIdx = gid;
  bool inBounds = (rowIdx < nVal);
  if (inBounds) {
    uint rowStart = rowPtr[rowIdx];
    uint one32 = ((uint)(1));
    uint rowIdxPlusOne = (rowIdx + one32);
    uint rowEnd = rowPtr[rowIdxPlusOne];
    float degreeSumAccum = as_float(0x00000000u);
    for (int j = 0; j < (int)((rowEnd - rowStart)); ++j) {
      uint jU32 = ((uint)(j));
      uint absIdx = (rowStart + jU32);
      float adjVal = adjacencyValues[absIdx];
      float negAdjVal = (as_float(0x00000000u) - adjVal);
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
const char* k_opencl_navatala_graph_laplacian_from_adjacency_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_laplacian_from_adjacency_f64(__global const double* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const uint* n, __global double* laplacianValues, __global double* degree) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint rowIdx = gid;
  bool inBounds = (rowIdx < nVal);
  if (inBounds) {
    uint rowStart = rowPtr[rowIdx];
    uint one32 = ((uint)(1));
    uint rowIdxPlusOne = (rowIdx + one32);
    uint rowEnd = rowPtr[rowIdxPlusOne];
    double degreeSumAccum = as_double(0x0000000000000000ul);
    for (int j = 0; j < (int)((rowEnd - rowStart)); ++j) {
      uint jU32 = ((uint)(j));
      uint absIdx = (rowStart + jU32);
      double adjVal = adjacencyValues[absIdx];
      double negAdjVal = (as_double(0x0000000000000000ul) - adjVal);
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

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_pack_f32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_pack_f32 = {
  1,
  "navatala_graph_pack_f32",
  "opencl",
  "navatala_graph_pack_f32",
  "kernel:opencl:navatala_graph_pack_f32",
  "abi-r1:opencl:navatala_graph_pack_f32",
  "abi-r1:opencl:navatala_graph_pack_f32",
  4,
  kAbiArgs_opencl_navatala_graph_pack_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_pack_f64[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_pack_f64 = {
  1,
  "navatala_graph_pack_f64",
  "opencl",
  "navatala_graph_pack_f64",
  "kernel:opencl:navatala_graph_pack_f64",
  "abi-r1:opencl:navatala_graph_pack_f64",
  "abi-r1:opencl:navatala_graph_pack_f64",
  4,
  kAbiArgs_opencl_navatala_graph_pack_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_pack_i32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_pack_i32 = {
  1,
  "navatala_graph_pack_i32",
  "opencl",
  "navatala_graph_pack_i32",
  "kernel:opencl:navatala_graph_pack_i32",
  "abi-r1:opencl:navatala_graph_pack_i32",
  "abi-r1:opencl:navatala_graph_pack_i32",
  4,
  kAbiArgs_opencl_navatala_graph_pack_i32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_pack_u32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_pack_u32 = {
  1,
  "navatala_graph_pack_u32",
  "opencl",
  "navatala_graph_pack_u32",
  "kernel:opencl:navatala_graph_pack_u32",
  "abi-r1:opencl:navatala_graph_pack_u32",
  "abi-r1:opencl:navatala_graph_pack_u32",
  4,
  kAbiArgs_opencl_navatala_graph_pack_u32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_unpack_f32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_unpack_f32 = {
  1,
  "navatala_graph_unpack_f32",
  "opencl",
  "navatala_graph_unpack_f32",
  "kernel:opencl:navatala_graph_unpack_f32",
  "abi-r1:opencl:navatala_graph_unpack_f32",
  "abi-r1:opencl:navatala_graph_unpack_f32",
  4,
  kAbiArgs_opencl_navatala_graph_unpack_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_unpack_f64[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_unpack_f64 = {
  1,
  "navatala_graph_unpack_f64",
  "opencl",
  "navatala_graph_unpack_f64",
  "kernel:opencl:navatala_graph_unpack_f64",
  "abi-r1:opencl:navatala_graph_unpack_f64",
  "abi-r1:opencl:navatala_graph_unpack_f64",
  4,
  kAbiArgs_opencl_navatala_graph_unpack_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_unpack_i32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_unpack_i32 = {
  1,
  "navatala_graph_unpack_i32",
  "opencl",
  "navatala_graph_unpack_i32",
  "kernel:opencl:navatala_graph_unpack_i32",
  "abi-r1:opencl:navatala_graph_unpack_i32",
  "abi-r1:opencl:navatala_graph_unpack_i32",
  4,
  kAbiArgs_opencl_navatala_graph_unpack_i32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_unpack_u32[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_unpack_u32 = {
  1,
  "navatala_graph_unpack_u32",
  "opencl",
  "navatala_graph_unpack_u32",
  "kernel:opencl:navatala_graph_unpack_u32",
  "abi-r1:opencl:navatala_graph_unpack_u32",
  "abi-r1:opencl:navatala_graph_unpack_u32",
  4,
  kAbiArgs_opencl_navatala_graph_unpack_u32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_pack_f32x3[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_pack_f32x3 = {
  1,
  "navatala_graph_pack_f32x3",
  "opencl",
  "navatala_graph_pack_f32x3",
  "kernel:opencl:navatala_graph_pack_f32x3",
  "abi-r1:opencl:navatala_graph_pack_f32x3",
  "abi-r1:opencl:navatala_graph_pack_f32x3",
  5,
  kAbiArgs_opencl_navatala_graph_pack_f32x3
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_unpack_f32x3[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_unpack_f32x3 = {
  1,
  "navatala_graph_unpack_f32x3",
  "opencl",
  "navatala_graph_unpack_f32x3",
  "kernel:opencl:navatala_graph_unpack_f32x3",
  "abi-r1:opencl:navatala_graph_unpack_f32x3",
  "abi-r1:opencl:navatala_graph_unpack_f32x3",
  5,
  kAbiArgs_opencl_navatala_graph_unpack_f32x3
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_pack_f32x6[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_pack_f32x6 = {
  1,
  "navatala_graph_pack_f32x6",
  "opencl",
  "navatala_graph_pack_f32x6",
  "kernel:opencl:navatala_graph_pack_f32x6",
  "abi-r1:opencl:navatala_graph_pack_f32x6",
  "abi-r1:opencl:navatala_graph_pack_f32x6",
  5,
  kAbiArgs_opencl_navatala_graph_pack_f32x6
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_unpack_f32x6[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_unpack_f32x6 = {
  1,
  "navatala_graph_unpack_f32x6",
  "opencl",
  "navatala_graph_unpack_f32x6",
  "kernel:opencl:navatala_graph_unpack_f32x6",
  "abi-r1:opencl:navatala_graph_unpack_f32x6",
  "abi-r1:opencl:navatala_graph_unpack_f32x6",
  5,
  kAbiArgs_opencl_navatala_graph_unpack_f32x6
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_pack_f32x9[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_pack_f32x9 = {
  1,
  "navatala_graph_pack_f32x9",
  "opencl",
  "navatala_graph_pack_f32x9",
  "kernel:opencl:navatala_graph_pack_f32x9",
  "abi-r1:opencl:navatala_graph_pack_f32x9",
  "abi-r1:opencl:navatala_graph_pack_f32x9",
  5,
  kAbiArgs_opencl_navatala_graph_pack_f32x9
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_unpack_f32x9[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_unpack_f32x9 = {
  1,
  "navatala_graph_unpack_f32x9",
  "opencl",
  "navatala_graph_unpack_f32x9",
  "kernel:opencl:navatala_graph_unpack_f32x9",
  "abi-r1:opencl:navatala_graph_unpack_f32x9",
  "abi-r1:opencl:navatala_graph_unpack_f32x9",
  5,
  kAbiArgs_opencl_navatala_graph_unpack_f32x9
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_pack_f64x3[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_pack_f64x3 = {
  1,
  "navatala_graph_pack_f64x3",
  "opencl",
  "navatala_graph_pack_f64x3",
  "kernel:opencl:navatala_graph_pack_f64x3",
  "abi-r1:opencl:navatala_graph_pack_f64x3",
  "abi-r1:opencl:navatala_graph_pack_f64x3",
  5,
  kAbiArgs_opencl_navatala_graph_pack_f64x3
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_unpack_f64x3[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_unpack_f64x3 = {
  1,
  "navatala_graph_unpack_f64x3",
  "opencl",
  "navatala_graph_unpack_f64x3",
  "kernel:opencl:navatala_graph_unpack_f64x3",
  "abi-r1:opencl:navatala_graph_unpack_f64x3",
  "abi-r1:opencl:navatala_graph_unpack_f64x3",
  5,
  kAbiArgs_opencl_navatala_graph_unpack_f64x3
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_pack_f64x6[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_pack_f64x6 = {
  1,
  "navatala_graph_pack_f64x6",
  "opencl",
  "navatala_graph_pack_f64x6",
  "kernel:opencl:navatala_graph_pack_f64x6",
  "abi-r1:opencl:navatala_graph_pack_f64x6",
  "abi-r1:opencl:navatala_graph_pack_f64x6",
  5,
  kAbiArgs_opencl_navatala_graph_pack_f64x6
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_unpack_f64x6[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_unpack_f64x6 = {
  1,
  "navatala_graph_unpack_f64x6",
  "opencl",
  "navatala_graph_unpack_f64x6",
  "kernel:opencl:navatala_graph_unpack_f64x6",
  "abi-r1:opencl:navatala_graph_unpack_f64x6",
  "abi-r1:opencl:navatala_graph_unpack_f64x6",
  5,
  kAbiArgs_opencl_navatala_graph_unpack_f64x6
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_pack_f64x9[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_pack_f64x9 = {
  1,
  "navatala_graph_pack_f64x9",
  "opencl",
  "navatala_graph_pack_f64x9",
  "kernel:opencl:navatala_graph_pack_f64x9",
  "abi-r1:opencl:navatala_graph_pack_f64x9",
  "abi-r1:opencl:navatala_graph_pack_f64x9",
  5,
  kAbiArgs_opencl_navatala_graph_pack_f64x9
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_unpack_f64x9[] = {
  { "indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "src", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_unpack_f64x9 = {
  1,
  "navatala_graph_unpack_f64x9",
  "opencl",
  "navatala_graph_unpack_f64x9",
  "kernel:opencl:navatala_graph_unpack_f64x9",
  "abi-r1:opencl:navatala_graph_unpack_f64x9",
  "abi-r1:opencl:navatala_graph_unpack_f64x9",
  5,
  kAbiArgs_opencl_navatala_graph_unpack_f64x9
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_degree_out_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_degree_out_u32 = {
  1,
  "navatala_graph_degree_out_u32",
  "opencl",
  "navatala_graph_degree_out_u32",
  "kernel:opencl:navatala_graph_degree_out_u32",
  "abi-r1:opencl:navatala_graph_degree_out_u32",
  "abi-r1:opencl:navatala_graph_degree_out_u32",
  3,
  kAbiArgs_opencl_navatala_graph_degree_out_u32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_degree_in_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_degree_in_u32 = {
  1,
  "navatala_graph_degree_in_u32",
  "opencl",
  "navatala_graph_degree_in_u32",
  "kernel:opencl:navatala_graph_degree_in_u32",
  "abi-r1:opencl:navatala_graph_degree_in_u32",
  "abi-r1:opencl:navatala_graph_degree_in_u32",
  3,
  kAbiArgs_opencl_navatala_graph_degree_in_u32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_spmv_weighted_f32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_spmv_weighted_f32 = {
  1,
  "navatala_graph_spmv_weighted_f32",
  "opencl",
  "navatala_graph_spmv_weighted_f32",
  "kernel:opencl:navatala_graph_spmv_weighted_f32",
  "abi-r1:opencl:navatala_graph_spmv_weighted_f32",
  "abi-r1:opencl:navatala_graph_spmv_weighted_f32",
  6,
  kAbiArgs_opencl_navatala_graph_spmv_weighted_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_spmv_weighted_subgroup_f32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_spmv_weighted_subgroup_f32 = {
  1,
  "navatala_graph_spmv_weighted_subgroup_f32",
  "opencl",
  "navatala_graph_spmv_weighted_subgroup_f32",
  "kernel:opencl:navatala_graph_spmv_weighted_subgroup_f32",
  "abi-r1:opencl:navatala_graph_spmv_weighted_subgroup_f32",
  "abi-r1:opencl:navatala_graph_spmv_weighted_subgroup_f32",
  6,
  kAbiArgs_opencl_navatala_graph_spmv_weighted_subgroup_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_spmv_unweighted_f32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_spmv_unweighted_f32 = {
  1,
  "navatala_graph_spmv_unweighted_f32",
  "opencl",
  "navatala_graph_spmv_unweighted_f32",
  "kernel:opencl:navatala_graph_spmv_unweighted_f32",
  "abi-r1:opencl:navatala_graph_spmv_unweighted_f32",
  "abi-r1:opencl:navatala_graph_spmv_unweighted_f32",
  5,
  kAbiArgs_opencl_navatala_graph_spmv_unweighted_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_spmv_unweighted_subgroup_f32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_spmv_unweighted_subgroup_f32 = {
  1,
  "navatala_graph_spmv_unweighted_subgroup_f32",
  "opencl",
  "navatala_graph_spmv_unweighted_subgroup_f32",
  "kernel:opencl:navatala_graph_spmv_unweighted_subgroup_f32",
  "abi-r1:opencl:navatala_graph_spmv_unweighted_subgroup_f32",
  "abi-r1:opencl:navatala_graph_spmv_unweighted_subgroup_f32",
  5,
  kAbiArgs_opencl_navatala_graph_spmv_unweighted_subgroup_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_axpy2_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "z", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_axpy2_f32 = {
  1,
  "navatala_graph_axpy2_f32",
  "opencl",
  "navatala_graph_axpy2_f32",
  "kernel:opencl:navatala_graph_axpy2_f32",
  "abi-r1:opencl:navatala_graph_axpy2_f32",
  "abi-r1:opencl:navatala_graph_axpy2_f32",
  6,
  kAbiArgs_opencl_navatala_graph_axpy2_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_scale_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_scale_f32 = {
  1,
  "navatala_graph_scale_f32",
  "opencl",
  "navatala_graph_scale_f32",
  "kernel:opencl:navatala_graph_scale_f32",
  "abi-r1:opencl:navatala_graph_scale_f32",
  "abi-r1:opencl:navatala_graph_scale_f32",
  4,
  kAbiArgs_opencl_navatala_graph_scale_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_scale_add_bias_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "bias", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_scale_add_bias_f32 = {
  1,
  "navatala_graph_scale_add_bias_f32",
  "opencl",
  "navatala_graph_scale_add_bias_f32",
  "kernel:opencl:navatala_graph_scale_add_bias_f32",
  "abi-r1:opencl:navatala_graph_scale_add_bias_f32",
  "abi-r1:opencl:navatala_graph_scale_add_bias_f32",
  5,
  kAbiArgs_opencl_navatala_graph_scale_add_bias_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_l1norm_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_l1norm_f32 = {
  1,
  "navatala_graph_l1norm_f32",
  "opencl",
  "navatala_graph_l1norm_f32",
  "kernel:opencl:navatala_graph_l1norm_f32",
  "abi-r1:opencl:navatala_graph_l1norm_f32",
  "abi-r1:opencl:navatala_graph_l1norm_f32",
  3,
  kAbiArgs_opencl_navatala_graph_l1norm_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_recip_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_recip_f32 = {
  1,
  "navatala_graph_recip_f32",
  "opencl",
  "navatala_graph_recip_f32",
  "kernel:opencl:navatala_graph_recip_f32",
  "abi-r1:opencl:navatala_graph_recip_f32",
  "abi-r1:opencl:navatala_graph_recip_f32",
  2,
  kAbiArgs_opencl_navatala_graph_recip_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_l1_diff_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_l1_diff_f32 = {
  1,
  "navatala_graph_l1_diff_f32",
  "opencl",
  "navatala_graph_l1_diff_f32",
  "kernel:opencl:navatala_graph_l1_diff_f32",
  "abi-r1:opencl:navatala_graph_l1_diff_f32",
  "abi-r1:opencl:navatala_graph_l1_diff_f32",
  4,
  kAbiArgs_opencl_navatala_graph_l1_diff_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_count_diff_u32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_count_diff_u32 = {
  1,
  "navatala_graph_count_diff_u32",
  "opencl",
  "navatala_graph_count_diff_u32",
  "kernel:opencl:navatala_graph_count_diff_u32",
  "abi-r1:opencl:navatala_graph_count_diff_u32",
  "abi-r1:opencl:navatala_graph_count_diff_u32",
  4,
  kAbiArgs_opencl_navatala_graph_count_diff_u32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_copy_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_copy_f32 = {
  1,
  "navatala_graph_copy_f32",
  "opencl",
  "navatala_graph_copy_f32",
  "kernel:opencl:navatala_graph_copy_f32",
  "abi-r1:opencl:navatala_graph_copy_f32",
  "abi-r1:opencl:navatala_graph_copy_f32",
  3,
  kAbiArgs_opencl_navatala_graph_copy_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_copy_u32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_copy_u32 = {
  1,
  "navatala_graph_copy_u32",
  "opencl",
  "navatala_graph_copy_u32",
  "kernel:opencl:navatala_graph_copy_u32",
  "abi-r1:opencl:navatala_graph_copy_u32",
  "abi-r1:opencl:navatala_graph_copy_u32",
  3,
  kAbiArgs_opencl_navatala_graph_copy_u32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_relax_vertex_pull_f32[] = {
  { "inOffsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "inSrcs", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "inWeights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dist", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "inf", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_relax_vertex_pull_f32 = {
  1,
  "navatala_graph_relax_vertex_pull_f32",
  "opencl",
  "navatala_graph_relax_vertex_pull_f32",
  "kernel:opencl:navatala_graph_relax_vertex_pull_f32",
  "abi-r1:opencl:navatala_graph_relax_vertex_pull_f32",
  "abi-r1:opencl:navatala_graph_relax_vertex_pull_f32",
  7,
  kAbiArgs_opencl_navatala_graph_relax_vertex_pull_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_symmetrize_reverse_u32[] = {
  { "srcs", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dsts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numEdges", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outSrcs", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outDsts", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_symmetrize_reverse_u32 = {
  1,
  "navatala_graph_symmetrize_reverse_u32",
  "opencl",
  "navatala_graph_symmetrize_reverse_u32",
  "kernel:opencl:navatala_graph_symmetrize_reverse_u32",
  "abi-r1:opencl:navatala_graph_symmetrize_reverse_u32",
  "abi-r1:opencl:navatala_graph_symmetrize_reverse_u32",
  5,
  kAbiArgs_opencl_navatala_graph_symmetrize_reverse_u32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_apply_renumber_u32[] = {
  { "mapping", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "newIds", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_apply_renumber_u32 = {
  1,
  "navatala_graph_apply_renumber_u32",
  "opencl",
  "navatala_graph_apply_renumber_u32",
  "kernel:opencl:navatala_graph_apply_renumber_u32",
  "abi-r1:opencl:navatala_graph_apply_renumber_u32",
  "abi-r1:opencl:navatala_graph_apply_renumber_u32",
  3,
  kAbiArgs_opencl_navatala_graph_apply_renumber_u32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_peel_below_threshold_u32[] = {
  { "active", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "threshold", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numVertices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "active", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_peel_below_threshold_u32 = {
  1,
  "navatala_graph_peel_below_threshold_u32",
  "opencl",
  "navatala_graph_peel_below_threshold_u32",
  "kernel:opencl:navatala_graph_peel_below_threshold_u32",
  "abi-r1:opencl:navatala_graph_peel_below_threshold_u32",
  "abi-r1:opencl:navatala_graph_peel_below_threshold_u32",
  5,
  kAbiArgs_opencl_navatala_graph_peel_below_threshold_u32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_pair_intersection_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numPairs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "inter", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_pair_intersection_u32 = {
  1,
  "navatala_graph_pair_intersection_u32",
  "opencl",
  "navatala_graph_pair_intersection_u32",
  "kernel:opencl:navatala_graph_pair_intersection_u32",
  "abi-r1:opencl:navatala_graph_pair_intersection_u32",
  "abi-r1:opencl:navatala_graph_pair_intersection_u32",
  6,
  kAbiArgs_opencl_navatala_graph_pair_intersection_u32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_jaccard_f32[] = {
  { "inter", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numPairs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coeff", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_jaccard_f32 = {
  1,
  "navatala_graph_jaccard_f32",
  "opencl",
  "navatala_graph_jaccard_f32",
  "kernel:opencl:navatala_graph_jaccard_f32",
  "abi-r1:opencl:navatala_graph_jaccard_f32",
  "abi-r1:opencl:navatala_graph_jaccard_f32",
  6,
  kAbiArgs_opencl_navatala_graph_jaccard_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_overlap_f32[] = {
  { "inter", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numPairs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coeff", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_overlap_f32 = {
  1,
  "navatala_graph_overlap_f32",
  "opencl",
  "navatala_graph_overlap_f32",
  "kernel:opencl:navatala_graph_overlap_f32",
  "abi-r1:opencl:navatala_graph_overlap_f32",
  "abi-r1:opencl:navatala_graph_overlap_f32",
  6,
  kAbiArgs_opencl_navatala_graph_overlap_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_sorensen_f32[] = {
  { "inter", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numPairs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coeff", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_sorensen_f32 = {
  1,
  "navatala_graph_sorensen_f32",
  "opencl",
  "navatala_graph_sorensen_f32",
  "kernel:opencl:navatala_graph_sorensen_f32",
  "abi-r1:opencl:navatala_graph_sorensen_f32",
  "abi-r1:opencl:navatala_graph_sorensen_f32",
  6,
  kAbiArgs_opencl_navatala_graph_sorensen_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_cosine_f32[] = {
  { "inter", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pairsB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numPairs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coeff", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_cosine_f32 = {
  1,
  "navatala_graph_cosine_f32",
  "opencl",
  "navatala_graph_cosine_f32",
  "kernel:opencl:navatala_graph_cosine_f32",
  "abi-r1:opencl:navatala_graph_cosine_f32",
  "abi-r1:opencl:navatala_graph_cosine_f32",
  6,
  kAbiArgs_opencl_navatala_graph_cosine_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_uniform_random_walk_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "startVertices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "maxLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numWalks", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "walks", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_uniform_random_walk_u32 = {
  1,
  "navatala_graph_uniform_random_walk_u32",
  "opencl",
  "navatala_graph_uniform_random_walk_u32",
  "kernel:opencl:navatala_graph_uniform_random_walk_u32",
  "abi-r1:opencl:navatala_graph_uniform_random_walk_u32",
  "abi-r1:opencl:navatala_graph_uniform_random_walk_u32",
  7,
  kAbiArgs_opencl_navatala_graph_uniform_random_walk_u32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_out_degree_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_out_degree_u32 = {
  1,
  "navatala_graph_out_degree_u32",
  "opencl",
  "navatala_graph_out_degree_u32",
  "kernel:opencl:navatala_graph_out_degree_u32",
  "abi-r1:opencl:navatala_graph_out_degree_u32",
  "abi-r1:opencl:navatala_graph_out_degree_u32",
  3,
  kAbiArgs_opencl_navatala_graph_out_degree_u32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_in_degree_u32[] = {
  { "offsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numVertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_in_degree_u32 = {
  1,
  "navatala_graph_in_degree_u32",
  "opencl",
  "navatala_graph_in_degree_u32",
  "kernel:opencl:navatala_graph_in_degree_u32",
  "abi-r1:opencl:navatala_graph_in_degree_u32",
  "abi-r1:opencl:navatala_graph_in_degree_u32",
  3,
  kAbiArgs_opencl_navatala_graph_in_degree_u32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_rbf_affinity_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sigma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "W", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_rbf_affinity_f32 = {
  1,
  "navatala_graph_rbf_affinity_f32",
  "opencl",
  "navatala_graph_rbf_affinity_f32",
  "kernel:opencl:navatala_graph_rbf_affinity_f32",
  "abi-r1:opencl:navatala_graph_rbf_affinity_f32",
  "abi-r1:opencl:navatala_graph_rbf_affinity_f32",
  5,
  kAbiArgs_opencl_navatala_graph_rbf_affinity_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_rbf_affinity_f64[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sigma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "W", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_rbf_affinity_f64 = {
  1,
  "navatala_graph_rbf_affinity_f64",
  "opencl",
  "navatala_graph_rbf_affinity_f64",
  "kernel:opencl:navatala_graph_rbf_affinity_f64",
  "abi-r1:opencl:navatala_graph_rbf_affinity_f64",
  "abi-r1:opencl:navatala_graph_rbf_affinity_f64",
  5,
  kAbiArgs_opencl_navatala_graph_rbf_affinity_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_compute_degree_matrix_f32[] = {
  { "W", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "D", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_compute_degree_matrix_f32 = {
  1,
  "navatala_graph_compute_degree_matrix_f32",
  "opencl",
  "navatala_graph_compute_degree_matrix_f32",
  "kernel:opencl:navatala_graph_compute_degree_matrix_f32",
  "abi-r1:opencl:navatala_graph_compute_degree_matrix_f32",
  "abi-r1:opencl:navatala_graph_compute_degree_matrix_f32",
  3,
  kAbiArgs_opencl_navatala_graph_compute_degree_matrix_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_compute_degree_matrix_f64[] = {
  { "W", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "D", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_compute_degree_matrix_f64 = {
  1,
  "navatala_graph_compute_degree_matrix_f64",
  "opencl",
  "navatala_graph_compute_degree_matrix_f64",
  "kernel:opencl:navatala_graph_compute_degree_matrix_f64",
  "abi-r1:opencl:navatala_graph_compute_degree_matrix_f64",
  "abi-r1:opencl:navatala_graph_compute_degree_matrix_f64",
  3,
  kAbiArgs_opencl_navatala_graph_compute_degree_matrix_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_compute_laplacian_f32[] = {
  { "W", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "D", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "L", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_compute_laplacian_f32 = {
  1,
  "navatala_graph_compute_laplacian_f32",
  "opencl",
  "navatala_graph_compute_laplacian_f32",
  "kernel:opencl:navatala_graph_compute_laplacian_f32",
  "abi-r1:opencl:navatala_graph_compute_laplacian_f32",
  "abi-r1:opencl:navatala_graph_compute_laplacian_f32",
  4,
  kAbiArgs_opencl_navatala_graph_compute_laplacian_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_compute_laplacian_f64[] = {
  { "W", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "D", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "L", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_compute_laplacian_f64 = {
  1,
  "navatala_graph_compute_laplacian_f64",
  "opencl",
  "navatala_graph_compute_laplacian_f64",
  "kernel:opencl:navatala_graph_compute_laplacian_f64",
  "abi-r1:opencl:navatala_graph_compute_laplacian_f64",
  "abi-r1:opencl:navatala_graph_compute_laplacian_f64",
  4,
  kAbiArgs_opencl_navatala_graph_compute_laplacian_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_normalized_laplacian_f32[] = {
  { "W", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "D", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "L_sym", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_normalized_laplacian_f32 = {
  1,
  "navatala_graph_normalized_laplacian_f32",
  "opencl",
  "navatala_graph_normalized_laplacian_f32",
  "kernel:opencl:navatala_graph_normalized_laplacian_f32",
  "abi-r1:opencl:navatala_graph_normalized_laplacian_f32",
  "abi-r1:opencl:navatala_graph_normalized_laplacian_f32",
  4,
  kAbiArgs_opencl_navatala_graph_normalized_laplacian_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_normalized_laplacian_f64[] = {
  { "W", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "D", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "L_sym", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_normalized_laplacian_f64 = {
  1,
  "navatala_graph_normalized_laplacian_f64",
  "opencl",
  "navatala_graph_normalized_laplacian_f64",
  "kernel:opencl:navatala_graph_normalized_laplacian_f64",
  "abi-r1:opencl:navatala_graph_normalized_laplacian_f64",
  "abi-r1:opencl:navatala_graph_normalized_laplacian_f64",
  4,
  kAbiArgs_opencl_navatala_graph_normalized_laplacian_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_normalize_eigenvector_rows_f32[] = {
  { "V", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "V_norm", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_normalize_eigenvector_rows_f32 = {
  1,
  "navatala_graph_normalize_eigenvector_rows_f32",
  "opencl",
  "navatala_graph_normalize_eigenvector_rows_f32",
  "kernel:opencl:navatala_graph_normalize_eigenvector_rows_f32",
  "abi-r1:opencl:navatala_graph_normalize_eigenvector_rows_f32",
  "abi-r1:opencl:navatala_graph_normalize_eigenvector_rows_f32",
  4,
  kAbiArgs_opencl_navatala_graph_normalize_eigenvector_rows_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_normalize_eigenvector_rows_f64[] = {
  { "V", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "V_norm", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_normalize_eigenvector_rows_f64 = {
  1,
  "navatala_graph_normalize_eigenvector_rows_f64",
  "opencl",
  "navatala_graph_normalize_eigenvector_rows_f64",
  "kernel:opencl:navatala_graph_normalize_eigenvector_rows_f64",
  "abi-r1:opencl:navatala_graph_normalize_eigenvector_rows_f64",
  "abi-r1:opencl:navatala_graph_normalize_eigenvector_rows_f64",
  4,
  kAbiArgs_opencl_navatala_graph_normalize_eigenvector_rows_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_spectral_embedding_f32[] = {
  { "eigenvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "embedding", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_spectral_embedding_f32 = {
  1,
  "navatala_graph_spectral_embedding_f32",
  "opencl",
  "navatala_graph_spectral_embedding_f32",
  "kernel:opencl:navatala_graph_spectral_embedding_f32",
  "abi-r1:opencl:navatala_graph_spectral_embedding_f32",
  "abi-r1:opencl:navatala_graph_spectral_embedding_f32",
  4,
  kAbiArgs_opencl_navatala_graph_spectral_embedding_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_spectral_embedding_f64[] = {
  { "eigenvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "embedding", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_spectral_embedding_f64 = {
  1,
  "navatala_graph_spectral_embedding_f64",
  "opencl",
  "navatala_graph_spectral_embedding_f64",
  "kernel:opencl:navatala_graph_spectral_embedding_f64",
  "abi-r1:opencl:navatala_graph_spectral_embedding_f64",
  "abi-r1:opencl:navatala_graph_spectral_embedding_f64",
  4,
  kAbiArgs_opencl_navatala_graph_spectral_embedding_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_compute_modularity_f32[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusters", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClusters", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalEdgeWeight", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "modularity", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_compute_modularity_f32 = {
  1,
  "navatala_graph_compute_modularity_f32",
  "opencl",
  "navatala_graph_compute_modularity_f32",
  "kernel:opencl:navatala_graph_compute_modularity_f32",
  "abi-r1:opencl:navatala_graph_compute_modularity_f32",
  "abi-r1:opencl:navatala_graph_compute_modularity_f32",
  8,
  kAbiArgs_opencl_navatala_graph_compute_modularity_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_compute_modularity_f64[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusters", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClusters", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalEdgeWeight", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "modularity", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_compute_modularity_f64 = {
  1,
  "navatala_graph_compute_modularity_f64",
  "opencl",
  "navatala_graph_compute_modularity_f64",
  "kernel:opencl:navatala_graph_compute_modularity_f64",
  "abi-r1:opencl:navatala_graph_compute_modularity_f64",
  "abi-r1:opencl:navatala_graph_compute_modularity_f64",
  8,
  kAbiArgs_opencl_navatala_graph_compute_modularity_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_modularity_matrix_vec_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_modularity_matrix_vec_f32 = {
  1,
  "navatala_graph_modularity_matrix_vec_f32",
  "opencl",
  "navatala_graph_modularity_matrix_vec_f32",
  "kernel:opencl:navatala_graph_modularity_matrix_vec_f32",
  "abi-r1:opencl:navatala_graph_modularity_matrix_vec_f32",
  "abi-r1:opencl:navatala_graph_modularity_matrix_vec_f32",
  9,
  kAbiArgs_opencl_navatala_graph_modularity_matrix_vec_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_modularity_matrix_vec_f64[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_modularity_matrix_vec_f64 = {
  1,
  "navatala_graph_modularity_matrix_vec_f64",
  "opencl",
  "navatala_graph_modularity_matrix_vec_f64",
  "kernel:opencl:navatala_graph_modularity_matrix_vec_f64",
  "abi-r1:opencl:navatala_graph_modularity_matrix_vec_f64",
  "abi-r1:opencl:navatala_graph_modularity_matrix_vec_f64",
  9,
  kAbiArgs_opencl_navatala_graph_modularity_matrix_vec_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_construct_indicator_f32[] = {
  { "clusters", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusterIndex", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indicator", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_construct_indicator_f32 = {
  1,
  "navatala_graph_construct_indicator_f32",
  "opencl",
  "navatala_graph_construct_indicator_f32",
  "kernel:opencl:navatala_graph_construct_indicator_f32",
  "abi-r1:opencl:navatala_graph_construct_indicator_f32",
  "abi-r1:opencl:navatala_graph_construct_indicator_f32",
  4,
  kAbiArgs_opencl_navatala_graph_construct_indicator_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_construct_indicator_f64[] = {
  { "clusters", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusterIndex", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indicator", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_construct_indicator_f64 = {
  1,
  "navatala_graph_construct_indicator_f64",
  "opencl",
  "navatala_graph_construct_indicator_f64",
  "kernel:opencl:navatala_graph_construct_indicator_f64",
  "abi-r1:opencl:navatala_graph_construct_indicator_f64",
  "abi-r1:opencl:navatala_graph_construct_indicator_f64",
  4,
  kAbiArgs_opencl_navatala_graph_construct_indicator_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_compute_edge_cut_f32[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusters", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "edgeCut", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_compute_edge_cut_f32 = {
  1,
  "navatala_graph_compute_edge_cut_f32",
  "opencl",
  "navatala_graph_compute_edge_cut_f32",
  "kernel:opencl:navatala_graph_compute_edge_cut_f32",
  "abi-r1:opencl:navatala_graph_compute_edge_cut_f32",
  "abi-r1:opencl:navatala_graph_compute_edge_cut_f32",
  6,
  kAbiArgs_opencl_navatala_graph_compute_edge_cut_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_compute_edge_cut_f64[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusters", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "edgeCut", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_compute_edge_cut_f64 = {
  1,
  "navatala_graph_compute_edge_cut_f64",
  "opencl",
  "navatala_graph_compute_edge_cut_f64",
  "kernel:opencl:navatala_graph_compute_edge_cut_f64",
  "abi-r1:opencl:navatala_graph_compute_edge_cut_f64",
  "abi-r1:opencl:navatala_graph_compute_edge_cut_f64",
  6,
  kAbiArgs_opencl_navatala_graph_compute_edge_cut_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_compute_partition_cost_f32[] = {
  { "edgeCutPerCluster", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusterSizes", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numClusters", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cost", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_compute_partition_cost_f32 = {
  1,
  "navatala_graph_compute_partition_cost_f32",
  "opencl",
  "navatala_graph_compute_partition_cost_f32",
  "kernel:opencl:navatala_graph_compute_partition_cost_f32",
  "abi-r1:opencl:navatala_graph_compute_partition_cost_f32",
  "abi-r1:opencl:navatala_graph_compute_partition_cost_f32",
  4,
  kAbiArgs_opencl_navatala_graph_compute_partition_cost_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_compute_partition_cost_f64[] = {
  { "edgeCutPerCluster", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusterSizes", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numClusters", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cost", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_compute_partition_cost_f64 = {
  1,
  "navatala_graph_compute_partition_cost_f64",
  "opencl",
  "navatala_graph_compute_partition_cost_f64",
  "kernel:opencl:navatala_graph_compute_partition_cost_f64",
  "abi-r1:opencl:navatala_graph_compute_partition_cost_f64",
  "abi-r1:opencl:navatala_graph_compute_partition_cost_f64",
  4,
  kAbiArgs_opencl_navatala_graph_compute_partition_cost_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_analyze_partition_f32[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusters", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClusters", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "edgeCut", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cost", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_analyze_partition_f32 = {
  1,
  "navatala_graph_analyze_partition_f32",
  "opencl",
  "navatala_graph_analyze_partition_f32",
  "kernel:opencl:navatala_graph_analyze_partition_f32",
  "abi-r1:opencl:navatala_graph_analyze_partition_f32",
  "abi-r1:opencl:navatala_graph_analyze_partition_f32",
  8,
  kAbiArgs_opencl_navatala_graph_analyze_partition_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_analyze_partition_f64[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusters", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClusters", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "edgeCut", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "cost", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_analyze_partition_f64 = {
  1,
  "navatala_graph_analyze_partition_f64",
  "opencl",
  "navatala_graph_analyze_partition_f64",
  "kernel:opencl:navatala_graph_analyze_partition_f64",
  "abi-r1:opencl:navatala_graph_analyze_partition_f64",
  "abi-r1:opencl:navatala_graph_analyze_partition_f64",
  8,
  kAbiArgs_opencl_navatala_graph_analyze_partition_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_transform_eigen_matrix_f32[] = {
  { "eigenvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "transformed", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_transform_eigen_matrix_f32 = {
  1,
  "navatala_graph_transform_eigen_matrix_f32",
  "opencl",
  "navatala_graph_transform_eigen_matrix_f32",
  "kernel:opencl:navatala_graph_transform_eigen_matrix_f32",
  "abi-r1:opencl:navatala_graph_transform_eigen_matrix_f32",
  "abi-r1:opencl:navatala_graph_transform_eigen_matrix_f32",
  4,
  kAbiArgs_opencl_navatala_graph_transform_eigen_matrix_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_transform_eigen_matrix_f64[] = {
  { "eigenvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "transformed", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_transform_eigen_matrix_f64 = {
  1,
  "navatala_graph_transform_eigen_matrix_f64",
  "opencl",
  "navatala_graph_transform_eigen_matrix_f64",
  "kernel:opencl:navatala_graph_transform_eigen_matrix_f64",
  "abi-r1:opencl:navatala_graph_transform_eigen_matrix_f64",
  "abi-r1:opencl:navatala_graph_transform_eigen_matrix_f64",
  4,
  kAbiArgs_opencl_navatala_graph_transform_eigen_matrix_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_laplacian_from_adjacency_f32[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "laplacianValues", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degree", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_laplacian_from_adjacency_f32 = {
  1,
  "navatala_graph_laplacian_from_adjacency_f32",
  "opencl",
  "navatala_graph_laplacian_from_adjacency_f32",
  "kernel:opencl:navatala_graph_laplacian_from_adjacency_f32",
  "abi-r1:opencl:navatala_graph_laplacian_from_adjacency_f32",
  "abi-r1:opencl:navatala_graph_laplacian_from_adjacency_f32",
  6,
  kAbiArgs_opencl_navatala_graph_laplacian_from_adjacency_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_graph_laplacian_from_adjacency_f64[] = {
  { "adjacencyValues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowPtr", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colInd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "laplacianValues", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degree", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_graph_laplacian_from_adjacency_f64 = {
  1,
  "navatala_graph_laplacian_from_adjacency_f64",
  "opencl",
  "navatala_graph_laplacian_from_adjacency_f64",
  "kernel:opencl:navatala_graph_laplacian_from_adjacency_f64",
  "abi-r1:opencl:navatala_graph_laplacian_from_adjacency_f64",
  "abi-r1:opencl:navatala_graph_laplacian_from_adjacency_f64",
  6,
  kAbiArgs_opencl_navatala_graph_laplacian_from_adjacency_f64
};

bool tryGetKernelAbiManifest_opencl_graph(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "opencl" && kernelName == "navatala_graph_pack_f32") {
    out = &kAbiManifest_opencl_navatala_graph_pack_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_f64") {
    out = &kAbiManifest_opencl_navatala_graph_pack_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_i32") {
    out = &kAbiManifest_opencl_navatala_graph_pack_i32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_u32") {
    out = &kAbiManifest_opencl_navatala_graph_pack_u32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_f32") {
    out = &kAbiManifest_opencl_navatala_graph_unpack_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_f64") {
    out = &kAbiManifest_opencl_navatala_graph_unpack_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_i32") {
    out = &kAbiManifest_opencl_navatala_graph_unpack_i32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_u32") {
    out = &kAbiManifest_opencl_navatala_graph_unpack_u32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_f32x3") {
    out = &kAbiManifest_opencl_navatala_graph_pack_f32x3;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_f32x3") {
    out = &kAbiManifest_opencl_navatala_graph_unpack_f32x3;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_f32x6") {
    out = &kAbiManifest_opencl_navatala_graph_pack_f32x6;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_f32x6") {
    out = &kAbiManifest_opencl_navatala_graph_unpack_f32x6;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_f32x9") {
    out = &kAbiManifest_opencl_navatala_graph_pack_f32x9;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_f32x9") {
    out = &kAbiManifest_opencl_navatala_graph_unpack_f32x9;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_f64x3") {
    out = &kAbiManifest_opencl_navatala_graph_pack_f64x3;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_f64x3") {
    out = &kAbiManifest_opencl_navatala_graph_unpack_f64x3;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_f64x6") {
    out = &kAbiManifest_opencl_navatala_graph_pack_f64x6;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_f64x6") {
    out = &kAbiManifest_opencl_navatala_graph_unpack_f64x6;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_f64x9") {
    out = &kAbiManifest_opencl_navatala_graph_pack_f64x9;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_f64x9") {
    out = &kAbiManifest_opencl_navatala_graph_unpack_f64x9;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_degree_out_u32") {
    out = &kAbiManifest_opencl_navatala_graph_degree_out_u32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_degree_in_u32") {
    out = &kAbiManifest_opencl_navatala_graph_degree_in_u32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_spmv_weighted_f32") {
    out = &kAbiManifest_opencl_navatala_graph_spmv_weighted_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_spmv_weighted_subgroup_f32") {
    out = &kAbiManifest_opencl_navatala_graph_spmv_weighted_subgroup_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_spmv_unweighted_f32") {
    out = &kAbiManifest_opencl_navatala_graph_spmv_unweighted_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_spmv_unweighted_subgroup_f32") {
    out = &kAbiManifest_opencl_navatala_graph_spmv_unweighted_subgroup_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_axpy2_f32") {
    out = &kAbiManifest_opencl_navatala_graph_axpy2_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_scale_f32") {
    out = &kAbiManifest_opencl_navatala_graph_scale_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_scale_add_bias_f32") {
    out = &kAbiManifest_opencl_navatala_graph_scale_add_bias_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_l1norm_f32") {
    out = &kAbiManifest_opencl_navatala_graph_l1norm_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_recip_f32") {
    out = &kAbiManifest_opencl_navatala_graph_recip_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_l1_diff_f32") {
    out = &kAbiManifest_opencl_navatala_graph_l1_diff_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_count_diff_u32") {
    out = &kAbiManifest_opencl_navatala_graph_count_diff_u32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_copy_f32") {
    out = &kAbiManifest_opencl_navatala_graph_copy_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_copy_u32") {
    out = &kAbiManifest_opencl_navatala_graph_copy_u32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_relax_vertex_pull_f32") {
    out = &kAbiManifest_opencl_navatala_graph_relax_vertex_pull_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_symmetrize_reverse_u32") {
    out = &kAbiManifest_opencl_navatala_graph_symmetrize_reverse_u32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_apply_renumber_u32") {
    out = &kAbiManifest_opencl_navatala_graph_apply_renumber_u32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_peel_below_threshold_u32") {
    out = &kAbiManifest_opencl_navatala_graph_peel_below_threshold_u32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pair_intersection_u32") {
    out = &kAbiManifest_opencl_navatala_graph_pair_intersection_u32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_jaccard_f32") {
    out = &kAbiManifest_opencl_navatala_graph_jaccard_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_overlap_f32") {
    out = &kAbiManifest_opencl_navatala_graph_overlap_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_sorensen_f32") {
    out = &kAbiManifest_opencl_navatala_graph_sorensen_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_cosine_f32") {
    out = &kAbiManifest_opencl_navatala_graph_cosine_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_uniform_random_walk_u32") {
    out = &kAbiManifest_opencl_navatala_graph_uniform_random_walk_u32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_out_degree_u32") {
    out = &kAbiManifest_opencl_navatala_graph_out_degree_u32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_in_degree_u32") {
    out = &kAbiManifest_opencl_navatala_graph_in_degree_u32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_rbf_affinity_f32") {
    out = &kAbiManifest_opencl_navatala_graph_rbf_affinity_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_rbf_affinity_f64") {
    out = &kAbiManifest_opencl_navatala_graph_rbf_affinity_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_degree_matrix_f32") {
    out = &kAbiManifest_opencl_navatala_graph_compute_degree_matrix_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_degree_matrix_f64") {
    out = &kAbiManifest_opencl_navatala_graph_compute_degree_matrix_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_laplacian_f32") {
    out = &kAbiManifest_opencl_navatala_graph_compute_laplacian_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_laplacian_f64") {
    out = &kAbiManifest_opencl_navatala_graph_compute_laplacian_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_normalized_laplacian_f32") {
    out = &kAbiManifest_opencl_navatala_graph_normalized_laplacian_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_normalized_laplacian_f64") {
    out = &kAbiManifest_opencl_navatala_graph_normalized_laplacian_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_normalize_eigenvector_rows_f32") {
    out = &kAbiManifest_opencl_navatala_graph_normalize_eigenvector_rows_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_normalize_eigenvector_rows_f64") {
    out = &kAbiManifest_opencl_navatala_graph_normalize_eigenvector_rows_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_spectral_embedding_f32") {
    out = &kAbiManifest_opencl_navatala_graph_spectral_embedding_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_spectral_embedding_f64") {
    out = &kAbiManifest_opencl_navatala_graph_spectral_embedding_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_modularity_f32") {
    out = &kAbiManifest_opencl_navatala_graph_compute_modularity_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_modularity_f64") {
    out = &kAbiManifest_opencl_navatala_graph_compute_modularity_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_modularity_matrix_vec_f32") {
    out = &kAbiManifest_opencl_navatala_graph_modularity_matrix_vec_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_modularity_matrix_vec_f64") {
    out = &kAbiManifest_opencl_navatala_graph_modularity_matrix_vec_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_construct_indicator_f32") {
    out = &kAbiManifest_opencl_navatala_graph_construct_indicator_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_construct_indicator_f64") {
    out = &kAbiManifest_opencl_navatala_graph_construct_indicator_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_edge_cut_f32") {
    out = &kAbiManifest_opencl_navatala_graph_compute_edge_cut_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_edge_cut_f64") {
    out = &kAbiManifest_opencl_navatala_graph_compute_edge_cut_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_partition_cost_f32") {
    out = &kAbiManifest_opencl_navatala_graph_compute_partition_cost_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_partition_cost_f64") {
    out = &kAbiManifest_opencl_navatala_graph_compute_partition_cost_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_analyze_partition_f32") {
    out = &kAbiManifest_opencl_navatala_graph_analyze_partition_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_analyze_partition_f64") {
    out = &kAbiManifest_opencl_navatala_graph_analyze_partition_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_transform_eigen_matrix_f32") {
    out = &kAbiManifest_opencl_navatala_graph_transform_eigen_matrix_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_transform_eigen_matrix_f64") {
    out = &kAbiManifest_opencl_navatala_graph_transform_eigen_matrix_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_laplacian_from_adjacency_f32") {
    out = &kAbiManifest_opencl_navatala_graph_laplacian_from_adjacency_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_laplacian_from_adjacency_f64") {
    out = &kAbiManifest_opencl_navatala_graph_laplacian_from_adjacency_f64;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_opencl_graph(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "opencl" && kernelName == "navatala_graph_pack_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_pack_f32";
    std::string_view sv(k_opencl_navatala_graph_pack_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_pack_f64";
    std::string_view sv(k_opencl_navatala_graph_pack_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_pack_i32";
    std::string_view sv(k_opencl_navatala_graph_pack_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_pack_u32";
    std::string_view sv(k_opencl_navatala_graph_pack_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_unpack_f32";
    std::string_view sv(k_opencl_navatala_graph_unpack_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_unpack_f64";
    std::string_view sv(k_opencl_navatala_graph_unpack_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_unpack_i32";
    std::string_view sv(k_opencl_navatala_graph_unpack_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_unpack_u32";
    std::string_view sv(k_opencl_navatala_graph_unpack_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_f32x3") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_pack_f32x3";
    std::string_view sv(k_opencl_navatala_graph_pack_f32x3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_f32x3") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_unpack_f32x3";
    std::string_view sv(k_opencl_navatala_graph_unpack_f32x3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_f32x6") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_pack_f32x6";
    std::string_view sv(k_opencl_navatala_graph_pack_f32x6);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_f32x6") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_unpack_f32x6";
    std::string_view sv(k_opencl_navatala_graph_unpack_f32x6);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_f32x9") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_pack_f32x9";
    std::string_view sv(k_opencl_navatala_graph_pack_f32x9);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_f32x9") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_unpack_f32x9";
    std::string_view sv(k_opencl_navatala_graph_unpack_f32x9);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_f64x3") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_pack_f64x3";
    std::string_view sv(k_opencl_navatala_graph_pack_f64x3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_f64x3") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_unpack_f64x3";
    std::string_view sv(k_opencl_navatala_graph_unpack_f64x3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_f64x6") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_pack_f64x6";
    std::string_view sv(k_opencl_navatala_graph_pack_f64x6);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_f64x6") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_unpack_f64x6";
    std::string_view sv(k_opencl_navatala_graph_unpack_f64x6);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pack_f64x9") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_pack_f64x9";
    std::string_view sv(k_opencl_navatala_graph_pack_f64x9);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_unpack_f64x9") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_unpack_f64x9";
    std::string_view sv(k_opencl_navatala_graph_unpack_f64x9);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_degree_out_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_degree_out_u32";
    std::string_view sv(k_opencl_navatala_graph_degree_out_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_degree_in_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_degree_in_u32";
    std::string_view sv(k_opencl_navatala_graph_degree_in_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_spmv_weighted_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_spmv_weighted_f32";
    std::string_view sv(k_opencl_navatala_graph_spmv_weighted_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_spmv_weighted_subgroup_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_spmv_weighted_subgroup_f32";
    std::string_view sv(k_opencl_navatala_graph_spmv_weighted_subgroup_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_spmv_unweighted_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_spmv_unweighted_f32";
    std::string_view sv(k_opencl_navatala_graph_spmv_unweighted_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_spmv_unweighted_subgroup_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_spmv_unweighted_subgroup_f32";
    std::string_view sv(k_opencl_navatala_graph_spmv_unweighted_subgroup_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_axpy2_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_axpy2_f32";
    std::string_view sv(k_opencl_navatala_graph_axpy2_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_scale_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_scale_f32";
    std::string_view sv(k_opencl_navatala_graph_scale_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_scale_add_bias_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_scale_add_bias_f32";
    std::string_view sv(k_opencl_navatala_graph_scale_add_bias_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_l1norm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_l1norm_f32";
    std::string_view sv(k_opencl_navatala_graph_l1norm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_recip_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_recip_f32";
    std::string_view sv(k_opencl_navatala_graph_recip_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_l1_diff_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_l1_diff_f32";
    std::string_view sv(k_opencl_navatala_graph_l1_diff_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_count_diff_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_count_diff_u32";
    std::string_view sv(k_opencl_navatala_graph_count_diff_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_copy_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_copy_f32";
    std::string_view sv(k_opencl_navatala_graph_copy_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_copy_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_copy_u32";
    std::string_view sv(k_opencl_navatala_graph_copy_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_relax_vertex_pull_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_relax_vertex_pull_f32";
    std::string_view sv(k_opencl_navatala_graph_relax_vertex_pull_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_symmetrize_reverse_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_symmetrize_reverse_u32";
    std::string_view sv(k_opencl_navatala_graph_symmetrize_reverse_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_apply_renumber_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_apply_renumber_u32";
    std::string_view sv(k_opencl_navatala_graph_apply_renumber_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_peel_below_threshold_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_peel_below_threshold_u32";
    std::string_view sv(k_opencl_navatala_graph_peel_below_threshold_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_pair_intersection_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_pair_intersection_u32";
    std::string_view sv(k_opencl_navatala_graph_pair_intersection_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_jaccard_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_jaccard_f32";
    std::string_view sv(k_opencl_navatala_graph_jaccard_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_overlap_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_overlap_f32";
    std::string_view sv(k_opencl_navatala_graph_overlap_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_sorensen_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_sorensen_f32";
    std::string_view sv(k_opencl_navatala_graph_sorensen_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_cosine_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_cosine_f32";
    std::string_view sv(k_opencl_navatala_graph_cosine_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_uniform_random_walk_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_uniform_random_walk_u32";
    std::string_view sv(k_opencl_navatala_graph_uniform_random_walk_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_out_degree_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_out_degree_u32";
    std::string_view sv(k_opencl_navatala_graph_out_degree_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_in_degree_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_in_degree_u32";
    std::string_view sv(k_opencl_navatala_graph_in_degree_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_rbf_affinity_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_rbf_affinity_f32";
    std::string_view sv(k_opencl_navatala_graph_rbf_affinity_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_rbf_affinity_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_rbf_affinity_f64";
    std::string_view sv(k_opencl_navatala_graph_rbf_affinity_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_degree_matrix_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_compute_degree_matrix_f32";
    std::string_view sv(k_opencl_navatala_graph_compute_degree_matrix_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_degree_matrix_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_compute_degree_matrix_f64";
    std::string_view sv(k_opencl_navatala_graph_compute_degree_matrix_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_laplacian_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_compute_laplacian_f32";
    std::string_view sv(k_opencl_navatala_graph_compute_laplacian_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_laplacian_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_compute_laplacian_f64";
    std::string_view sv(k_opencl_navatala_graph_compute_laplacian_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_normalized_laplacian_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_normalized_laplacian_f32";
    std::string_view sv(k_opencl_navatala_graph_normalized_laplacian_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_normalized_laplacian_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_normalized_laplacian_f64";
    std::string_view sv(k_opencl_navatala_graph_normalized_laplacian_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_normalize_eigenvector_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_normalize_eigenvector_rows_f32";
    std::string_view sv(k_opencl_navatala_graph_normalize_eigenvector_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_normalize_eigenvector_rows_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_normalize_eigenvector_rows_f64";
    std::string_view sv(k_opencl_navatala_graph_normalize_eigenvector_rows_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_spectral_embedding_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_spectral_embedding_f32";
    std::string_view sv(k_opencl_navatala_graph_spectral_embedding_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_spectral_embedding_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_spectral_embedding_f64";
    std::string_view sv(k_opencl_navatala_graph_spectral_embedding_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_modularity_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_compute_modularity_f32";
    std::string_view sv(k_opencl_navatala_graph_compute_modularity_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_modularity_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_compute_modularity_f64";
    std::string_view sv(k_opencl_navatala_graph_compute_modularity_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_modularity_matrix_vec_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_modularity_matrix_vec_f32";
    std::string_view sv(k_opencl_navatala_graph_modularity_matrix_vec_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_modularity_matrix_vec_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_modularity_matrix_vec_f64";
    std::string_view sv(k_opencl_navatala_graph_modularity_matrix_vec_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_construct_indicator_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_construct_indicator_f32";
    std::string_view sv(k_opencl_navatala_graph_construct_indicator_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_construct_indicator_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_construct_indicator_f64";
    std::string_view sv(k_opencl_navatala_graph_construct_indicator_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_edge_cut_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_compute_edge_cut_f32";
    std::string_view sv(k_opencl_navatala_graph_compute_edge_cut_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_edge_cut_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_compute_edge_cut_f64";
    std::string_view sv(k_opencl_navatala_graph_compute_edge_cut_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_partition_cost_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_compute_partition_cost_f32";
    std::string_view sv(k_opencl_navatala_graph_compute_partition_cost_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_compute_partition_cost_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_compute_partition_cost_f64";
    std::string_view sv(k_opencl_navatala_graph_compute_partition_cost_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_analyze_partition_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_analyze_partition_f32";
    std::string_view sv(k_opencl_navatala_graph_analyze_partition_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_analyze_partition_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_analyze_partition_f64";
    std::string_view sv(k_opencl_navatala_graph_analyze_partition_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_transform_eigen_matrix_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_transform_eigen_matrix_f32";
    std::string_view sv(k_opencl_navatala_graph_transform_eigen_matrix_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_transform_eigen_matrix_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_transform_eigen_matrix_f64";
    std::string_view sv(k_opencl_navatala_graph_transform_eigen_matrix_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_laplacian_from_adjacency_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_laplacian_from_adjacency_f32";
    std::string_view sv(k_opencl_navatala_graph_laplacian_from_adjacency_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_graph_laplacian_from_adjacency_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_graph_laplacian_from_adjacency_f64";
    std::string_view sv(k_opencl_navatala_graph_laplacian_from_adjacency_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

