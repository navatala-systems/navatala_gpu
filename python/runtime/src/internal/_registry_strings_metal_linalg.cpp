// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_linalg` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_linalg_norm_linf_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_norm_linf_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint gid = ((uint)(int(__gid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  float val = abs(_input[gid]);
  if (inBounds) {
    sdata[lid] = val;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint linfF32RedStride = 128u;
  for (int linfF32RedStep = 0; linfF32RedStep < (int)(8); ++linfF32RedStep) {
    uint linfF32Stride = linfF32RedStride;
    if (lid < linfF32Stride) {
      float other = sdata[(lid + linfF32Stride)];
      float mine = sdata[lid];
      bool mineGtOther = (mine > other);
      float maxVal = ((mineGtOther) ? (mine) : (other));
      sdata[lid] = maxVal;
    }
    uint linfF32StrideToHalve = linfF32RedStride;
    uint linfF32NextStride = (linfF32StrideToHalve >> 1u);
    linfF32RedStride = linfF32NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    result[0] = sdata[0];
  }
}

)kernel";
const char* k_metal_navatala_linalg_eig_select_top_k_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_eig_select_top_k_f32(device const float* eigenvalues [[buffer(0)]], device const float* eigenvectors [[buffer(1)]], device const uint* k [[buffer(2)]], device const uint* n [[buffer(3)]], device float* topKValues [[buffer(4)]], device float* topKVectors [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint kVal = k[0];
  uint nVal = n[0];
  uint totalWork = (kVal * nVal);
  bool inBounds = (gid < totalWork);
  if (inBounds) {
    uint col = (gid / nVal);
    uint row = (gid % nVal);
    bool isFirstRow = (row == 0u);
    if (isFirstRow) {
      float eigenval = eigenvalues[col];
      topKValues[col] = eigenval;
    }
    uint srcIdx = ((col * nVal) + row);
    float vecElem = eigenvectors[srcIdx];
    topKVectors[srcIdx] = vecElem;
  }
}

)kernel";
const char* k_metal_navatala_linalg_cast_f32_to_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_cast_f32_to_f16(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    float x = _input[gid];
    half y = ((half)(x));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_linalg_cast_f16_to_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_cast_f16_to_f32(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    half x = _input[gid];
    float y = ((float)(x));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_linalg_reduce_sum_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_reduce_sum_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    half xf16 = _input[gid];
    float xf32 = ((float)(xf16));
    sdata[lid] = xf32;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint sumF16ReductionStride = 128u;
  for (int sumF16ReductionStep = 0; sumF16ReductionStep < (int)(8); ++sumF16ReductionStep) {
    uint sumF16Stride = sumF16ReductionStride;
    if (lid < sumF16Stride) {
      float other = sdata[(lid + sumF16Stride)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    uint sumF16StrideToHalve = sumF16ReductionStride;
    uint sumF16NextStride = (sumF16StrideToHalve >> 1u);
    sumF16ReductionStride = sumF16NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float partialSum = sdata[0];
    result[0] = partialSum;
  }
}

)kernel";
const char* k_metal_navatala_linalg_reduce_max_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_reduce_max_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    half xf16 = _input[gid];
    float xf32 = ((float)(xf16));
    sdata[lid] = xf32;
  } else {
    sdata[lid] = as_type<float>(0xff7ffffdu);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint maxF16ReductionStride = 128u;
  for (int maxF16ReductionStep = 0; maxF16ReductionStep < (int)(8); ++maxF16ReductionStep) {
    uint maxF16Stride = maxF16ReductionStride;
    if (lid < maxF16Stride) {
      float other = sdata[(lid + maxF16Stride)];
      float mine = sdata[lid];
      bool isGreater = (mine > other);
      float maxVal = ((isGreater) ? (mine) : (other));
      sdata[lid] = maxVal;
    }
    uint maxF16StrideToHalve = maxF16ReductionStride;
    uint maxF16NextStride = (maxF16StrideToHalve >> 1u);
    maxF16ReductionStride = maxF16NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float maxResult = sdata[0];
    result[0] = maxResult;
  }
}

)kernel";
const char* k_metal_navatala_linalg_elt_add_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_elt_add_f32(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device const uint* n [[buffer(2)]], device float* C [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint i = ((uint)(int(__gid.x)));
  if (i < n[0u]) {
    float a = A[i];
    float b = B[i];
    C[i] = (a + b);
  }
}

)kernel";
const char* k_metal_navatala_linalg_elt_sub_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_elt_sub_f32(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device const uint* n [[buffer(2)]], device float* C [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint i = ((uint)(int(__gid.x)));
  if (i < n[0u]) {
    float a = A[i];
    float b = B[i];
    C[i] = (a - b);
  }
}

)kernel";
const char* k_metal_navatala_linalg_elt_mul_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_elt_mul_f32(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device const uint* n [[buffer(2)]], device float* C [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint i = ((uint)(int(__gid.x)));
  if (i < n[0u]) {
    float a = A[i];
    float b = B[i];
    C[i] = (a * b);
  }
}

)kernel";
const char* k_metal_navatala_linalg_elt_div_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_elt_div_f32(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device const uint* n [[buffer(2)]], device float* C [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint i = ((uint)(int(__gid.x)));
  if (i < n[0u]) {
    float a = A[i];
    float b = B[i];
    C[i] = (a / b);
  }
}

)kernel";
const char* k_metal_navatala_linalg_elt_pow_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_elt_pow_f32(device const float* A [[buffer(0)]], device const float* p [[buffer(1)]], device const uint* n [[buffer(2)]], device float* C [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint i = ((uint)(int(__gid.x)));
  float pVal = p[0u];
  if (i < n[0u]) {
    float a = A[i];
    float logA = log(a);
    float pLogA = (pVal * logA);
    C[i] = exp(pLogA);
  }
}

)kernel";
const char* k_metal_navatala_linalg_elt_sqrt_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_elt_sqrt_f32(device const float* A [[buffer(0)]], device const uint* n [[buffer(1)]], device float* C [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint i = ((uint)(int(__gid.x)));
  if (i < n[0u]) {
    float a = A[i];
    C[i] = sqrt(a);
  }
}

)kernel";
const char* k_metal_navatala_linalg_elt_exp_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_elt_exp_f32(device const float* A [[buffer(0)]], device const uint* n [[buffer(1)]], device float* C [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint i = ((uint)(int(__gid.x)));
  if (i < n[0u]) {
    float a = A[i];
    C[i] = exp(a);
  }
}

)kernel";
const char* k_metal_navatala_linalg_elt_log_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_elt_log_f32(device const float* A [[buffer(0)]], device const uint* n [[buffer(1)]], device float* C [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint i = ((uint)(int(__gid.x)));
  if (i < n[0u]) {
    float a = A[i];
    C[i] = log(a);
  }
}

)kernel";
const char* k_metal_navatala_linalg_elt_abs_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_elt_abs_f32(device const float* A [[buffer(0)]], device const uint* n [[buffer(1)]], device float* C [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint i = ((uint)(int(__gid.x)));
  if (i < n[0u]) {
    float a = A[i];
    C[i] = abs(a);
  }
}

)kernel";
const char* k_metal_navatala_linalg_elt_sign_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_elt_sign_f32(device const float* A [[buffer(0)]], device const uint* n [[buffer(1)]], device float* C [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint i = ((uint)(int(__gid.x)));
  if (i < n[0u]) {
    float a = A[i];
    bool isNeg = (a < as_type<float>(0x00000000u));
    float negOrZero = ((isNeg) ? (as_type<float>(0xbf800000u)) : (as_type<float>(0x00000000u)));
    bool isPos = (a > as_type<float>(0x00000000u));
    float result = ((isPos) ? (as_type<float>(0x3f800000u)) : (negOrZero));
    C[i] = result;
  }
}

)kernel";
const char* k_metal_navatala_linalg_transpose_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_transpose_f32(device const float* A [[buffer(0)]], device const uint* m [[buffer(1)]], device const uint* n [[buffer(2)]], device float* B [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint rows = m[0];
  uint cols = n[0];
  uint totalElems = (rows * cols);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / cols);
    uint j = (gid % cols);
    uint srcIdx = ((i * cols) + j);
    uint dstIdx = ((j * rows) + i);
    float val = A[srcIdx];
    B[dstIdx] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_transpose_in_place_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_transpose_in_place_f32(device const uint* n [[buffer(0)]], device float* A [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint dim = n[0];
  uint numSwaps = ((dim * (dim - 1u)) / 2u);
  bool inBounds = (gid < numSwaps);
  if (inBounds) {
    float gidF = ((float)(gid));
    float dimF = ((float)(dim));
    float twoNm1 = ((as_type<float>(0x40000000u) * dimF) - as_type<float>(0x3f800000u));
    float disc = ((twoNm1 * twoNm1) - (as_type<float>(0x41000000u) * gidF));
    float sqrtDisc = sqrt(disc);
    float iF = floor(((twoNm1 - sqrtDisc) / as_type<float>(0x40000000u)));
    uint i = ((uint)(iF));
    uint rowStart = ((i * (((2u * dim) - i) - 1u)) / 2u);
    uint offset = (gid - rowStart);
    uint j = ((i + 1u) + offset);
    uint idx_ij = ((i * dim) + j);
    uint idx_ji = ((j * dim) + i);
    float val_ij = A[idx_ij];
    float val_ji = A[idx_ji];
    A[idx_ij] = val_ji;
    A[idx_ji] = val_ij;
  }
}

)kernel";
const char* k_metal_navatala_linalg_transpose_batched_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_transpose_batched_f32(device const float* A [[buffer(0)]], device const uint* m [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device float* B [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint rows = m[0];
  uint cols = n[0];
  uint numBatches = batchSize[0];
  uint matSize = (rows * cols);
  uint totalElems = (matSize * numBatches);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint batchIdx = (gid / matSize);
    uint localIdx = (gid % matSize);
    uint i = (localIdx / cols);
    uint j = (localIdx % cols);
    uint srcIdx = ((batchIdx * matSize) + ((i * cols) + j));
    uint dstIdx = ((batchIdx * matSize) + ((j * rows) + i));
    float val = A[srcIdx];
    B[dstIdx] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_map_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_map_f32(device const float* A [[buffer(0)]], device const uint* m [[buffer(1)]], device const uint* n [[buffer(2)]], device float* result [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint idx = gid;
  if (idx < (m[0u] * n[0u])) {
    float val = A[idx];
    result[idx] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_reduce_row_sum_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_reduce_row_sum_f32(device const float* A [[buffer(0)]], device const uint* m [[buffer(1)]], device const uint* n [[buffer(2)]], device float* result [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint row = gid;
  if (row < m[0u]) {
    float sumAccum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(n[0u]); ++j) {
      uint jU32 = ((uint)(j));
      uint idx = ((row * n[0u]) + jU32);
      float val = A[idx];
      float currentSum = sumAccum;
      float nextSum = (currentSum + val);
      sumAccum = nextSum;
    }
    float finalSum = sumAccum;
    result[row] = finalSum;
  }
}

)kernel";
const char* k_metal_navatala_linalg_reduce_col_sum_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_reduce_col_sum_f32(device const float* A [[buffer(0)]], device const uint* m [[buffer(1)]], device const uint* n [[buffer(2)]], device float* result [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint col = gid;
  if (col < n[0u]) {
    float sumAccum = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(m[0u]); ++i) {
      uint iU32 = ((uint)(i));
      uint idx = ((iU32 * n[0u]) + col);
      float val = A[idx];
      float currentSum = sumAccum;
      float nextSum = (currentSum + val);
      sumAccum = nextSum;
    }
    float finalSum = sumAccum;
    result[col] = finalSum;
  }
}

)kernel";
const char* k_metal_navatala_linalg_reduce_row_max_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_reduce_row_max_f32(device const float* A [[buffer(0)]], device const uint* m [[buffer(1)]], device const uint* n [[buffer(2)]], device float* result [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint row = gid;
  if (row < m[0u]) {
    uint firstIdx = (row * n[0u]);
    float initMaxVal = A[firstIdx];
    float maxAccum = initMaxVal;
    for (int j = 0; j < (int)(n[0u]); ++j) {
      uint jU32 = ((uint)(j));
      uint idx = ((row * n[0u]) + jU32);
      float val = A[idx];
      float currentMax = maxAccum;
      float nextMax = (((val > currentMax)) ? (val) : (currentMax));
      maxAccum = nextMax;
    }
    float finalMax = maxAccum;
    result[row] = finalMax;
  }
}

)kernel";
const char* k_metal_navatala_linalg_reduce_col_max_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_reduce_col_max_f32(device const float* A [[buffer(0)]], device const uint* m [[buffer(1)]], device const uint* n [[buffer(2)]], device float* result [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint col = gid;
  if (col < n[0u]) {
    float initMaxVal = A[col];
    float maxAccum = initMaxVal;
    for (int i = 0; i < (int)(m[0u]); ++i) {
      uint iU32 = ((uint)(i));
      uint idx = ((iU32 * n[0u]) + col);
      float val = A[idx];
      float currentMax = maxAccum;
      float nextMax = (((val > currentMax)) ? (val) : (currentMax));
      maxAccum = nextMax;
    }
    float finalMax = maxAccum;
    result[col] = finalMax;
  }
}

)kernel";
const char* k_metal_navatala_linalg_map_reduce_sum_f32 = R"kernel(
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

kernel void navatala_linalg_map_reduce_sum_f32(device const float* A [[buffer(0)]], device const uint* m [[buffer(1)]], device const uint* n [[buffer(2)]], device float* result [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint idx = gid;
  if (idx < (m[0u] * n[0u])) {
    float val = A[idx];
    atomic_fetch_add_explicit((device atomic_float*)(&result[0u]), val, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_linalg_outer_product_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_outer_product_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* m [[buffer(2)]], device const uint* n [[buffer(3)]], device float* C [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint rows = m[0u];
  uint cols = n[0u];
  uint totalElems = (rows * cols);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / cols);
    uint j = (gid % cols);
    float aVal = a[i];
    float bVal = b[j];
    uint outIdx = ((i * cols) + j);
    float product = (aVal * bVal);
    C[outIdx] = product;
  }
}

)kernel";
const char* k_metal_navatala_linalg_batched_dot_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_batched_dot_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* vecLen [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device float* result [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint len = vecLen[0u];
  uint numBatches = batchSize[0u];
  uint k = gid;
  bool inBounds = (k < numBatches);
  if (inBounds) {
    uint baseOffset = (k * len);
    float sumAccum = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(len); ++i) {
      uint iU32 = ((uint)(i));
      uint idx = (baseOffset + iU32);
      float aVal = a[idx];
      float bVal = b[idx];
      float prod = (aVal * bVal);
      float currentSum = sumAccum;
      float nextSum = (currentSum + prod);
      sumAccum = nextSum;
    }
    float finalSum = sumAccum;
    result[k] = finalSum;
  }
}

)kernel";
const char* k_metal_navatala_linalg_trace_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_trace_f32(device const float* A [[buffer(0)]], device const uint* n [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint dim = n[0u];
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    float traceAccum = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(dim); ++i) {
      uint iU32 = ((uint)(i));
      uint diagIdx = ((iU32 * dim) + iU32);
      float diagVal = A[diagIdx];
      float currentTrace = traceAccum;
      float nextTrace = (currentTrace + diagVal);
      traceAccum = nextTrace;
    }
    float finalTrace = traceAccum;
    result[0u] = finalTrace;
  }
}

)kernel";
const char* k_metal_navatala_linalg_frobenius_norm_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_frobenius_norm_f32(device const float* A [[buffer(0)]], device const uint* m [[buffer(1)]], device const uint* n [[buffer(2)]], device float* result [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint rows = m[0u];
  uint cols = n[0u];
  uint totalElems = (rows * cols);
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    float sumSqAccum = as_type<float>(0x00000000u);
    for (int idx = 0; idx < (int)(totalElems); ++idx) {
      uint idxU32 = ((uint)(idx));
      float val = A[idxU32];
      float valSq = (val * val);
      float currentSumSq = sumSqAccum;
      float nextSumSq = (currentSumSq + valSq);
      sumSqAccum = nextSumSq;
    }
    float finalSumSq = sumSqAccum;
    float norm = sqrt(finalSumSq);
    result[0u] = norm;
  }
}

)kernel";
const char* k_metal_navatala_linalg_extract_upper_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_extract_upper_f32(device const float* A [[buffer(0)]], device const uint* n [[buffer(1)]], device float* B [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint idx = ((uint)(int(__gid.x)));
  uint nVal = n[0u];
  uint totalElements = (nVal * nVal);
  if (idx < totalElements) {
    uint i = (idx / nVal);
    uint j = (idx % nVal);
    float aVal = A[idx];
    bool isUpper = (i <= j);
    float result = ((isUpper) ? (aVal) : (as_type<float>(0x00000000u)));
    B[idx] = result;
  }
}

)kernel";
const char* k_metal_navatala_linalg_extract_lower_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_extract_lower_f32(device const float* A [[buffer(0)]], device const uint* n [[buffer(1)]], device float* B [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint idx = ((uint)(int(__gid.x)));
  uint nVal = n[0u];
  uint totalElements = (nVal * nVal);
  if (idx < totalElements) {
    uint i = (idx / nVal);
    uint j = (idx % nVal);
    float aVal = A[idx];
    bool isLower = (i >= j);
    float result = ((isLower) ? (aVal) : (as_type<float>(0x00000000u)));
    B[idx] = result;
  }
}

)kernel";
const char* k_metal_navatala_linalg_set_upper_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_set_upper_f32(device const float* val [[buffer(0)]], device const uint* n [[buffer(1)]], device float* A [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint idx = ((uint)(int(__gid.x)));
  uint nVal = n[0u];
  float fillVal = val[0u];
  uint totalElements = (nVal * nVal);
  if (idx < totalElements) {
    uint i = (idx / nVal);
    uint j = (idx % nVal);
    bool isStrictUpper = (i < j);
    if (isStrictUpper) {
      A[idx] = fillVal;
    }
  }
}

)kernel";
const char* k_metal_navatala_linalg_set_lower_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_set_lower_f32(device const float* val [[buffer(0)]], device const uint* n [[buffer(1)]], device float* A [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint idx = ((uint)(int(__gid.x)));
  uint nVal = n[0u];
  float fillVal = val[0u];
  uint totalElements = (nVal * nVal);
  if (idx < totalElements) {
    uint i = (idx / nVal);
    uint j = (idx % nVal);
    bool isStrictLower = (i > j);
    if (isStrictLower) {
      A[idx] = fillVal;
    }
  }
}

)kernel";
const char* k_metal_navatala_linalg_gather_rows_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_gather_rows_f32(device const float* A [[buffer(0)]], device const uint* indices [[buffer(1)]], device const uint* k [[buffer(2)]], device const uint* n [[buffer(3)]], device float* _out [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint tid = gid;
  uint kVal = k[0u];
  uint nVal = n[0u];
  uint totalElems = (kVal * nVal);
  if (tid < totalElems) {
    uint rowIdx = (tid / nVal);
    uint colIdx = (tid % nVal);
    uint srcRow = indices[rowIdx];
    uint srcIdx = ((srcRow * nVal) + colIdx);
    float val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_gather_cols_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_gather_cols_f32(device const float* A [[buffer(0)]], device const uint* indices [[buffer(1)]], device const uint* m [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* kCount [[buffer(4)]], device float* _out [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint tid = gid;
  uint mVal = m[0u];
  uint nVal = n[0u];
  uint kVal = kCount[0u];
  uint totalElems = (mVal * kVal);
  if (tid < totalElems) {
    uint rowIdx = (tid / kVal);
    uint outColIdx = (tid % kVal);
    uint srcCol = indices[outColIdx];
    uint srcIdx = ((rowIdx * nVal) + srcCol);
    float val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_gather_elements_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_gather_elements_f32(device const float* A [[buffer(0)]], device const uint* rowIdx [[buffer(1)]], device const uint* colIdx [[buffer(2)]], device const uint* count [[buffer(3)]], device const uint* n [[buffer(4)]], device float* _out [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint tid = gid;
  uint countVal = count[0u];
  uint nVal = n[0u];
  if (tid < countVal) {
    uint row = rowIdx[tid];
    uint col = colIdx[tid];
    uint srcIdx = ((row * nVal) + col);
    float val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_gather_batched_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_gather_batched_f32(device const float* A [[buffer(0)]], device const uint* indices [[buffer(1)]], device const uint* batchSize [[buffer(2)]], device const uint* m [[buffer(3)]], device const uint* n [[buffer(4)]], device const uint* kCount [[buffer(5)]], device float* _out [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint tid = gid;
  uint batchVal = batchSize[0u];
  uint mVal = m[0u];
  uint nVal = n[0u];
  uint kVal = kCount[0u];
  uint outBatchStride = (kVal * nVal);
  uint totalElems = (batchVal * outBatchStride);
  if (tid < totalElems) {
    uint batchIdx = (tid / outBatchStride);
    uint withinBatch = (tid % outBatchStride);
    uint rowIdx = (withinBatch / nVal);
    uint colIdx = (withinBatch % nVal);
    uint indicesIdx = ((batchIdx * kVal) + rowIdx);
    uint srcRow = indices[indicesIdx];
    uint srcBatchStride = (mVal * nVal);
    uint srcIdx = (((batchIdx * srcBatchStride) + (srcRow * nVal)) + colIdx);
    float val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_scatter_rows_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_scatter_rows_f32(device const float* src [[buffer(0)]], device const uint* indices [[buffer(1)]], device const uint* k [[buffer(2)]], device const uint* n [[buffer(3)]], device float* A [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint tid = gid;
  uint kVal = k[0u];
  uint nVal = n[0u];
  uint totalElems = (kVal * nVal);
  if (tid < totalElems) {
    uint rowIdx = (tid / nVal);
    uint colIdx = (tid % nVal);
    uint dstRow = indices[rowIdx];
    uint dstIdx = ((dstRow * nVal) + colIdx);
    float val = src[tid];
    A[dstIdx] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_scatter_cols_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_scatter_cols_f32(device const float* src [[buffer(0)]], device const uint* indices [[buffer(1)]], device const uint* m [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* n [[buffer(4)]], device float* A [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint tid = gid;
  uint mVal = m[0u];
  uint kVal = k[0u];
  uint nVal = n[0u];
  uint totalElems = (mVal * kVal);
  if (tid < totalElems) {
    uint row = (tid / kVal);
    uint srcColIdx = (tid % kVal);
    uint dstCol = indices[srcColIdx];
    uint dstIdx = ((row * nVal) + dstCol);
    float val = src[tid];
    A[dstIdx] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_scatter_add_f32 = R"kernel(
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

kernel void navatala_linalg_scatter_add_f32(device const float* src [[buffer(0)]], device const uint* indices [[buffer(1)]], device const uint* count [[buffer(2)]], device float* A [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint tid = gid;
  uint n = count[0u];
  if (tid < n) {
    uint dstIdx = indices[tid];
    float val = src[tid];
    atomic_fetch_add_explicit((device atomic_float*)(&A[dstIdx]), val, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_linalg_scatter_max_f32 = R"kernel(
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

static inline float gpu_atomic_min_float(device atomic_float* ptr, float value) {
  float cur = atomic_load_explicit(ptr, memory_order_relaxed);
  while (value < cur) {
    if (atomic_compare_exchange_weak_explicit(ptr, &cur, value,
        memory_order_relaxed, memory_order_relaxed)) { return value; }
  }
  return cur;
}

static inline float gpu_atomic_max_float(device atomic_float* ptr, float value) {
  float cur = atomic_load_explicit(ptr, memory_order_relaxed);
  while (value > cur) {
    if (atomic_compare_exchange_weak_explicit(ptr, &cur, value,
        memory_order_relaxed, memory_order_relaxed)) { return value; }
  }
  return cur;
}

kernel void navatala_linalg_scatter_max_f32(device const float* src [[buffer(0)]], device const uint* indices [[buffer(1)]], device const uint* count [[buffer(2)]], device float* A [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint tid = gid;
  uint n = count[0u];
  if (tid < n) {
    uint dstIdx = indices[tid];
    float val = src[tid];
    gpu_atomic_max_float((device atomic_float*)(&A[dstIdx]), val);
  }
}

)kernel";
const char* k_metal_navatala_linalg_slice_rows_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_slice_rows_f32(device const float* A [[buffer(0)]], device const uint* rowStart [[buffer(1)]], device const uint* rowEnd [[buffer(2)]], device const uint* numCols [[buffer(3)]], device float* _out [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint idx = gid;
  uint rs = rowStart[0u];
  uint re = rowEnd[0u];
  uint n = numCols[0u];
  uint numRows = (re - rs);
  uint totalElems = (numRows * n);
  if (idx < totalElems) {
    uint outRow = (idx / n);
    uint outCol = (idx % n);
    uint srcRow = (rs + outRow);
    uint srcIdx = ((srcRow * n) + outCol);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_slice_cols_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_slice_cols_f32(device const float* A [[buffer(0)]], device const uint* colStart [[buffer(1)]], device const uint* colEnd [[buffer(2)]], device const uint* numRows [[buffer(3)]], device const uint* srcCols [[buffer(4)]], device float* _out [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint idx = gid;
  uint cs = colStart[0u];
  uint ce = colEnd[0u];
  uint m = numRows[0u];
  uint n = srcCols[0u];
  uint outCols = (ce - cs);
  uint totalElems = (m * outCols);
  if (idx < totalElems) {
    uint outRow = (idx / outCols);
    uint outCol = (idx % outCols);
    uint srcCol = (cs + outCol);
    uint srcIdx = ((outRow * n) + srcCol);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_slice_block_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_slice_block_f32(device const float* A [[buffer(0)]], device const uint* rowStart [[buffer(1)]], device const uint* rowEnd [[buffer(2)]], device const uint* colStart [[buffer(3)]], device const uint* colEnd [[buffer(4)]], device const uint* srcCols [[buffer(5)]], device float* _out [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint idx = gid;
  uint rs = rowStart[0u];
  uint re = rowEnd[0u];
  uint cs = colStart[0u];
  uint ce = colEnd[0u];
  uint n = srcCols[0u];
  uint outRows = (re - rs);
  uint outCols = (ce - cs);
  uint totalElems = (outRows * outCols);
  if (idx < totalElems) {
    uint outRow = (idx / outCols);
    uint outCol = (idx % outCols);
    uint srcRow = (rs + outRow);
    uint srcCol = (cs + outCol);
    uint srcIdx = ((srcRow * n) + srcCol);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_strided_slice_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_strided_slice_f32(device const float* A [[buffer(0)]], device const uint* start [[buffer(1)]], device const uint* stride [[buffer(2)]], device const uint* count [[buffer(3)]], device float* _out [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint idx = gid;
  uint s = start[0u];
  uint st = stride[0u];
  uint n = count[0u];
  if (idx < n) {
    uint srcIdx = (s + (idx * st));
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_extract_diagonal_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_extract_diagonal_f32(device const float* A [[buffer(0)]], device const uint* n [[buffer(1)]], device float* _out [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint idx = gid;
  uint size = n[0u];
  if (idx < size) {
    uint stride = (size + 1u);
    uint srcIdx = (idx * stride);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_extract_diagonal_k_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_extract_diagonal_k_f32(device const float* A [[buffer(0)]], device const uint* n [[buffer(1)]], device const uint* k [[buffer(2)]], device float* _out [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint idx = gid;
  uint size = n[0u];
  uint offset = k[0u];
  uint diagLen = (size - offset);
  if (idx < diagLen) {
    uint row = idx;
    uint col = (idx + offset);
    uint srcIdx = ((row * size) + col);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_set_diagonal_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_set_diagonal_f32(device const float* val [[buffer(0)]], device const uint* n [[buffer(1)]], device float* A [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint idx = gid;
  uint size = n[0u];
  if (idx < size) {
    uint stride = (size + 1u);
    uint destIdx = (idx * stride);
    float v = val[idx];
    A[destIdx] = v;
  }
}

)kernel";
const char* k_metal_navatala_linalg_diag_to_matrix_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_diag_to_matrix_f32(device const float* v [[buffer(0)]], device const uint* n [[buffer(1)]], device float* A [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint idx = gid;
  uint size = n[0u];
  uint totalElems = (size * size);
  if (idx < totalElems) {
    uint row = (idx / size);
    uint col = (idx % size);
    bool isDiag = (row == col);
    float diagVal = v[row];
    float result = ((isDiag) ? (diagVal) : (as_type<float>(0x00000000u)));
    A[idx] = result;
  }
}

)kernel";
const char* k_metal_navatala_linalg_shift_rows_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_shift_rows_f32(device const float* A [[buffer(0)]], device const uint* k [[buffer(1)]], device const uint* m [[buffer(2)]], device const uint* n [[buffer(3)]], device float* _out [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint idx = gid;
  uint kVal = k[0u];
  uint mVal = m[0u];
  uint nVal = n[0u];
  uint totalElems = (mVal * nVal);
  if (idx < totalElems) {
    uint row = (idx / nVal);
    uint col = (idx % nVal);
    bool inBounds = (row >= kVal);
    uint srcRow = (row - kVal);
    uint srcIdx = ((srcRow * nVal) + col);
    float srcVal = A[srcIdx];
    float result = ((inBounds) ? (srcVal) : (as_type<float>(0x00000000u)));
    _out[idx] = result;
  }
}

)kernel";
const char* k_metal_navatala_linalg_shift_cols_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_shift_cols_f32(device const float* A [[buffer(0)]], device const uint* k [[buffer(1)]], device const uint* m [[buffer(2)]], device const uint* n [[buffer(3)]], device float* _out [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint idx = gid;
  uint kVal = k[0u];
  uint mVal = m[0u];
  uint nVal = n[0u];
  uint totalElems = (mVal * nVal);
  if (idx < totalElems) {
    uint row = (idx / nVal);
    uint col = (idx % nVal);
    bool inBounds = (col >= kVal);
    uint srcCol = (col - kVal);
    uint srcIdx = ((row * nVal) + srcCol);
    float srcVal = A[srcIdx];
    float result = ((inBounds) ? (srcVal) : (as_type<float>(0x00000000u)));
    _out[idx] = result;
  }
}

)kernel";
const char* k_metal_navatala_linalg_roll_rows_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_roll_rows_f32(device const float* A [[buffer(0)]], device const uint* k [[buffer(1)]], device const uint* m [[buffer(2)]], device const uint* n [[buffer(3)]], device float* _out [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint idx = gid;
  uint kVal = k[0u];
  uint mVal = m[0u];
  uint nVal = n[0u];
  uint totalElems = (mVal * nVal);
  if (idx < totalElems) {
    uint row = (idx / nVal);
    uint col = (idx % nVal);
    uint kMod = (kVal % mVal);
    uint rowPlusM = (row + mVal);
    uint rowMinusK = (rowPlusM - kMod);
    uint srcRow = (rowMinusK % mVal);
    uint srcIdx = ((srcRow * nVal) + col);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_roll_cols_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_roll_cols_f32(device const float* A [[buffer(0)]], device const uint* k [[buffer(1)]], device const uint* m [[buffer(2)]], device const uint* n [[buffer(3)]], device float* _out [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint idx = gid;
  uint kVal = k[0u];
  uint mVal = m[0u];
  uint nVal = n[0u];
  uint totalElems = (mVal * nVal);
  if (idx < totalElems) {
    uint row = (idx / nVal);
    uint col = (idx % nVal);
    uint kMod = (kVal % nVal);
    uint colPlusN = (col + nVal);
    uint colMinusK = (colPlusN - kMod);
    uint srcCol = (colMinusK % nVal);
    uint srcIdx = ((row * nVal) + srcCol);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_metal_navatala_linalg_sort_rows_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_sort_rows_f32(device const float* A [[buffer(0)]], device const uint* m [[buffer(1)]], device const uint* n [[buffer(2)]], device float* _out [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  threadgroup float svals[256];
  uint loopMyParity = 0u;
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  float loopMyVal = as_type<float>(0x00000000u);
  uint loopPartnerIdx = 0u;
  float loopPartnerVal = as_type<float>(0x00000000u);
  uint rowIdx = ((uint)(int(__tgid.x)));
  uint colIdx = ((uint)(int(__tid.x)));
  uint mVal = m[0u];
  uint nVal = n[0u];
  if (rowIdx < mVal) {
    bool inBounds = (colIdx < nVal);
    uint linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      float val = A[linearIdx];
      svals[colIdx] = val;
    } else {
      svals[colIdx] = as_type<float>(0x7f7fffffu);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int phase = 0; phase < (int)(nVal); ++phase) {
      uint phaseU32 = ((uint)(phase));
      loopMyParity = (colIdx % 2u);
      uint myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % 2u));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((colIdx + 1u) < nVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[colIdx];
        float myValRead = loopMyVal;
        loopPartnerIdx = (colIdx + 1u);
        uint partnerIdxVal = loopPartnerIdx;
        loopPartnerVal = svals[partnerIdxVal];
        float partnerValRead = loopPartnerVal;
        if (myValRead > partnerValRead) {
          svals[colIdx] = partnerValRead;
          svals[partnerIdxVal] = myValRead;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (inBounds) {
      float sortedVal = svals[colIdx];
      _out[linearIdx] = sortedVal;
    }
  }
}

)kernel";
const char* k_metal_navatala_linalg_sort_cols_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_sort_cols_f32(device const float* A [[buffer(0)]], device const uint* m [[buffer(1)]], device const uint* n [[buffer(2)]], device float* _out [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  threadgroup float svals[256];
  uint loopMyParity = 0u;
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  float loopMyVal = as_type<float>(0x00000000u);
  uint loopPartnerIdx = 0u;
  float loopPartnerVal = as_type<float>(0x00000000u);
  uint colIdx = ((uint)(int(__tgid.x)));
  uint rowIdx = ((uint)(int(__tid.x)));
  uint mVal = m[0u];
  uint nVal = n[0u];
  if (colIdx < nVal) {
    bool inBounds = (rowIdx < mVal);
    uint linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      float val = A[linearIdx];
      svals[rowIdx] = val;
    } else {
      svals[rowIdx] = as_type<float>(0x7f7fffffu);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int phase = 0; phase < (int)(mVal); ++phase) {
      uint phaseU32 = ((uint)(phase));
      loopMyParity = (rowIdx % 2u);
      uint myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % 2u));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((rowIdx + 1u) < mVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[rowIdx];
        float myValRead = loopMyVal;
        loopPartnerIdx = (rowIdx + 1u);
        uint partnerIdxVal = loopPartnerIdx;
        loopPartnerVal = svals[partnerIdxVal];
        float partnerValRead = loopPartnerVal;
        if (myValRead > partnerValRead) {
          svals[rowIdx] = partnerValRead;
          svals[partnerIdxVal] = myValRead;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (inBounds) {
      float sortedVal = svals[rowIdx];
      _out[linearIdx] = sortedVal;
    }
  }
}

)kernel";
const char* k_metal_navatala_linalg_argsort_rows_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_argsort_rows_f32(device const float* A [[buffer(0)]], device const uint* m [[buffer(1)]], device const uint* n [[buffer(2)]], device uint* _out [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  threadgroup float svals[256];
  threadgroup uint sidxs[256];
  uint loopMyParity = 0u;
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  float loopMyVal = as_type<float>(0x00000000u);
  uint loopMyIdx = 0u;
  uint loopPartnerPos = 0u;
  float loopPartnerVal = as_type<float>(0x00000000u);
  uint loopPartnerIdx = 0u;
  uint rowIdx = ((uint)(int(__tgid.x)));
  uint colIdx = ((uint)(int(__tid.x)));
  uint mVal = m[0u];
  uint nVal = n[0u];
  if (rowIdx < mVal) {
    bool inBounds = (colIdx < nVal);
    uint linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      float val = A[linearIdx];
      svals[colIdx] = val;
      sidxs[colIdx] = colIdx;
    } else {
      svals[colIdx] = as_type<float>(0x7f7fffffu);
      sidxs[colIdx] = colIdx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int phase = 0; phase < (int)(nVal); ++phase) {
      uint phaseU32 = ((uint)(phase));
      loopMyParity = (colIdx % 2u);
      uint myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % 2u));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((colIdx + 1u) < nVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[colIdx];
        float myValRead = loopMyVal;
        loopMyIdx = sidxs[colIdx];
        uint myIdxRead = loopMyIdx;
        loopPartnerPos = (colIdx + 1u);
        uint partnerPosVal = loopPartnerPos;
        loopPartnerVal = svals[partnerPosVal];
        float partnerValRead = loopPartnerVal;
        loopPartnerIdx = sidxs[partnerPosVal];
        uint partnerIdxRead = loopPartnerIdx;
        if (myValRead > partnerValRead) {
          svals[colIdx] = partnerValRead;
          sidxs[colIdx] = partnerIdxRead;
          svals[partnerPosVal] = myValRead;
          sidxs[partnerPosVal] = myIdxRead;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (inBounds) {
      uint sortedIdx = sidxs[colIdx];
      _out[linearIdx] = sortedIdx;
    }
  }
}

)kernel";
const char* k_metal_navatala_linalg_argsort_cols_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_linalg_argsort_cols_f32(device const float* A [[buffer(0)]], device const uint* m [[buffer(1)]], device const uint* n [[buffer(2)]], device uint* _out [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  threadgroup float svals[256];
  threadgroup uint sidxs[256];
  uint loopMyParity = 0u;
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  float loopMyVal = as_type<float>(0x00000000u);
  uint loopMyIdx = 0u;
  uint loopPartnerPos = 0u;
  float loopPartnerVal = as_type<float>(0x00000000u);
  uint loopPartnerIdx = 0u;
  uint colIdx = ((uint)(int(__tgid.x)));
  uint rowIdx = ((uint)(int(__tid.x)));
  uint mVal = m[0u];
  uint nVal = n[0u];
  if (colIdx < nVal) {
    bool inBounds = (rowIdx < mVal);
    uint linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      float val = A[linearIdx];
      svals[rowIdx] = val;
      sidxs[rowIdx] = rowIdx;
    } else {
      svals[rowIdx] = as_type<float>(0x7f7fffffu);
      sidxs[rowIdx] = rowIdx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int phase = 0; phase < (int)(mVal); ++phase) {
      uint phaseU32 = ((uint)(phase));
      loopMyParity = (rowIdx % 2u);
      uint myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % 2u));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((rowIdx + 1u) < mVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[rowIdx];
        float myValRead = loopMyVal;
        loopMyIdx = sidxs[rowIdx];
        uint myIdxRead = loopMyIdx;
        loopPartnerPos = (rowIdx + 1u);
        uint partnerPosVal = loopPartnerPos;
        loopPartnerVal = svals[partnerPosVal];
        float partnerValRead = loopPartnerVal;
        loopPartnerIdx = sidxs[partnerPosVal];
        uint partnerIdxRead = loopPartnerIdx;
        if (myValRead > partnerValRead) {
          svals[rowIdx] = partnerValRead;
          sidxs[rowIdx] = partnerIdxRead;
          svals[partnerPosVal] = myValRead;
          sidxs[partnerPosVal] = myIdxRead;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (inBounds) {
      uint sortedIdx = sidxs[rowIdx];
      _out[linearIdx] = sortedIdx;
    }
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_norm_linf_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_norm_linf_f32 = {
  1,
  "navatala_linalg_norm_linf_f32",
  "metal",
  "navatala_linalg_norm_linf_f32",
  "kernel:metal:navatala_linalg_norm_linf_f32",
  "abi-r1:metal:navatala_linalg_norm_linf_f32",
  "abi-r1:metal:navatala_linalg_norm_linf_f32",
  3,
  kAbiArgs_metal_navatala_linalg_norm_linf_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_eig_select_top_k_f32[] = {
  { "eigenvalues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "eigenvectors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "topKValues", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "topKVectors", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_eig_select_top_k_f32 = {
  1,
  "navatala_linalg_eig_select_top_k_f32",
  "metal",
  "navatala_linalg_eig_select_top_k_f32",
  "kernel:metal:navatala_linalg_eig_select_top_k_f32",
  "abi-r1:metal:navatala_linalg_eig_select_top_k_f32",
  "abi-r1:metal:navatala_linalg_eig_select_top_k_f32",
  6,
  kAbiArgs_metal_navatala_linalg_eig_select_top_k_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_cast_f32_to_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_cast_f32_to_f16 = {
  1,
  "navatala_linalg_cast_f32_to_f16",
  "metal",
  "navatala_linalg_cast_f32_to_f16",
  "kernel:metal:navatala_linalg_cast_f32_to_f16",
  "abi-r1:metal:navatala_linalg_cast_f32_to_f16",
  "abi-r1:metal:navatala_linalg_cast_f32_to_f16",
  3,
  kAbiArgs_metal_navatala_linalg_cast_f32_to_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_cast_f16_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_cast_f16_to_f32 = {
  1,
  "navatala_linalg_cast_f16_to_f32",
  "metal",
  "navatala_linalg_cast_f16_to_f32",
  "kernel:metal:navatala_linalg_cast_f16_to_f32",
  "abi-r1:metal:navatala_linalg_cast_f16_to_f32",
  "abi-r1:metal:navatala_linalg_cast_f16_to_f32",
  3,
  kAbiArgs_metal_navatala_linalg_cast_f16_to_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_reduce_sum_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_reduce_sum_f16 = {
  1,
  "navatala_linalg_reduce_sum_f16",
  "metal",
  "navatala_linalg_reduce_sum_f16",
  "kernel:metal:navatala_linalg_reduce_sum_f16",
  "abi-r1:metal:navatala_linalg_reduce_sum_f16",
  "abi-r1:metal:navatala_linalg_reduce_sum_f16",
  3,
  kAbiArgs_metal_navatala_linalg_reduce_sum_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_reduce_max_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_reduce_max_f16 = {
  1,
  "navatala_linalg_reduce_max_f16",
  "metal",
  "navatala_linalg_reduce_max_f16",
  "kernel:metal:navatala_linalg_reduce_max_f16",
  "abi-r1:metal:navatala_linalg_reduce_max_f16",
  "abi-r1:metal:navatala_linalg_reduce_max_f16",
  3,
  kAbiArgs_metal_navatala_linalg_reduce_max_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_elt_add_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_elt_add_f32 = {
  1,
  "navatala_linalg_elt_add_f32",
  "metal",
  "navatala_linalg_elt_add_f32",
  "kernel:metal:navatala_linalg_elt_add_f32",
  "abi-r1:metal:navatala_linalg_elt_add_f32",
  "abi-r1:metal:navatala_linalg_elt_add_f32",
  4,
  kAbiArgs_metal_navatala_linalg_elt_add_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_elt_sub_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_elt_sub_f32 = {
  1,
  "navatala_linalg_elt_sub_f32",
  "metal",
  "navatala_linalg_elt_sub_f32",
  "kernel:metal:navatala_linalg_elt_sub_f32",
  "abi-r1:metal:navatala_linalg_elt_sub_f32",
  "abi-r1:metal:navatala_linalg_elt_sub_f32",
  4,
  kAbiArgs_metal_navatala_linalg_elt_sub_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_elt_mul_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_elt_mul_f32 = {
  1,
  "navatala_linalg_elt_mul_f32",
  "metal",
  "navatala_linalg_elt_mul_f32",
  "kernel:metal:navatala_linalg_elt_mul_f32",
  "abi-r1:metal:navatala_linalg_elt_mul_f32",
  "abi-r1:metal:navatala_linalg_elt_mul_f32",
  4,
  kAbiArgs_metal_navatala_linalg_elt_mul_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_elt_div_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_elt_div_f32 = {
  1,
  "navatala_linalg_elt_div_f32",
  "metal",
  "navatala_linalg_elt_div_f32",
  "kernel:metal:navatala_linalg_elt_div_f32",
  "abi-r1:metal:navatala_linalg_elt_div_f32",
  "abi-r1:metal:navatala_linalg_elt_div_f32",
  4,
  kAbiArgs_metal_navatala_linalg_elt_div_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_elt_pow_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "p", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_elt_pow_f32 = {
  1,
  "navatala_linalg_elt_pow_f32",
  "metal",
  "navatala_linalg_elt_pow_f32",
  "kernel:metal:navatala_linalg_elt_pow_f32",
  "abi-r1:metal:navatala_linalg_elt_pow_f32",
  "abi-r1:metal:navatala_linalg_elt_pow_f32",
  4,
  kAbiArgs_metal_navatala_linalg_elt_pow_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_elt_sqrt_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_elt_sqrt_f32 = {
  1,
  "navatala_linalg_elt_sqrt_f32",
  "metal",
  "navatala_linalg_elt_sqrt_f32",
  "kernel:metal:navatala_linalg_elt_sqrt_f32",
  "abi-r1:metal:navatala_linalg_elt_sqrt_f32",
  "abi-r1:metal:navatala_linalg_elt_sqrt_f32",
  3,
  kAbiArgs_metal_navatala_linalg_elt_sqrt_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_elt_exp_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_elt_exp_f32 = {
  1,
  "navatala_linalg_elt_exp_f32",
  "metal",
  "navatala_linalg_elt_exp_f32",
  "kernel:metal:navatala_linalg_elt_exp_f32",
  "abi-r1:metal:navatala_linalg_elt_exp_f32",
  "abi-r1:metal:navatala_linalg_elt_exp_f32",
  3,
  kAbiArgs_metal_navatala_linalg_elt_exp_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_elt_log_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_elt_log_f32 = {
  1,
  "navatala_linalg_elt_log_f32",
  "metal",
  "navatala_linalg_elt_log_f32",
  "kernel:metal:navatala_linalg_elt_log_f32",
  "abi-r1:metal:navatala_linalg_elt_log_f32",
  "abi-r1:metal:navatala_linalg_elt_log_f32",
  3,
  kAbiArgs_metal_navatala_linalg_elt_log_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_elt_abs_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_elt_abs_f32 = {
  1,
  "navatala_linalg_elt_abs_f32",
  "metal",
  "navatala_linalg_elt_abs_f32",
  "kernel:metal:navatala_linalg_elt_abs_f32",
  "abi-r1:metal:navatala_linalg_elt_abs_f32",
  "abi-r1:metal:navatala_linalg_elt_abs_f32",
  3,
  kAbiArgs_metal_navatala_linalg_elt_abs_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_elt_sign_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_elt_sign_f32 = {
  1,
  "navatala_linalg_elt_sign_f32",
  "metal",
  "navatala_linalg_elt_sign_f32",
  "kernel:metal:navatala_linalg_elt_sign_f32",
  "abi-r1:metal:navatala_linalg_elt_sign_f32",
  "abi-r1:metal:navatala_linalg_elt_sign_f32",
  3,
  kAbiArgs_metal_navatala_linalg_elt_sign_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_transpose_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_transpose_f32 = {
  1,
  "navatala_linalg_transpose_f32",
  "metal",
  "navatala_linalg_transpose_f32",
  "kernel:metal:navatala_linalg_transpose_f32",
  "abi-r1:metal:navatala_linalg_transpose_f32",
  "abi-r1:metal:navatala_linalg_transpose_f32",
  4,
  kAbiArgs_metal_navatala_linalg_transpose_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_transpose_in_place_f32[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_transpose_in_place_f32 = {
  1,
  "navatala_linalg_transpose_in_place_f32",
  "metal",
  "navatala_linalg_transpose_in_place_f32",
  "kernel:metal:navatala_linalg_transpose_in_place_f32",
  "abi-r1:metal:navatala_linalg_transpose_in_place_f32",
  "abi-r1:metal:navatala_linalg_transpose_in_place_f32",
  2,
  kAbiArgs_metal_navatala_linalg_transpose_in_place_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_transpose_batched_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_transpose_batched_f32 = {
  1,
  "navatala_linalg_transpose_batched_f32",
  "metal",
  "navatala_linalg_transpose_batched_f32",
  "kernel:metal:navatala_linalg_transpose_batched_f32",
  "abi-r1:metal:navatala_linalg_transpose_batched_f32",
  "abi-r1:metal:navatala_linalg_transpose_batched_f32",
  5,
  kAbiArgs_metal_navatala_linalg_transpose_batched_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_map_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_map_f32 = {
  1,
  "navatala_linalg_map_f32",
  "metal",
  "navatala_linalg_map_f32",
  "kernel:metal:navatala_linalg_map_f32",
  "abi-r1:metal:navatala_linalg_map_f32",
  "abi-r1:metal:navatala_linalg_map_f32",
  4,
  kAbiArgs_metal_navatala_linalg_map_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_reduce_row_sum_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_reduce_row_sum_f32 = {
  1,
  "navatala_linalg_reduce_row_sum_f32",
  "metal",
  "navatala_linalg_reduce_row_sum_f32",
  "kernel:metal:navatala_linalg_reduce_row_sum_f32",
  "abi-r1:metal:navatala_linalg_reduce_row_sum_f32",
  "abi-r1:metal:navatala_linalg_reduce_row_sum_f32",
  4,
  kAbiArgs_metal_navatala_linalg_reduce_row_sum_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_reduce_col_sum_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_reduce_col_sum_f32 = {
  1,
  "navatala_linalg_reduce_col_sum_f32",
  "metal",
  "navatala_linalg_reduce_col_sum_f32",
  "kernel:metal:navatala_linalg_reduce_col_sum_f32",
  "abi-r1:metal:navatala_linalg_reduce_col_sum_f32",
  "abi-r1:metal:navatala_linalg_reduce_col_sum_f32",
  4,
  kAbiArgs_metal_navatala_linalg_reduce_col_sum_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_reduce_row_max_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_reduce_row_max_f32 = {
  1,
  "navatala_linalg_reduce_row_max_f32",
  "metal",
  "navatala_linalg_reduce_row_max_f32",
  "kernel:metal:navatala_linalg_reduce_row_max_f32",
  "abi-r1:metal:navatala_linalg_reduce_row_max_f32",
  "abi-r1:metal:navatala_linalg_reduce_row_max_f32",
  4,
  kAbiArgs_metal_navatala_linalg_reduce_row_max_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_reduce_col_max_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_reduce_col_max_f32 = {
  1,
  "navatala_linalg_reduce_col_max_f32",
  "metal",
  "navatala_linalg_reduce_col_max_f32",
  "kernel:metal:navatala_linalg_reduce_col_max_f32",
  "abi-r1:metal:navatala_linalg_reduce_col_max_f32",
  "abi-r1:metal:navatala_linalg_reduce_col_max_f32",
  4,
  kAbiArgs_metal_navatala_linalg_reduce_col_max_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_map_reduce_sum_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_map_reduce_sum_f32 = {
  1,
  "navatala_linalg_map_reduce_sum_f32",
  "metal",
  "navatala_linalg_map_reduce_sum_f32",
  "kernel:metal:navatala_linalg_map_reduce_sum_f32",
  "abi-r1:metal:navatala_linalg_map_reduce_sum_f32",
  "abi-r1:metal:navatala_linalg_map_reduce_sum_f32",
  4,
  kAbiArgs_metal_navatala_linalg_map_reduce_sum_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_outer_product_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_outer_product_f32 = {
  1,
  "navatala_linalg_outer_product_f32",
  "metal",
  "navatala_linalg_outer_product_f32",
  "kernel:metal:navatala_linalg_outer_product_f32",
  "abi-r1:metal:navatala_linalg_outer_product_f32",
  "abi-r1:metal:navatala_linalg_outer_product_f32",
  5,
  kAbiArgs_metal_navatala_linalg_outer_product_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_batched_dot_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vecLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_batched_dot_f32 = {
  1,
  "navatala_linalg_batched_dot_f32",
  "metal",
  "navatala_linalg_batched_dot_f32",
  "kernel:metal:navatala_linalg_batched_dot_f32",
  "abi-r1:metal:navatala_linalg_batched_dot_f32",
  "abi-r1:metal:navatala_linalg_batched_dot_f32",
  5,
  kAbiArgs_metal_navatala_linalg_batched_dot_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_trace_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_trace_f32 = {
  1,
  "navatala_linalg_trace_f32",
  "metal",
  "navatala_linalg_trace_f32",
  "kernel:metal:navatala_linalg_trace_f32",
  "abi-r1:metal:navatala_linalg_trace_f32",
  "abi-r1:metal:navatala_linalg_trace_f32",
  3,
  kAbiArgs_metal_navatala_linalg_trace_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_frobenius_norm_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_frobenius_norm_f32 = {
  1,
  "navatala_linalg_frobenius_norm_f32",
  "metal",
  "navatala_linalg_frobenius_norm_f32",
  "kernel:metal:navatala_linalg_frobenius_norm_f32",
  "abi-r1:metal:navatala_linalg_frobenius_norm_f32",
  "abi-r1:metal:navatala_linalg_frobenius_norm_f32",
  4,
  kAbiArgs_metal_navatala_linalg_frobenius_norm_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_extract_upper_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_extract_upper_f32 = {
  1,
  "navatala_linalg_extract_upper_f32",
  "metal",
  "navatala_linalg_extract_upper_f32",
  "kernel:metal:navatala_linalg_extract_upper_f32",
  "abi-r1:metal:navatala_linalg_extract_upper_f32",
  "abi-r1:metal:navatala_linalg_extract_upper_f32",
  3,
  kAbiArgs_metal_navatala_linalg_extract_upper_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_extract_lower_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_extract_lower_f32 = {
  1,
  "navatala_linalg_extract_lower_f32",
  "metal",
  "navatala_linalg_extract_lower_f32",
  "kernel:metal:navatala_linalg_extract_lower_f32",
  "abi-r1:metal:navatala_linalg_extract_lower_f32",
  "abi-r1:metal:navatala_linalg_extract_lower_f32",
  3,
  kAbiArgs_metal_navatala_linalg_extract_lower_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_set_upper_f32[] = {
  { "val", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_set_upper_f32 = {
  1,
  "navatala_linalg_set_upper_f32",
  "metal",
  "navatala_linalg_set_upper_f32",
  "kernel:metal:navatala_linalg_set_upper_f32",
  "abi-r1:metal:navatala_linalg_set_upper_f32",
  "abi-r1:metal:navatala_linalg_set_upper_f32",
  3,
  kAbiArgs_metal_navatala_linalg_set_upper_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_set_lower_f32[] = {
  { "val", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_set_lower_f32 = {
  1,
  "navatala_linalg_set_lower_f32",
  "metal",
  "navatala_linalg_set_lower_f32",
  "kernel:metal:navatala_linalg_set_lower_f32",
  "abi-r1:metal:navatala_linalg_set_lower_f32",
  "abi-r1:metal:navatala_linalg_set_lower_f32",
  3,
  kAbiArgs_metal_navatala_linalg_set_lower_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_gather_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_gather_rows_f32 = {
  1,
  "navatala_linalg_gather_rows_f32",
  "metal",
  "navatala_linalg_gather_rows_f32",
  "kernel:metal:navatala_linalg_gather_rows_f32",
  "abi-r1:metal:navatala_linalg_gather_rows_f32",
  "abi-r1:metal:navatala_linalg_gather_rows_f32",
  5,
  kAbiArgs_metal_navatala_linalg_gather_rows_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_gather_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kCount", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_gather_cols_f32 = {
  1,
  "navatala_linalg_gather_cols_f32",
  "metal",
  "navatala_linalg_gather_cols_f32",
  "kernel:metal:navatala_linalg_gather_cols_f32",
  "abi-r1:metal:navatala_linalg_gather_cols_f32",
  "abi-r1:metal:navatala_linalg_gather_cols_f32",
  6,
  kAbiArgs_metal_navatala_linalg_gather_cols_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_gather_elements_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_gather_elements_f32 = {
  1,
  "navatala_linalg_gather_elements_f32",
  "metal",
  "navatala_linalg_gather_elements_f32",
  "kernel:metal:navatala_linalg_gather_elements_f32",
  "abi-r1:metal:navatala_linalg_gather_elements_f32",
  "abi-r1:metal:navatala_linalg_gather_elements_f32",
  6,
  kAbiArgs_metal_navatala_linalg_gather_elements_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_gather_batched_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kCount", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_gather_batched_f32 = {
  1,
  "navatala_linalg_gather_batched_f32",
  "metal",
  "navatala_linalg_gather_batched_f32",
  "kernel:metal:navatala_linalg_gather_batched_f32",
  "abi-r1:metal:navatala_linalg_gather_batched_f32",
  "abi-r1:metal:navatala_linalg_gather_batched_f32",
  7,
  kAbiArgs_metal_navatala_linalg_gather_batched_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_scatter_rows_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_scatter_rows_f32 = {
  1,
  "navatala_linalg_scatter_rows_f32",
  "metal",
  "navatala_linalg_scatter_rows_f32",
  "kernel:metal:navatala_linalg_scatter_rows_f32",
  "abi-r1:metal:navatala_linalg_scatter_rows_f32",
  "abi-r1:metal:navatala_linalg_scatter_rows_f32",
  5,
  kAbiArgs_metal_navatala_linalg_scatter_rows_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_scatter_cols_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_scatter_cols_f32 = {
  1,
  "navatala_linalg_scatter_cols_f32",
  "metal",
  "navatala_linalg_scatter_cols_f32",
  "kernel:metal:navatala_linalg_scatter_cols_f32",
  "abi-r1:metal:navatala_linalg_scatter_cols_f32",
  "abi-r1:metal:navatala_linalg_scatter_cols_f32",
  6,
  kAbiArgs_metal_navatala_linalg_scatter_cols_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_scatter_add_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_scatter_add_f32 = {
  1,
  "navatala_linalg_scatter_add_f32",
  "metal",
  "navatala_linalg_scatter_add_f32",
  "kernel:metal:navatala_linalg_scatter_add_f32",
  "abi-r1:metal:navatala_linalg_scatter_add_f32",
  "abi-r1:metal:navatala_linalg_scatter_add_f32",
  4,
  kAbiArgs_metal_navatala_linalg_scatter_add_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_scatter_max_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_scatter_max_f32 = {
  1,
  "navatala_linalg_scatter_max_f32",
  "metal",
  "navatala_linalg_scatter_max_f32",
  "kernel:metal:navatala_linalg_scatter_max_f32",
  "abi-r1:metal:navatala_linalg_scatter_max_f32",
  "abi-r1:metal:navatala_linalg_scatter_max_f32",
  4,
  kAbiArgs_metal_navatala_linalg_scatter_max_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_slice_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowStart", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowEnd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numCols", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_slice_rows_f32 = {
  1,
  "navatala_linalg_slice_rows_f32",
  "metal",
  "navatala_linalg_slice_rows_f32",
  "kernel:metal:navatala_linalg_slice_rows_f32",
  "abi-r1:metal:navatala_linalg_slice_rows_f32",
  "abi-r1:metal:navatala_linalg_slice_rows_f32",
  5,
  kAbiArgs_metal_navatala_linalg_slice_rows_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_slice_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colStart", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colEnd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcCols", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_slice_cols_f32 = {
  1,
  "navatala_linalg_slice_cols_f32",
  "metal",
  "navatala_linalg_slice_cols_f32",
  "kernel:metal:navatala_linalg_slice_cols_f32",
  "abi-r1:metal:navatala_linalg_slice_cols_f32",
  "abi-r1:metal:navatala_linalg_slice_cols_f32",
  6,
  kAbiArgs_metal_navatala_linalg_slice_cols_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_slice_block_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowStart", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowEnd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colStart", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colEnd", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcCols", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_slice_block_f32 = {
  1,
  "navatala_linalg_slice_block_f32",
  "metal",
  "navatala_linalg_slice_block_f32",
  "kernel:metal:navatala_linalg_slice_block_f32",
  "abi-r1:metal:navatala_linalg_slice_block_f32",
  "abi-r1:metal:navatala_linalg_slice_block_f32",
  7,
  kAbiArgs_metal_navatala_linalg_slice_block_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_strided_slice_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "start", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stride", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_strided_slice_f32 = {
  1,
  "navatala_linalg_strided_slice_f32",
  "metal",
  "navatala_linalg_strided_slice_f32",
  "kernel:metal:navatala_linalg_strided_slice_f32",
  "abi-r1:metal:navatala_linalg_strided_slice_f32",
  "abi-r1:metal:navatala_linalg_strided_slice_f32",
  5,
  kAbiArgs_metal_navatala_linalg_strided_slice_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_extract_diagonal_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_extract_diagonal_f32 = {
  1,
  "navatala_linalg_extract_diagonal_f32",
  "metal",
  "navatala_linalg_extract_diagonal_f32",
  "kernel:metal:navatala_linalg_extract_diagonal_f32",
  "abi-r1:metal:navatala_linalg_extract_diagonal_f32",
  "abi-r1:metal:navatala_linalg_extract_diagonal_f32",
  3,
  kAbiArgs_metal_navatala_linalg_extract_diagonal_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_extract_diagonal_k_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_extract_diagonal_k_f32 = {
  1,
  "navatala_linalg_extract_diagonal_k_f32",
  "metal",
  "navatala_linalg_extract_diagonal_k_f32",
  "kernel:metal:navatala_linalg_extract_diagonal_k_f32",
  "abi-r1:metal:navatala_linalg_extract_diagonal_k_f32",
  "abi-r1:metal:navatala_linalg_extract_diagonal_k_f32",
  4,
  kAbiArgs_metal_navatala_linalg_extract_diagonal_k_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_set_diagonal_f32[] = {
  { "val", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_set_diagonal_f32 = {
  1,
  "navatala_linalg_set_diagonal_f32",
  "metal",
  "navatala_linalg_set_diagonal_f32",
  "kernel:metal:navatala_linalg_set_diagonal_f32",
  "abi-r1:metal:navatala_linalg_set_diagonal_f32",
  "abi-r1:metal:navatala_linalg_set_diagonal_f32",
  3,
  kAbiArgs_metal_navatala_linalg_set_diagonal_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_diag_to_matrix_f32[] = {
  { "v", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_diag_to_matrix_f32 = {
  1,
  "navatala_linalg_diag_to_matrix_f32",
  "metal",
  "navatala_linalg_diag_to_matrix_f32",
  "kernel:metal:navatala_linalg_diag_to_matrix_f32",
  "abi-r1:metal:navatala_linalg_diag_to_matrix_f32",
  "abi-r1:metal:navatala_linalg_diag_to_matrix_f32",
  3,
  kAbiArgs_metal_navatala_linalg_diag_to_matrix_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_shift_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_shift_rows_f32 = {
  1,
  "navatala_linalg_shift_rows_f32",
  "metal",
  "navatala_linalg_shift_rows_f32",
  "kernel:metal:navatala_linalg_shift_rows_f32",
  "abi-r1:metal:navatala_linalg_shift_rows_f32",
  "abi-r1:metal:navatala_linalg_shift_rows_f32",
  5,
  kAbiArgs_metal_navatala_linalg_shift_rows_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_shift_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_shift_cols_f32 = {
  1,
  "navatala_linalg_shift_cols_f32",
  "metal",
  "navatala_linalg_shift_cols_f32",
  "kernel:metal:navatala_linalg_shift_cols_f32",
  "abi-r1:metal:navatala_linalg_shift_cols_f32",
  "abi-r1:metal:navatala_linalg_shift_cols_f32",
  5,
  kAbiArgs_metal_navatala_linalg_shift_cols_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_roll_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_roll_rows_f32 = {
  1,
  "navatala_linalg_roll_rows_f32",
  "metal",
  "navatala_linalg_roll_rows_f32",
  "kernel:metal:navatala_linalg_roll_rows_f32",
  "abi-r1:metal:navatala_linalg_roll_rows_f32",
  "abi-r1:metal:navatala_linalg_roll_rows_f32",
  5,
  kAbiArgs_metal_navatala_linalg_roll_rows_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_roll_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_roll_cols_f32 = {
  1,
  "navatala_linalg_roll_cols_f32",
  "metal",
  "navatala_linalg_roll_cols_f32",
  "kernel:metal:navatala_linalg_roll_cols_f32",
  "abi-r1:metal:navatala_linalg_roll_cols_f32",
  "abi-r1:metal:navatala_linalg_roll_cols_f32",
  5,
  kAbiArgs_metal_navatala_linalg_roll_cols_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_sort_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_sort_rows_f32 = {
  1,
  "navatala_linalg_sort_rows_f32",
  "metal",
  "navatala_linalg_sort_rows_f32",
  "kernel:metal:navatala_linalg_sort_rows_f32",
  "abi-r1:metal:navatala_linalg_sort_rows_f32",
  "abi-r1:metal:navatala_linalg_sort_rows_f32",
  4,
  kAbiArgs_metal_navatala_linalg_sort_rows_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_sort_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_sort_cols_f32 = {
  1,
  "navatala_linalg_sort_cols_f32",
  "metal",
  "navatala_linalg_sort_cols_f32",
  "kernel:metal:navatala_linalg_sort_cols_f32",
  "abi-r1:metal:navatala_linalg_sort_cols_f32",
  "abi-r1:metal:navatala_linalg_sort_cols_f32",
  4,
  kAbiArgs_metal_navatala_linalg_sort_cols_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_argsort_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_argsort_rows_f32 = {
  1,
  "navatala_linalg_argsort_rows_f32",
  "metal",
  "navatala_linalg_argsort_rows_f32",
  "kernel:metal:navatala_linalg_argsort_rows_f32",
  "abi-r1:metal:navatala_linalg_argsort_rows_f32",
  "abi-r1:metal:navatala_linalg_argsort_rows_f32",
  4,
  kAbiArgs_metal_navatala_linalg_argsort_rows_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_linalg_argsort_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_linalg_argsort_cols_f32 = {
  1,
  "navatala_linalg_argsort_cols_f32",
  "metal",
  "navatala_linalg_argsort_cols_f32",
  "kernel:metal:navatala_linalg_argsort_cols_f32",
  "abi-r1:metal:navatala_linalg_argsort_cols_f32",
  "abi-r1:metal:navatala_linalg_argsort_cols_f32",
  4,
  kAbiArgs_metal_navatala_linalg_argsort_cols_f32
};

bool tryGetKernelAbiManifest_metal_linalg(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_linalg_norm_linf_f32") {
    out = &kAbiManifest_metal_navatala_linalg_norm_linf_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_eig_select_top_k_f32") {
    out = &kAbiManifest_metal_navatala_linalg_eig_select_top_k_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_cast_f32_to_f16") {
    out = &kAbiManifest_metal_navatala_linalg_cast_f32_to_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_cast_f16_to_f32") {
    out = &kAbiManifest_metal_navatala_linalg_cast_f16_to_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_reduce_sum_f16") {
    out = &kAbiManifest_metal_navatala_linalg_reduce_sum_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_reduce_max_f16") {
    out = &kAbiManifest_metal_navatala_linalg_reduce_max_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_add_f32") {
    out = &kAbiManifest_metal_navatala_linalg_elt_add_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_sub_f32") {
    out = &kAbiManifest_metal_navatala_linalg_elt_sub_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_mul_f32") {
    out = &kAbiManifest_metal_navatala_linalg_elt_mul_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_div_f32") {
    out = &kAbiManifest_metal_navatala_linalg_elt_div_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_pow_f32") {
    out = &kAbiManifest_metal_navatala_linalg_elt_pow_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_sqrt_f32") {
    out = &kAbiManifest_metal_navatala_linalg_elt_sqrt_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_exp_f32") {
    out = &kAbiManifest_metal_navatala_linalg_elt_exp_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_log_f32") {
    out = &kAbiManifest_metal_navatala_linalg_elt_log_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_abs_f32") {
    out = &kAbiManifest_metal_navatala_linalg_elt_abs_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_sign_f32") {
    out = &kAbiManifest_metal_navatala_linalg_elt_sign_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_transpose_f32") {
    out = &kAbiManifest_metal_navatala_linalg_transpose_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_transpose_in_place_f32") {
    out = &kAbiManifest_metal_navatala_linalg_transpose_in_place_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_transpose_batched_f32") {
    out = &kAbiManifest_metal_navatala_linalg_transpose_batched_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_map_f32") {
    out = &kAbiManifest_metal_navatala_linalg_map_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_reduce_row_sum_f32") {
    out = &kAbiManifest_metal_navatala_linalg_reduce_row_sum_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_reduce_col_sum_f32") {
    out = &kAbiManifest_metal_navatala_linalg_reduce_col_sum_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_reduce_row_max_f32") {
    out = &kAbiManifest_metal_navatala_linalg_reduce_row_max_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_reduce_col_max_f32") {
    out = &kAbiManifest_metal_navatala_linalg_reduce_col_max_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_map_reduce_sum_f32") {
    out = &kAbiManifest_metal_navatala_linalg_map_reduce_sum_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_outer_product_f32") {
    out = &kAbiManifest_metal_navatala_linalg_outer_product_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_batched_dot_f32") {
    out = &kAbiManifest_metal_navatala_linalg_batched_dot_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_trace_f32") {
    out = &kAbiManifest_metal_navatala_linalg_trace_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_frobenius_norm_f32") {
    out = &kAbiManifest_metal_navatala_linalg_frobenius_norm_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_extract_upper_f32") {
    out = &kAbiManifest_metal_navatala_linalg_extract_upper_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_extract_lower_f32") {
    out = &kAbiManifest_metal_navatala_linalg_extract_lower_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_set_upper_f32") {
    out = &kAbiManifest_metal_navatala_linalg_set_upper_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_set_lower_f32") {
    out = &kAbiManifest_metal_navatala_linalg_set_lower_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_gather_rows_f32") {
    out = &kAbiManifest_metal_navatala_linalg_gather_rows_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_gather_cols_f32") {
    out = &kAbiManifest_metal_navatala_linalg_gather_cols_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_gather_elements_f32") {
    out = &kAbiManifest_metal_navatala_linalg_gather_elements_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_gather_batched_f32") {
    out = &kAbiManifest_metal_navatala_linalg_gather_batched_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_scatter_rows_f32") {
    out = &kAbiManifest_metal_navatala_linalg_scatter_rows_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_scatter_cols_f32") {
    out = &kAbiManifest_metal_navatala_linalg_scatter_cols_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_scatter_add_f32") {
    out = &kAbiManifest_metal_navatala_linalg_scatter_add_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_scatter_max_f32") {
    out = &kAbiManifest_metal_navatala_linalg_scatter_max_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_slice_rows_f32") {
    out = &kAbiManifest_metal_navatala_linalg_slice_rows_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_slice_cols_f32") {
    out = &kAbiManifest_metal_navatala_linalg_slice_cols_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_slice_block_f32") {
    out = &kAbiManifest_metal_navatala_linalg_slice_block_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_strided_slice_f32") {
    out = &kAbiManifest_metal_navatala_linalg_strided_slice_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_extract_diagonal_f32") {
    out = &kAbiManifest_metal_navatala_linalg_extract_diagonal_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_extract_diagonal_k_f32") {
    out = &kAbiManifest_metal_navatala_linalg_extract_diagonal_k_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_set_diagonal_f32") {
    out = &kAbiManifest_metal_navatala_linalg_set_diagonal_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_diag_to_matrix_f32") {
    out = &kAbiManifest_metal_navatala_linalg_diag_to_matrix_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_shift_rows_f32") {
    out = &kAbiManifest_metal_navatala_linalg_shift_rows_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_shift_cols_f32") {
    out = &kAbiManifest_metal_navatala_linalg_shift_cols_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_roll_rows_f32") {
    out = &kAbiManifest_metal_navatala_linalg_roll_rows_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_roll_cols_f32") {
    out = &kAbiManifest_metal_navatala_linalg_roll_cols_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_sort_rows_f32") {
    out = &kAbiManifest_metal_navatala_linalg_sort_rows_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_sort_cols_f32") {
    out = &kAbiManifest_metal_navatala_linalg_sort_cols_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_argsort_rows_f32") {
    out = &kAbiManifest_metal_navatala_linalg_argsort_rows_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_argsort_cols_f32") {
    out = &kAbiManifest_metal_navatala_linalg_argsort_cols_f32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_linalg(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_linalg_norm_linf_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_norm_linf_f32";
    std::string_view sv(k_metal_navatala_linalg_norm_linf_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_eig_select_top_k_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_eig_select_top_k_f32";
    std::string_view sv(k_metal_navatala_linalg_eig_select_top_k_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_cast_f32_to_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_cast_f32_to_f16";
    std::string_view sv(k_metal_navatala_linalg_cast_f32_to_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_cast_f16_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_cast_f16_to_f32";
    std::string_view sv(k_metal_navatala_linalg_cast_f16_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_reduce_sum_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_reduce_sum_f16";
    std::string_view sv(k_metal_navatala_linalg_reduce_sum_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_reduce_max_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_reduce_max_f16";
    std::string_view sv(k_metal_navatala_linalg_reduce_max_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_add_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_elt_add_f32";
    std::string_view sv(k_metal_navatala_linalg_elt_add_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_sub_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_elt_sub_f32";
    std::string_view sv(k_metal_navatala_linalg_elt_sub_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_mul_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_elt_mul_f32";
    std::string_view sv(k_metal_navatala_linalg_elt_mul_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_div_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_elt_div_f32";
    std::string_view sv(k_metal_navatala_linalg_elt_div_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_pow_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_elt_pow_f32";
    std::string_view sv(k_metal_navatala_linalg_elt_pow_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_sqrt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_elt_sqrt_f32";
    std::string_view sv(k_metal_navatala_linalg_elt_sqrt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_exp_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_elt_exp_f32";
    std::string_view sv(k_metal_navatala_linalg_elt_exp_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_log_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_elt_log_f32";
    std::string_view sv(k_metal_navatala_linalg_elt_log_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_abs_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_elt_abs_f32";
    std::string_view sv(k_metal_navatala_linalg_elt_abs_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_elt_sign_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_elt_sign_f32";
    std::string_view sv(k_metal_navatala_linalg_elt_sign_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_transpose_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_transpose_f32";
    std::string_view sv(k_metal_navatala_linalg_transpose_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_transpose_in_place_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_transpose_in_place_f32";
    std::string_view sv(k_metal_navatala_linalg_transpose_in_place_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_transpose_batched_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_transpose_batched_f32";
    std::string_view sv(k_metal_navatala_linalg_transpose_batched_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_map_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_map_f32";
    std::string_view sv(k_metal_navatala_linalg_map_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_reduce_row_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_reduce_row_sum_f32";
    std::string_view sv(k_metal_navatala_linalg_reduce_row_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_reduce_col_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_reduce_col_sum_f32";
    std::string_view sv(k_metal_navatala_linalg_reduce_col_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_reduce_row_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_reduce_row_max_f32";
    std::string_view sv(k_metal_navatala_linalg_reduce_row_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_reduce_col_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_reduce_col_max_f32";
    std::string_view sv(k_metal_navatala_linalg_reduce_col_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_map_reduce_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_map_reduce_sum_f32";
    std::string_view sv(k_metal_navatala_linalg_map_reduce_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_outer_product_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_outer_product_f32";
    std::string_view sv(k_metal_navatala_linalg_outer_product_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_batched_dot_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_batched_dot_f32";
    std::string_view sv(k_metal_navatala_linalg_batched_dot_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_trace_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_trace_f32";
    std::string_view sv(k_metal_navatala_linalg_trace_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_frobenius_norm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_frobenius_norm_f32";
    std::string_view sv(k_metal_navatala_linalg_frobenius_norm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_extract_upper_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_extract_upper_f32";
    std::string_view sv(k_metal_navatala_linalg_extract_upper_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_extract_lower_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_extract_lower_f32";
    std::string_view sv(k_metal_navatala_linalg_extract_lower_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_set_upper_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_set_upper_f32";
    std::string_view sv(k_metal_navatala_linalg_set_upper_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_set_lower_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_set_lower_f32";
    std::string_view sv(k_metal_navatala_linalg_set_lower_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_gather_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_gather_rows_f32";
    std::string_view sv(k_metal_navatala_linalg_gather_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_gather_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_gather_cols_f32";
    std::string_view sv(k_metal_navatala_linalg_gather_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_gather_elements_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_gather_elements_f32";
    std::string_view sv(k_metal_navatala_linalg_gather_elements_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_gather_batched_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_gather_batched_f32";
    std::string_view sv(k_metal_navatala_linalg_gather_batched_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_scatter_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_scatter_rows_f32";
    std::string_view sv(k_metal_navatala_linalg_scatter_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_scatter_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_scatter_cols_f32";
    std::string_view sv(k_metal_navatala_linalg_scatter_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_scatter_add_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_scatter_add_f32";
    std::string_view sv(k_metal_navatala_linalg_scatter_add_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_scatter_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_scatter_max_f32";
    std::string_view sv(k_metal_navatala_linalg_scatter_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_slice_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_slice_rows_f32";
    std::string_view sv(k_metal_navatala_linalg_slice_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_slice_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_slice_cols_f32";
    std::string_view sv(k_metal_navatala_linalg_slice_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_slice_block_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_slice_block_f32";
    std::string_view sv(k_metal_navatala_linalg_slice_block_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_strided_slice_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_strided_slice_f32";
    std::string_view sv(k_metal_navatala_linalg_strided_slice_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_extract_diagonal_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_extract_diagonal_f32";
    std::string_view sv(k_metal_navatala_linalg_extract_diagonal_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_extract_diagonal_k_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_extract_diagonal_k_f32";
    std::string_view sv(k_metal_navatala_linalg_extract_diagonal_k_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_set_diagonal_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_set_diagonal_f32";
    std::string_view sv(k_metal_navatala_linalg_set_diagonal_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_diag_to_matrix_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_diag_to_matrix_f32";
    std::string_view sv(k_metal_navatala_linalg_diag_to_matrix_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_shift_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_shift_rows_f32";
    std::string_view sv(k_metal_navatala_linalg_shift_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_shift_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_shift_cols_f32";
    std::string_view sv(k_metal_navatala_linalg_shift_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_roll_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_roll_rows_f32";
    std::string_view sv(k_metal_navatala_linalg_roll_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_roll_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_roll_cols_f32";
    std::string_view sv(k_metal_navatala_linalg_roll_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_sort_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_sort_rows_f32";
    std::string_view sv(k_metal_navatala_linalg_sort_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_sort_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_sort_cols_f32";
    std::string_view sv(k_metal_navatala_linalg_sort_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_argsort_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_argsort_rows_f32";
    std::string_view sv(k_metal_navatala_linalg_argsort_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_linalg_argsort_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_linalg_argsort_cols_f32";
    std::string_view sv(k_metal_navatala_linalg_argsort_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

