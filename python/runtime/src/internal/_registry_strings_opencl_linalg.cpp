// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `opencl_linalg` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_opencl_navatala_linalg_norm_linf_f32 = R"kernel(
__kernel void navatala_linalg_norm_linf_f32(__global const float* _input, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint gid = ((uint)((int)(get_global_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  float val = fabs(_input[gid]);
  if (inBounds) {
    sdata[lid] = val;
  } else {
    sdata[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint linfF32RedStride = (uint)(128u);
  for (int linfF32RedStep = 0; linfF32RedStep < (int)(8); ++linfF32RedStep) {
    uint linfF32Stride = linfF32RedStride;
    if ((lid < linfF32Stride)) {
      float other = sdata[(lid + linfF32Stride)];
      float mine = sdata[lid];
      bool mineGtOther = (mine > other);
      float maxVal = ((mineGtOther) ? (mine) : (other));
      sdata[lid] = maxVal;
    }
    uint linfF32StrideToHalve = linfF32RedStride;
    uint linfF32NextStride = (linfF32StrideToHalve >> (uint)(1u));
    linfF32RedStride = linfF32NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    result[0] = sdata[0];
  }
}

)kernel";
const char* k_opencl_navatala_linalg_norm_linf_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_norm_linf_f64(__global const double* _input, __global const uint* count, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint gid = ((uint)((int)(get_global_id(0))));
  __local double sdata[256];
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  double val = fabs(_input[gid]);
  if (inBounds) {
    sdata[lid] = val;
  } else {
    sdata[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint linfF64RedStride = (uint)(128u);
  for (int linfF64RedStep = 0; linfF64RedStep < (int)(8); ++linfF64RedStep) {
    uint linfF64Stride = linfF64RedStride;
    if ((lid < linfF64Stride)) {
      double other = sdata[(lid + linfF64Stride)];
      double mine = sdata[lid];
      bool mineGtOther = (mine > other);
      double maxVal = ((mineGtOther) ? (mine) : (other));
      sdata[lid] = maxVal;
    }
    uint linfF64StrideToHalve = linfF64RedStride;
    uint linfF64NextStride = (linfF64StrideToHalve >> (uint)(1u));
    linfF64RedStride = linfF64NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    result[0] = sdata[0];
  }
}

)kernel";
const char* k_opencl_navatala_linalg_eig_select_top_k_f32 = R"kernel(
__kernel void navatala_linalg_eig_select_top_k_f32(__global const float* eigenvalues, __global const float* eigenvectors, __global const uint* k, __global const uint* n, __global float* topKValues, __global float* topKVectors) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint kVal = k[0];
  uint nVal = n[0];
  uint totalWork = (kVal * nVal);
  bool inBounds = (gid < totalWork);
  if (inBounds) {
    uint col = (gid / nVal);
    uint row = (gid % nVal);
    bool isFirstRow = (row == (uint)(0u));
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
const char* k_opencl_navatala_linalg_eig_select_top_k_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_eig_select_top_k_f64(__global const double* eigenvalues, __global const double* eigenvectors, __global const uint* k, __global const uint* n, __global double* topKValues, __global double* topKVectors) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint kVal = k[0];
  uint nVal = n[0];
  uint totalWork = (kVal * nVal);
  bool inBounds = (gid < totalWork);
  if (inBounds) {
    uint col = (gid / nVal);
    uint row = (gid % nVal);
    bool isFirstRow = (row == (uint)(0u));
    if (isFirstRow) {
      double eigenval = eigenvalues[col];
      topKValues[col] = eigenval;
    }
    uint srcIdx = ((col * nVal) + row);
    double vecElem = eigenvectors[srcIdx];
    topKVectors[srcIdx] = vecElem;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_cast_f32_to_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_linalg_cast_f32_to_f16(__global const float* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    float x = _input[gid];
    half y = ((half)(x));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_cast_f64_to_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_cast_f64_to_f16(__global const double* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    double x = _input[gid];
    float xf32 = ((float)(x));
    half y = ((half)(xf32));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_cast_f16_to_f32 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_linalg_cast_f16_to_f32(__global const half* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    half x = _input[gid];
    float y = ((float)(x));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_cast_f16_to_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_cast_f16_to_f64(__global const half* _input, __global const uint* count, __global double* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    half x = _input[gid];
    float xf32 = ((float)(x));
    double y = ((double)(xf32));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_reduce_sum_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_linalg_reduce_sum_f16(__global const half* _input, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    half xf16 = _input[gid];
    float xf32 = ((float)(xf16));
    sdata[lid] = xf32;
  } else {
    sdata[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint sumF16ReductionStride = (uint)(128u);
  for (int sumF16ReductionStep = 0; sumF16ReductionStep < (int)(8); ++sumF16ReductionStep) {
    uint sumF16Stride = sumF16ReductionStride;
    if ((lid < sumF16Stride)) {
      float other = sdata[(lid + sumF16Stride)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    uint sumF16StrideToHalve = sumF16ReductionStride;
    uint sumF16NextStride = (sumF16StrideToHalve >> (uint)(1u));
    sumF16ReductionStride = sumF16NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float partialSum = sdata[0];
    result[0] = partialSum;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_reduce_max_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_linalg_reduce_max_f16(__global const half* _input, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    half xf16 = _input[gid];
    float xf32 = ((float)(xf16));
    sdata[lid] = xf32;
  } else {
    sdata[lid] = as_float(0xff7ffffdu);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint maxF16ReductionStride = (uint)(128u);
  for (int maxF16ReductionStep = 0; maxF16ReductionStep < (int)(8); ++maxF16ReductionStep) {
    uint maxF16Stride = maxF16ReductionStride;
    if ((lid < maxF16Stride)) {
      float other = sdata[(lid + maxF16Stride)];
      float mine = sdata[lid];
      bool isGreater = (mine > other);
      float maxVal = ((isGreater) ? (mine) : (other));
      sdata[lid] = maxVal;
    }
    uint maxF16StrideToHalve = maxF16ReductionStride;
    uint maxF16NextStride = (maxF16StrideToHalve >> (uint)(1u));
    maxF16ReductionStride = maxF16NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float maxResult = sdata[0];
    result[0] = maxResult;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_add_f32 = R"kernel(
__kernel void navatala_linalg_elt_add_f32(__global const float* A, __global const float* B, __global const uint* n, __global float* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    float a = A[i];
    float b = B[i];
    C[i] = (a + b);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_add_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_elt_add_f64(__global const double* A, __global const double* B, __global const uint* n, __global double* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    double a = A[i];
    double b = B[i];
    C[i] = (a + b);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_sub_f32 = R"kernel(
__kernel void navatala_linalg_elt_sub_f32(__global const float* A, __global const float* B, __global const uint* n, __global float* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    float a = A[i];
    float b = B[i];
    C[i] = (a - b);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_sub_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_elt_sub_f64(__global const double* A, __global const double* B, __global const uint* n, __global double* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    double a = A[i];
    double b = B[i];
    C[i] = (a - b);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_mul_f32 = R"kernel(
__kernel void navatala_linalg_elt_mul_f32(__global const float* A, __global const float* B, __global const uint* n, __global float* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    float a = A[i];
    float b = B[i];
    C[i] = (a * b);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_mul_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_elt_mul_f64(__global const double* A, __global const double* B, __global const uint* n, __global double* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    double a = A[i];
    double b = B[i];
    C[i] = (a * b);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_div_f32 = R"kernel(
__kernel void navatala_linalg_elt_div_f32(__global const float* A, __global const float* B, __global const uint* n, __global float* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    float a = A[i];
    float b = B[i];
    C[i] = (a / b);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_div_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_elt_div_f64(__global const double* A, __global const double* B, __global const uint* n, __global double* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    double a = A[i];
    double b = B[i];
    C[i] = (a / b);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_pow_f32 = R"kernel(
__kernel void navatala_linalg_elt_pow_f32(__global const float* A, __global const float* p, __global const uint* n, __global float* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  float pVal = p[(uint)(0u)];
  if ((i < n[(uint)(0u)])) {
    float a = A[i];
    float logA = log(a);
    float pLogA = (pVal * logA);
    C[i] = exp(pLogA);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_pow_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_elt_pow_f64(__global const double* A, __global const double* p, __global const uint* n, __global double* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  double pVal = p[(uint)(0u)];
  if ((i < n[(uint)(0u)])) {
    double a = A[i];
    double logA = log(a);
    double pLogA = (pVal * logA);
    C[i] = exp(pLogA);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_sqrt_f32 = R"kernel(
__kernel void navatala_linalg_elt_sqrt_f32(__global const float* A, __global const uint* n, __global float* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    float a = A[i];
    C[i] = sqrt(a);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_sqrt_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_elt_sqrt_f64(__global const double* A, __global const uint* n, __global double* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    double a = A[i];
    C[i] = sqrt(a);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_exp_f32 = R"kernel(
__kernel void navatala_linalg_elt_exp_f32(__global const float* A, __global const uint* n, __global float* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    float a = A[i];
    C[i] = exp(a);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_exp_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_elt_exp_f64(__global const double* A, __global const uint* n, __global double* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    double a = A[i];
    C[i] = exp(a);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_log_f32 = R"kernel(
__kernel void navatala_linalg_elt_log_f32(__global const float* A, __global const uint* n, __global float* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    float a = A[i];
    C[i] = log(a);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_log_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_elt_log_f64(__global const double* A, __global const uint* n, __global double* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    double a = A[i];
    C[i] = log(a);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_abs_f32 = R"kernel(
__kernel void navatala_linalg_elt_abs_f32(__global const float* A, __global const uint* n, __global float* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    float a = A[i];
    C[i] = fabs(a);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_abs_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_elt_abs_f64(__global const double* A, __global const uint* n, __global double* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    double a = A[i];
    C[i] = fabs(a);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_sign_f32 = R"kernel(
__kernel void navatala_linalg_elt_sign_f32(__global const float* A, __global const uint* n, __global float* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    float a = A[i];
    bool isNeg = (a < as_float(0x00000000u));
    float negOrZero = ((isNeg) ? (as_float(0xbf800000u)) : (as_float(0x00000000u)));
    bool isPos = (a > as_float(0x00000000u));
    float result = ((isPos) ? (as_float(0x3f800000u)) : (negOrZero));
    C[i] = result;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_elt_sign_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_elt_sign_f64(__global const double* A, __global const uint* n, __global double* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    double a = A[i];
    bool isNeg = (a < as_double(0x0000000000000000ul));
    double negOrZero = ((isNeg) ? (as_double(0xbff0000000000000ul)) : (as_double(0x0000000000000000ul)));
    bool isPos = (a > as_double(0x0000000000000000ul));
    double result = ((isPos) ? (as_double(0x3ff0000000000000ul)) : (negOrZero));
    C[i] = result;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_transpose_f32 = R"kernel(
__kernel void navatala_linalg_transpose_f32(__global const float* A, __global const uint* m, __global const uint* n, __global float* B) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
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
const char* k_opencl_navatala_linalg_transpose_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_transpose_f64(__global const double* A, __global const uint* m, __global const uint* n, __global double* B) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint rows = m[0];
  uint cols = n[0];
  uint totalElems = (rows * cols);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / cols);
    uint j = (gid % cols);
    uint srcIdx = ((i * cols) + j);
    uint dstIdx = ((j * rows) + i);
    double val = A[srcIdx];
    B[dstIdx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_transpose_in_place_f32 = R"kernel(
__kernel void navatala_linalg_transpose_in_place_f32(__global const uint* n, __global float* A) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint dim = n[0];
  uint numSwaps = ((dim * (dim - (uint)(1u))) / (uint)(2u));
  bool inBounds = (gid < numSwaps);
  if (inBounds) {
    float gidF = ((float)(gid));
    float dimF = ((float)(dim));
    float twoNm1 = ((as_float(0x40000000u) * dimF) - as_float(0x3f800000u));
    float disc = ((twoNm1 * twoNm1) - (as_float(0x41000000u) * gidF));
    float sqrtDisc = sqrt(disc);
    float iF = floor(((twoNm1 - sqrtDisc) / as_float(0x40000000u)));
    uint i = ((uint)(iF));
    uint rowStart = ((i * ((((uint)(2u) * dim) - i) - (uint)(1u))) / (uint)(2u));
    uint offset = (gid - rowStart);
    uint j = ((i + (uint)(1u)) + offset);
    uint idx_ij = ((i * dim) + j);
    uint idx_ji = ((j * dim) + i);
    float val_ij = A[idx_ij];
    float val_ji = A[idx_ji];
    A[idx_ij] = val_ji;
    A[idx_ji] = val_ij;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_transpose_in_place_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_transpose_in_place_f64(__global const uint* n, __global double* A) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint dim = n[0];
  uint numSwaps = ((dim * (dim - (uint)(1u))) / (uint)(2u));
  bool inBounds = (gid < numSwaps);
  if (inBounds) {
    double gidF = ((double)(gid));
    double dimF = ((double)(dim));
    double twoNm1 = ((as_double(0x4000000000000000ul) * dimF) - as_double(0x3ff0000000000000ul));
    double disc = ((twoNm1 * twoNm1) - (as_double(0x4020000000000000ul) * gidF));
    double sqrtDisc = sqrt(disc);
    double iF = floor(((twoNm1 - sqrtDisc) / as_double(0x4000000000000000ul)));
    uint i = ((uint)(iF));
    uint rowStart = ((i * ((((uint)(2u) * dim) - i) - (uint)(1u))) / (uint)(2u));
    uint offset = (gid - rowStart);
    uint j = ((i + (uint)(1u)) + offset);
    uint idx_ij = ((i * dim) + j);
    uint idx_ji = ((j * dim) + i);
    double val_ij = A[idx_ij];
    double val_ji = A[idx_ji];
    A[idx_ij] = val_ji;
    A[idx_ji] = val_ij;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_transpose_batched_f32 = R"kernel(
__kernel void navatala_linalg_transpose_batched_f32(__global const float* A, __global const uint* m, __global const uint* n, __global const uint* batchSize, __global float* B) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
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
const char* k_opencl_navatala_linalg_transpose_batched_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_transpose_batched_f64(__global const double* A, __global const uint* m, __global const uint* n, __global const uint* batchSize, __global double* B) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
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
    double val = A[srcIdx];
    B[dstIdx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_map_f32 = R"kernel(
__kernel void navatala_linalg_map_f32(__global const float* A, __global const uint* m, __global const uint* n, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  if ((idx < (m[(uint)(0u)] * n[(uint)(0u)]))) {
    float val = A[idx];
    result[idx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_map_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_map_f64(__global const double* A, __global const uint* m, __global const uint* n, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  if ((idx < (m[(uint)(0u)] * n[(uint)(0u)]))) {
    double val = A[idx];
    result[idx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_reduce_row_sum_f32 = R"kernel(
__kernel void navatala_linalg_reduce_row_sum_f32(__global const float* A, __global const uint* m, __global const uint* n, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint row = gid;
  if ((row < m[(uint)(0u)])) {
    float sumAccum = as_float(0x00000000u);
    for (int j = 0; j < (int)(n[(uint)(0u)]); ++j) {
      uint jU32 = ((uint)(j));
      uint idx = ((row * n[(uint)(0u)]) + jU32);
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
const char* k_opencl_navatala_linalg_reduce_row_sum_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_reduce_row_sum_f64(__global const double* A, __global const uint* m, __global const uint* n, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint row = gid;
  if ((row < m[(uint)(0u)])) {
    double sumAccum = as_double(0x0000000000000000ul);
    for (int j = 0; j < (int)(n[(uint)(0u)]); ++j) {
      uint jU32 = ((uint)(j));
      uint idx = ((row * n[(uint)(0u)]) + jU32);
      double val = A[idx];
      double currentSum = sumAccum;
      double nextSum = (currentSum + val);
      sumAccum = nextSum;
    }
    double finalSum = sumAccum;
    result[row] = finalSum;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_reduce_col_sum_f32 = R"kernel(
__kernel void navatala_linalg_reduce_col_sum_f32(__global const float* A, __global const uint* m, __global const uint* n, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint col = gid;
  if ((col < n[(uint)(0u)])) {
    float sumAccum = as_float(0x00000000u);
    for (int i = 0; i < (int)(m[(uint)(0u)]); ++i) {
      uint iU32 = ((uint)(i));
      uint idx = ((iU32 * n[(uint)(0u)]) + col);
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
const char* k_opencl_navatala_linalg_reduce_col_sum_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_reduce_col_sum_f64(__global const double* A, __global const uint* m, __global const uint* n, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint col = gid;
  if ((col < n[(uint)(0u)])) {
    double sumAccum = as_double(0x0000000000000000ul);
    for (int i = 0; i < (int)(m[(uint)(0u)]); ++i) {
      uint iU32 = ((uint)(i));
      uint idx = ((iU32 * n[(uint)(0u)]) + col);
      double val = A[idx];
      double currentSum = sumAccum;
      double nextSum = (currentSum + val);
      sumAccum = nextSum;
    }
    double finalSum = sumAccum;
    result[col] = finalSum;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_reduce_row_max_f32 = R"kernel(
__kernel void navatala_linalg_reduce_row_max_f32(__global const float* A, __global const uint* m, __global const uint* n, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint row = gid;
  if ((row < m[(uint)(0u)])) {
    uint firstIdx = (row * n[(uint)(0u)]);
    float initMaxVal = A[firstIdx];
    float maxAccum = initMaxVal;
    for (int j = 0; j < (int)(n[(uint)(0u)]); ++j) {
      uint jU32 = ((uint)(j));
      uint idx = ((row * n[(uint)(0u)]) + jU32);
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
const char* k_opencl_navatala_linalg_reduce_row_max_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_reduce_row_max_f64(__global const double* A, __global const uint* m, __global const uint* n, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint row = gid;
  if ((row < m[(uint)(0u)])) {
    uint firstIdx = (row * n[(uint)(0u)]);
    double initMaxVal = A[firstIdx];
    double maxAccum = initMaxVal;
    for (int j = 0; j < (int)(n[(uint)(0u)]); ++j) {
      uint jU32 = ((uint)(j));
      uint idx = ((row * n[(uint)(0u)]) + jU32);
      double val = A[idx];
      double currentMax = maxAccum;
      double nextMax = (((val > currentMax)) ? (val) : (currentMax));
      maxAccum = nextMax;
    }
    double finalMax = maxAccum;
    result[row] = finalMax;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_reduce_col_max_f32 = R"kernel(
__kernel void navatala_linalg_reduce_col_max_f32(__global const float* A, __global const uint* m, __global const uint* n, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint col = gid;
  if ((col < n[(uint)(0u)])) {
    float initMaxVal = A[col];
    float maxAccum = initMaxVal;
    for (int i = 0; i < (int)(m[(uint)(0u)]); ++i) {
      uint iU32 = ((uint)(i));
      uint idx = ((iU32 * n[(uint)(0u)]) + col);
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
const char* k_opencl_navatala_linalg_reduce_col_max_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_reduce_col_max_f64(__global const double* A, __global const uint* m, __global const uint* n, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint col = gid;
  if ((col < n[(uint)(0u)])) {
    double initMaxVal = A[col];
    double maxAccum = initMaxVal;
    for (int i = 0; i < (int)(m[(uint)(0u)]); ++i) {
      uint iU32 = ((uint)(i));
      uint idx = ((iU32 * n[(uint)(0u)]) + col);
      double val = A[idx];
      double currentMax = maxAccum;
      double nextMax = (((val > currentMax)) ? (val) : (currentMax));
      maxAccum = nextMax;
    }
    double finalMax = maxAccum;
    result[col] = finalMax;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_map_reduce_sum_f32 = R"kernel(
__kernel void navatala_linalg_map_reduce_sum_f32(__global const float* A, __global const uint* m, __global const uint* n, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  if ((idx < (m[(uint)(0u)] * n[(uint)(0u)]))) {
    float val = A[idx];
    atomic_add(&result[(uint)(0u)], val);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_map_reduce_sum_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_map_reduce_sum_f64(__global const double* A, __global const uint* m, __global const uint* n, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  if ((idx < (m[(uint)(0u)] * n[(uint)(0u)]))) {
    double val = A[idx];
    atomic_add(&result[(uint)(0u)], val);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_outer_product_f32 = R"kernel(
__kernel void navatala_linalg_outer_product_f32(__global const float* a, __global const float* b, __global const uint* m, __global const uint* n, __global float* C) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint rows = m[(uint)(0u)];
  uint cols = n[(uint)(0u)];
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
const char* k_opencl_navatala_linalg_outer_product_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_outer_product_f64(__global const double* a, __global const double* b, __global const uint* m, __global const uint* n, __global double* C) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint rows = m[(uint)(0u)];
  uint cols = n[(uint)(0u)];
  uint totalElems = (rows * cols);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / cols);
    uint j = (gid % cols);
    double aVal = a[i];
    double bVal = b[j];
    uint outIdx = ((i * cols) + j);
    double product = (aVal * bVal);
    C[outIdx] = product;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_batched_dot_f32 = R"kernel(
__kernel void navatala_linalg_batched_dot_f32(__global const float* a, __global const float* b, __global const uint* vecLen, __global const uint* batchSize, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint len = vecLen[(uint)(0u)];
  uint numBatches = batchSize[(uint)(0u)];
  uint k = gid;
  bool inBounds = (k < numBatches);
  if (inBounds) {
    uint baseOffset = (k * len);
    float sumAccum = as_float(0x00000000u);
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
const char* k_opencl_navatala_linalg_batched_dot_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_batched_dot_f64(__global const double* a, __global const double* b, __global const uint* vecLen, __global const uint* batchSize, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint len = vecLen[(uint)(0u)];
  uint numBatches = batchSize[(uint)(0u)];
  uint k = gid;
  bool inBounds = (k < numBatches);
  if (inBounds) {
    uint baseOffset = (k * len);
    double sumAccum = as_double(0x0000000000000000ul);
    for (int i = 0; i < (int)(len); ++i) {
      uint iU32 = ((uint)(i));
      uint idx = (baseOffset + iU32);
      double aVal = a[idx];
      double bVal = b[idx];
      double prod = (aVal * bVal);
      double currentSum = sumAccum;
      double nextSum = (currentSum + prod);
      sumAccum = nextSum;
    }
    double finalSum = sumAccum;
    result[k] = finalSum;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_trace_f32 = R"kernel(
__kernel void navatala_linalg_trace_f32(__global const float* A, __global const uint* n, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint dim = n[(uint)(0u)];
  bool isThread0 = (gid == (uint)(0u));
  if (isThread0) {
    float traceAccum = as_float(0x00000000u);
    for (int i = 0; i < (int)(dim); ++i) {
      uint iU32 = ((uint)(i));
      uint diagIdx = ((iU32 * dim) + iU32);
      float diagVal = A[diagIdx];
      float currentTrace = traceAccum;
      float nextTrace = (currentTrace + diagVal);
      traceAccum = nextTrace;
    }
    float finalTrace = traceAccum;
    result[(uint)(0u)] = finalTrace;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_trace_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_trace_f64(__global const double* A, __global const uint* n, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint dim = n[(uint)(0u)];
  bool isThread0 = (gid == (uint)(0u));
  if (isThread0) {
    double traceAccum = as_double(0x0000000000000000ul);
    for (int i = 0; i < (int)(dim); ++i) {
      uint iU32 = ((uint)(i));
      uint diagIdx = ((iU32 * dim) + iU32);
      double diagVal = A[diagIdx];
      double currentTrace = traceAccum;
      double nextTrace = (currentTrace + diagVal);
      traceAccum = nextTrace;
    }
    double finalTrace = traceAccum;
    result[(uint)(0u)] = finalTrace;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_frobenius_norm_f32 = R"kernel(
__kernel void navatala_linalg_frobenius_norm_f32(__global const float* A, __global const uint* m, __global const uint* n, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint rows = m[(uint)(0u)];
  uint cols = n[(uint)(0u)];
  uint totalElems = (rows * cols);
  bool isThread0 = (gid == (uint)(0u));
  if (isThread0) {
    float sumSqAccum = as_float(0x00000000u);
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
    result[(uint)(0u)] = norm;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_frobenius_norm_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_frobenius_norm_f64(__global const double* A, __global const uint* m, __global const uint* n, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint rows = m[(uint)(0u)];
  uint cols = n[(uint)(0u)];
  uint totalElems = (rows * cols);
  bool isThread0 = (gid == (uint)(0u));
  if (isThread0) {
    double sumSqAccum = as_double(0x0000000000000000ul);
    for (int idx = 0; idx < (int)(totalElems); ++idx) {
      uint idxU32 = ((uint)(idx));
      double val = A[idxU32];
      double valSq = (val * val);
      double currentSumSq = sumSqAccum;
      double nextSumSq = (currentSumSq + valSq);
      sumSqAccum = nextSumSq;
    }
    double finalSumSq = sumSqAccum;
    double norm = sqrt(finalSumSq);
    result[(uint)(0u)] = norm;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_extract_upper_f32 = R"kernel(
__kernel void navatala_linalg_extract_upper_f32(__global const float* A, __global const uint* n, __global float* B) {
  int gid0 = (int)get_global_id(0);
  uint idx = ((uint)((int)(get_global_id(0))));
  uint nVal = n[(uint)(0u)];
  uint totalElements = (nVal * nVal);
  if ((idx < totalElements)) {
    uint i = (idx / nVal);
    uint j = (idx % nVal);
    float aVal = A[idx];
    bool isUpper = (i <= j);
    float result = ((isUpper) ? (aVal) : (as_float(0x00000000u)));
    B[idx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_extract_upper_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_extract_upper_f64(__global const double* A, __global const uint* n, __global double* B) {
  int gid0 = (int)get_global_id(0);
  uint idx = ((uint)((int)(get_global_id(0))));
  uint nVal = n[(uint)(0u)];
  uint totalElements = (nVal * nVal);
  if ((idx < totalElements)) {
    uint i = (idx / nVal);
    uint j = (idx % nVal);
    double aVal = A[idx];
    bool isUpper = (i <= j);
    double result = ((isUpper) ? (aVal) : (as_double(0x0000000000000000ul)));
    B[idx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_extract_lower_f32 = R"kernel(
__kernel void navatala_linalg_extract_lower_f32(__global const float* A, __global const uint* n, __global float* B) {
  int gid0 = (int)get_global_id(0);
  uint idx = ((uint)((int)(get_global_id(0))));
  uint nVal = n[(uint)(0u)];
  uint totalElements = (nVal * nVal);
  if ((idx < totalElements)) {
    uint i = (idx / nVal);
    uint j = (idx % nVal);
    float aVal = A[idx];
    bool isLower = (i >= j);
    float result = ((isLower) ? (aVal) : (as_float(0x00000000u)));
    B[idx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_extract_lower_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_extract_lower_f64(__global const double* A, __global const uint* n, __global double* B) {
  int gid0 = (int)get_global_id(0);
  uint idx = ((uint)((int)(get_global_id(0))));
  uint nVal = n[(uint)(0u)];
  uint totalElements = (nVal * nVal);
  if ((idx < totalElements)) {
    uint i = (idx / nVal);
    uint j = (idx % nVal);
    double aVal = A[idx];
    bool isLower = (i >= j);
    double result = ((isLower) ? (aVal) : (as_double(0x0000000000000000ul)));
    B[idx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_set_upper_f32 = R"kernel(
__kernel void navatala_linalg_set_upper_f32(__global const float* val, __global const uint* n, __global float* A) {
  int gid0 = (int)get_global_id(0);
  uint idx = ((uint)((int)(get_global_id(0))));
  uint nVal = n[(uint)(0u)];
  float fillVal = val[(uint)(0u)];
  uint totalElements = (nVal * nVal);
  if ((idx < totalElements)) {
    uint i = (idx / nVal);
    uint j = (idx % nVal);
    bool isStrictUpper = (i < j);
    if (isStrictUpper) {
      A[idx] = fillVal;
    }
  }
}

)kernel";
const char* k_opencl_navatala_linalg_set_upper_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_set_upper_f64(__global const double* val, __global const uint* n, __global double* A) {
  int gid0 = (int)get_global_id(0);
  uint idx = ((uint)((int)(get_global_id(0))));
  uint nVal = n[(uint)(0u)];
  double fillVal = val[(uint)(0u)];
  uint totalElements = (nVal * nVal);
  if ((idx < totalElements)) {
    uint i = (idx / nVal);
    uint j = (idx % nVal);
    bool isStrictUpper = (i < j);
    if (isStrictUpper) {
      A[idx] = fillVal;
    }
  }
}

)kernel";
const char* k_opencl_navatala_linalg_set_lower_f32 = R"kernel(
__kernel void navatala_linalg_set_lower_f32(__global const float* val, __global const uint* n, __global float* A) {
  int gid0 = (int)get_global_id(0);
  uint idx = ((uint)((int)(get_global_id(0))));
  uint nVal = n[(uint)(0u)];
  float fillVal = val[(uint)(0u)];
  uint totalElements = (nVal * nVal);
  if ((idx < totalElements)) {
    uint i = (idx / nVal);
    uint j = (idx % nVal);
    bool isStrictLower = (i > j);
    if (isStrictLower) {
      A[idx] = fillVal;
    }
  }
}

)kernel";
const char* k_opencl_navatala_linalg_set_lower_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_set_lower_f64(__global const double* val, __global const uint* n, __global double* A) {
  int gid0 = (int)get_global_id(0);
  uint idx = ((uint)((int)(get_global_id(0))));
  uint nVal = n[(uint)(0u)];
  double fillVal = val[(uint)(0u)];
  uint totalElements = (nVal * nVal);
  if ((idx < totalElements)) {
    uint i = (idx / nVal);
    uint j = (idx % nVal);
    bool isStrictLower = (i > j);
    if (isStrictLower) {
      A[idx] = fillVal;
    }
  }
}

)kernel";
const char* k_opencl_navatala_linalg_gather_rows_f32 = R"kernel(
__kernel void navatala_linalg_gather_rows_f32(__global const float* A, __global const uint* indices, __global const uint* k, __global const uint* n, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint kVal = k[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint totalElems = (kVal * nVal);
  if ((tid < totalElems)) {
    uint rowIdx = (tid / nVal);
    uint colIdx = (tid % nVal);
    uint srcRow = indices[rowIdx];
    uint srcIdx = ((srcRow * nVal) + colIdx);
    float val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_gather_rows_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_gather_rows_f64(__global const double* A, __global const uint* indices, __global const uint* k, __global const uint* n, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint kVal = k[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint totalElems = (kVal * nVal);
  if ((tid < totalElems)) {
    uint rowIdx = (tid / nVal);
    uint colIdx = (tid % nVal);
    uint srcRow = indices[rowIdx];
    uint srcIdx = ((srcRow * nVal) + colIdx);
    double val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_gather_cols_f32 = R"kernel(
__kernel void navatala_linalg_gather_cols_f32(__global const float* A, __global const uint* indices, __global const uint* m, __global const uint* n, __global const uint* kCount, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint kVal = kCount[(uint)(0u)];
  uint totalElems = (mVal * kVal);
  if ((tid < totalElems)) {
    uint rowIdx = (tid / kVal);
    uint outColIdx = (tid % kVal);
    uint srcCol = indices[outColIdx];
    uint srcIdx = ((rowIdx * nVal) + srcCol);
    float val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_gather_cols_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_gather_cols_f64(__global const double* A, __global const uint* indices, __global const uint* m, __global const uint* n, __global const uint* kCount, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint kVal = kCount[(uint)(0u)];
  uint totalElems = (mVal * kVal);
  if ((tid < totalElems)) {
    uint rowIdx = (tid / kVal);
    uint outColIdx = (tid % kVal);
    uint srcCol = indices[outColIdx];
    uint srcIdx = ((rowIdx * nVal) + srcCol);
    double val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_gather_elements_f32 = R"kernel(
__kernel void navatala_linalg_gather_elements_f32(__global const float* A, __global const uint* rowIdx, __global const uint* colIdx, __global const uint* count, __global const uint* n, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint countVal = count[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  if ((tid < countVal)) {
    uint row = rowIdx[tid];
    uint col = colIdx[tid];
    uint srcIdx = ((row * nVal) + col);
    float val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_gather_elements_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_gather_elements_f64(__global const double* A, __global const uint* rowIdx, __global const uint* colIdx, __global const uint* count, __global const uint* n, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint countVal = count[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  if ((tid < countVal)) {
    uint row = rowIdx[tid];
    uint col = colIdx[tid];
    uint srcIdx = ((row * nVal) + col);
    double val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_gather_batched_f32 = R"kernel(
__kernel void navatala_linalg_gather_batched_f32(__global const float* A, __global const uint* indices, __global const uint* batchSize, __global const uint* m, __global const uint* n, __global const uint* kCount, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint batchVal = batchSize[(uint)(0u)];
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint kVal = kCount[(uint)(0u)];
  uint outBatchStride = (kVal * nVal);
  uint totalElems = (batchVal * outBatchStride);
  if ((tid < totalElems)) {
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
const char* k_opencl_navatala_linalg_gather_batched_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_gather_batched_f64(__global const double* A, __global const uint* indices, __global const uint* batchSize, __global const uint* m, __global const uint* n, __global const uint* kCount, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint batchVal = batchSize[(uint)(0u)];
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint kVal = kCount[(uint)(0u)];
  uint outBatchStride = (kVal * nVal);
  uint totalElems = (batchVal * outBatchStride);
  if ((tid < totalElems)) {
    uint batchIdx = (tid / outBatchStride);
    uint withinBatch = (tid % outBatchStride);
    uint rowIdx = (withinBatch / nVal);
    uint colIdx = (withinBatch % nVal);
    uint indicesIdx = ((batchIdx * kVal) + rowIdx);
    uint srcRow = indices[indicesIdx];
    uint srcBatchStride = (mVal * nVal);
    uint srcIdx = (((batchIdx * srcBatchStride) + (srcRow * nVal)) + colIdx);
    double val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_scatter_rows_f32 = R"kernel(
__kernel void navatala_linalg_scatter_rows_f32(__global const float* src, __global const uint* indices, __global const uint* k, __global const uint* n, __global float* A) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint kVal = k[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint totalElems = (kVal * nVal);
  if ((tid < totalElems)) {
    uint rowIdx = (tid / nVal);
    uint colIdx = (tid % nVal);
    uint dstRow = indices[rowIdx];
    uint dstIdx = ((dstRow * nVal) + colIdx);
    float val = src[tid];
    A[dstIdx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_scatter_rows_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_scatter_rows_f64(__global const double* src, __global const uint* indices, __global const uint* k, __global const uint* n, __global double* A) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint kVal = k[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint totalElems = (kVal * nVal);
  if ((tid < totalElems)) {
    uint rowIdx = (tid / nVal);
    uint colIdx = (tid % nVal);
    uint dstRow = indices[rowIdx];
    uint dstIdx = ((dstRow * nVal) + colIdx);
    double val = src[tid];
    A[dstIdx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_scatter_cols_f32 = R"kernel(
__kernel void navatala_linalg_scatter_cols_f32(__global const float* src, __global const uint* indices, __global const uint* m, __global const uint* k, __global const uint* n, __global float* A) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint mVal = m[(uint)(0u)];
  uint kVal = k[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint totalElems = (mVal * kVal);
  if ((tid < totalElems)) {
    uint row = (tid / kVal);
    uint srcColIdx = (tid % kVal);
    uint dstCol = indices[srcColIdx];
    uint dstIdx = ((row * nVal) + dstCol);
    float val = src[tid];
    A[dstIdx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_scatter_cols_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_scatter_cols_f64(__global const double* src, __global const uint* indices, __global const uint* m, __global const uint* k, __global const uint* n, __global double* A) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint mVal = m[(uint)(0u)];
  uint kVal = k[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint totalElems = (mVal * kVal);
  if ((tid < totalElems)) {
    uint row = (tid / kVal);
    uint srcColIdx = (tid % kVal);
    uint dstCol = indices[srcColIdx];
    uint dstIdx = ((row * nVal) + dstCol);
    double val = src[tid];
    A[dstIdx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_scatter_add_f32 = R"kernel(
__kernel void navatala_linalg_scatter_add_f32(__global const float* src, __global const uint* indices, __global const uint* count, __global float* A) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint n = count[(uint)(0u)];
  if ((tid < n)) {
    uint dstIdx = indices[tid];
    float val = src[tid];
    atomic_add(&A[dstIdx], val);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_scatter_add_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_scatter_add_f64(__global const double* src, __global const uint* indices, __global const uint* count, __global double* A) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint n = count[(uint)(0u)];
  if ((tid < n)) {
    uint dstIdx = indices[tid];
    double val = src[tid];
    atomic_add(&A[dstIdx], val);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_scatter_max_f32 = R"kernel(
__kernel void navatala_linalg_scatter_max_f32(__global const float* src, __global const uint* indices, __global const uint* count, __global float* A) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint n = count[(uint)(0u)];
  if ((tid < n)) {
    uint dstIdx = indices[tid];
    float val = src[tid];
    atomic_max(&A[dstIdx], val);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_scatter_max_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_scatter_max_f64(__global const double* src, __global const uint* indices, __global const uint* count, __global double* A) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint tid = gid;
  uint n = count[(uint)(0u)];
  if ((tid < n)) {
    uint dstIdx = indices[tid];
    double val = src[tid];
    atomic_max(&A[dstIdx], val);
  }
}

)kernel";
const char* k_opencl_navatala_linalg_slice_rows_f32 = R"kernel(
__kernel void navatala_linalg_slice_rows_f32(__global const float* A, __global const uint* rowStart, __global const uint* rowEnd, __global const uint* numCols, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint rs = rowStart[(uint)(0u)];
  uint re = rowEnd[(uint)(0u)];
  uint n = numCols[(uint)(0u)];
  uint numRows = (re - rs);
  uint totalElems = (numRows * n);
  if ((idx < totalElems)) {
    uint outRow = (idx / n);
    uint outCol = (idx % n);
    uint srcRow = (rs + outRow);
    uint srcIdx = ((srcRow * n) + outCol);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_slice_rows_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_slice_rows_f64(__global const double* A, __global const uint* rowStart, __global const uint* rowEnd, __global const uint* numCols, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint rs = rowStart[(uint)(0u)];
  uint re = rowEnd[(uint)(0u)];
  uint n = numCols[(uint)(0u)];
  uint numRows = (re - rs);
  uint totalElems = (numRows * n);
  if ((idx < totalElems)) {
    uint outRow = (idx / n);
    uint outCol = (idx % n);
    uint srcRow = (rs + outRow);
    uint srcIdx = ((srcRow * n) + outCol);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_slice_cols_f32 = R"kernel(
__kernel void navatala_linalg_slice_cols_f32(__global const float* A, __global const uint* colStart, __global const uint* colEnd, __global const uint* numRows, __global const uint* srcCols, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint cs = colStart[(uint)(0u)];
  uint ce = colEnd[(uint)(0u)];
  uint m = numRows[(uint)(0u)];
  uint n = srcCols[(uint)(0u)];
  uint outCols = (ce - cs);
  uint totalElems = (m * outCols);
  if ((idx < totalElems)) {
    uint outRow = (idx / outCols);
    uint outCol = (idx % outCols);
    uint srcCol = (cs + outCol);
    uint srcIdx = ((outRow * n) + srcCol);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_slice_cols_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_slice_cols_f64(__global const double* A, __global const uint* colStart, __global const uint* colEnd, __global const uint* numRows, __global const uint* srcCols, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint cs = colStart[(uint)(0u)];
  uint ce = colEnd[(uint)(0u)];
  uint m = numRows[(uint)(0u)];
  uint n = srcCols[(uint)(0u)];
  uint outCols = (ce - cs);
  uint totalElems = (m * outCols);
  if ((idx < totalElems)) {
    uint outRow = (idx / outCols);
    uint outCol = (idx % outCols);
    uint srcCol = (cs + outCol);
    uint srcIdx = ((outRow * n) + srcCol);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_slice_block_f32 = R"kernel(
__kernel void navatala_linalg_slice_block_f32(__global const float* A, __global const uint* rowStart, __global const uint* rowEnd, __global const uint* colStart, __global const uint* colEnd, __global const uint* srcCols, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint rs = rowStart[(uint)(0u)];
  uint re = rowEnd[(uint)(0u)];
  uint cs = colStart[(uint)(0u)];
  uint ce = colEnd[(uint)(0u)];
  uint n = srcCols[(uint)(0u)];
  uint outRows = (re - rs);
  uint outCols = (ce - cs);
  uint totalElems = (outRows * outCols);
  if ((idx < totalElems)) {
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
const char* k_opencl_navatala_linalg_slice_block_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_slice_block_f64(__global const double* A, __global const uint* rowStart, __global const uint* rowEnd, __global const uint* colStart, __global const uint* colEnd, __global const uint* srcCols, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint rs = rowStart[(uint)(0u)];
  uint re = rowEnd[(uint)(0u)];
  uint cs = colStart[(uint)(0u)];
  uint ce = colEnd[(uint)(0u)];
  uint n = srcCols[(uint)(0u)];
  uint outRows = (re - rs);
  uint outCols = (ce - cs);
  uint totalElems = (outRows * outCols);
  if ((idx < totalElems)) {
    uint outRow = (idx / outCols);
    uint outCol = (idx % outCols);
    uint srcRow = (rs + outRow);
    uint srcCol = (cs + outCol);
    uint srcIdx = ((srcRow * n) + srcCol);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_strided_slice_f32 = R"kernel(
__kernel void navatala_linalg_strided_slice_f32(__global const float* A, __global const uint* start, __global const uint* stride, __global const uint* count, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint s = start[(uint)(0u)];
  uint st = stride[(uint)(0u)];
  uint n = count[(uint)(0u)];
  if ((idx < n)) {
    uint srcIdx = (s + (idx * st));
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_strided_slice_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_strided_slice_f64(__global const double* A, __global const uint* start, __global const uint* stride, __global const uint* count, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint s = start[(uint)(0u)];
  uint st = stride[(uint)(0u)];
  uint n = count[(uint)(0u)];
  if ((idx < n)) {
    uint srcIdx = (s + (idx * st));
    double val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_extract_diagonal_f32 = R"kernel(
__kernel void navatala_linalg_extract_diagonal_f32(__global const float* A, __global const uint* n, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint size = n[(uint)(0u)];
  if ((idx < size)) {
    uint stride = (size + (uint)(1u));
    uint srcIdx = (idx * stride);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_extract_diagonal_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_extract_diagonal_f64(__global const double* A, __global const uint* n, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint size = n[(uint)(0u)];
  if ((idx < size)) {
    uint stride = (size + (uint)(1u));
    uint srcIdx = (idx * stride);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_extract_diagonal_k_f32 = R"kernel(
__kernel void navatala_linalg_extract_diagonal_k_f32(__global const float* A, __global const uint* n, __global const uint* k, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint size = n[(uint)(0u)];
  uint offset = k[(uint)(0u)];
  uint diagLen = (size - offset);
  if ((idx < diagLen)) {
    uint row = idx;
    uint col = (idx + offset);
    uint srcIdx = ((row * size) + col);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_extract_diagonal_k_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_extract_diagonal_k_f64(__global const double* A, __global const uint* n, __global const uint* k, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint size = n[(uint)(0u)];
  uint offset = k[(uint)(0u)];
  uint diagLen = (size - offset);
  if ((idx < diagLen)) {
    uint row = idx;
    uint col = (idx + offset);
    uint srcIdx = ((row * size) + col);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_set_diagonal_f32 = R"kernel(
__kernel void navatala_linalg_set_diagonal_f32(__global const float* val, __global const uint* n, __global float* A) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint size = n[(uint)(0u)];
  if ((idx < size)) {
    uint stride = (size + (uint)(1u));
    uint destIdx = (idx * stride);
    float v = val[idx];
    A[destIdx] = v;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_set_diagonal_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_set_diagonal_f64(__global const double* val, __global const uint* n, __global double* A) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint size = n[(uint)(0u)];
  if ((idx < size)) {
    uint stride = (size + (uint)(1u));
    uint destIdx = (idx * stride);
    double v = val[idx];
    A[destIdx] = v;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_diag_to_matrix_f32 = R"kernel(
__kernel void navatala_linalg_diag_to_matrix_f32(__global const float* v, __global const uint* n, __global float* A) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint size = n[(uint)(0u)];
  uint totalElems = (size * size);
  if ((idx < totalElems)) {
    uint row = (idx / size);
    uint col = (idx % size);
    bool isDiag = (row == col);
    float diagVal = v[row];
    float result = ((isDiag) ? (diagVal) : (as_float(0x00000000u)));
    A[idx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_diag_to_matrix_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_diag_to_matrix_f64(__global const double* v, __global const uint* n, __global double* A) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint size = n[(uint)(0u)];
  uint totalElems = (size * size);
  if ((idx < totalElems)) {
    uint row = (idx / size);
    uint col = (idx % size);
    bool isDiag = (row == col);
    double diagVal = v[row];
    double result = ((isDiag) ? (diagVal) : (as_double(0x0000000000000000ul)));
    A[idx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_shift_rows_f32 = R"kernel(
__kernel void navatala_linalg_shift_rows_f32(__global const float* A, __global const uint* k, __global const uint* m, __global const uint* n, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint kVal = k[(uint)(0u)];
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint totalElems = (mVal * nVal);
  if ((idx < totalElems)) {
    uint row = (idx / nVal);
    uint col = (idx % nVal);
    bool inBounds = (row >= kVal);
    uint srcRow = (row - kVal);
    uint srcIdx = ((srcRow * nVal) + col);
    float srcVal = A[srcIdx];
    float result = ((inBounds) ? (srcVal) : (as_float(0x00000000u)));
    _out[idx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_shift_rows_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_shift_rows_f64(__global const double* A, __global const uint* k, __global const uint* m, __global const uint* n, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint kVal = k[(uint)(0u)];
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint totalElems = (mVal * nVal);
  if ((idx < totalElems)) {
    uint row = (idx / nVal);
    uint col = (idx % nVal);
    bool inBounds = (row >= kVal);
    uint srcRow = (row - kVal);
    uint srcIdx = ((srcRow * nVal) + col);
    double srcVal = A[srcIdx];
    double result = ((inBounds) ? (srcVal) : (as_double(0x0000000000000000ul)));
    _out[idx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_shift_cols_f32 = R"kernel(
__kernel void navatala_linalg_shift_cols_f32(__global const float* A, __global const uint* k, __global const uint* m, __global const uint* n, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint kVal = k[(uint)(0u)];
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint totalElems = (mVal * nVal);
  if ((idx < totalElems)) {
    uint row = (idx / nVal);
    uint col = (idx % nVal);
    bool inBounds = (col >= kVal);
    uint srcCol = (col - kVal);
    uint srcIdx = ((row * nVal) + srcCol);
    float srcVal = A[srcIdx];
    float result = ((inBounds) ? (srcVal) : (as_float(0x00000000u)));
    _out[idx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_shift_cols_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_shift_cols_f64(__global const double* A, __global const uint* k, __global const uint* m, __global const uint* n, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint kVal = k[(uint)(0u)];
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint totalElems = (mVal * nVal);
  if ((idx < totalElems)) {
    uint row = (idx / nVal);
    uint col = (idx % nVal);
    bool inBounds = (col >= kVal);
    uint srcCol = (col - kVal);
    uint srcIdx = ((row * nVal) + srcCol);
    double srcVal = A[srcIdx];
    double result = ((inBounds) ? (srcVal) : (as_double(0x0000000000000000ul)));
    _out[idx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_roll_rows_f32 = R"kernel(
__kernel void navatala_linalg_roll_rows_f32(__global const float* A, __global const uint* k, __global const uint* m, __global const uint* n, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint kVal = k[(uint)(0u)];
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint totalElems = (mVal * nVal);
  if ((idx < totalElems)) {
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
const char* k_opencl_navatala_linalg_roll_rows_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_roll_rows_f64(__global const double* A, __global const uint* k, __global const uint* m, __global const uint* n, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint kVal = k[(uint)(0u)];
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint totalElems = (mVal * nVal);
  if ((idx < totalElems)) {
    uint row = (idx / nVal);
    uint col = (idx % nVal);
    uint kMod = (kVal % mVal);
    uint rowPlusM = (row + mVal);
    uint rowMinusK = (rowPlusM - kMod);
    uint srcRow = (rowMinusK % mVal);
    uint srcIdx = ((srcRow * nVal) + col);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_roll_cols_f32 = R"kernel(
__kernel void navatala_linalg_roll_cols_f32(__global const float* A, __global const uint* k, __global const uint* m, __global const uint* n, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint kVal = k[(uint)(0u)];
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint totalElems = (mVal * nVal);
  if ((idx < totalElems)) {
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
const char* k_opencl_navatala_linalg_roll_cols_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_roll_cols_f64(__global const double* A, __global const uint* k, __global const uint* m, __global const uint* n, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint kVal = k[(uint)(0u)];
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  uint totalElems = (mVal * nVal);
  if ((idx < totalElems)) {
    uint row = (idx / nVal);
    uint col = (idx % nVal);
    uint kMod = (kVal % nVal);
    uint colPlusN = (col + nVal);
    uint colMinusK = (colPlusN - kMod);
    uint srcCol = (colMinusK % nVal);
    uint srcIdx = ((row * nVal) + srcCol);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_linalg_sort_rows_f32 = R"kernel(
__kernel void navatala_linalg_sort_rows_f32(__global const float* A, __global const uint* m, __global const uint* n, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  __local float svals[256];
  uint loopMyParity = (uint)(0u);
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  float loopMyVal = as_float(0x00000000u);
  uint loopPartnerIdx = (uint)(0u);
  float loopPartnerVal = as_float(0x00000000u);
  uint rowIdx = ((uint)((int)(get_group_id(0))));
  uint colIdx = ((uint)((int)(get_local_id(0))));
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  if ((rowIdx < mVal)) {
    bool inBounds = (colIdx < nVal);
    uint linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      float val = A[linearIdx];
      svals[colIdx] = val;
    } else {
      svals[colIdx] = as_float(0x7f7fffffu);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int phase = 0; phase < (int)(nVal); ++phase) {
      uint phaseU32 = ((uint)(phase));
      loopMyParity = (colIdx % (uint)(2u));
      uint myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % (uint)(2u)));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((colIdx + (uint)(1u)) < nVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[colIdx];
        float myValRead = loopMyVal;
        loopPartnerIdx = (colIdx + (uint)(1u));
        uint partnerIdxVal = loopPartnerIdx;
        loopPartnerVal = svals[partnerIdxVal];
        float partnerValRead = loopPartnerVal;
        if ((myValRead > partnerValRead)) {
          svals[colIdx] = partnerValRead;
          svals[partnerIdxVal] = myValRead;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (inBounds) {
      float sortedVal = svals[colIdx];
      _out[linearIdx] = sortedVal;
    }
  }
}

)kernel";
const char* k_opencl_navatala_linalg_sort_rows_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_sort_rows_f64(__global const double* A, __global const uint* m, __global const uint* n, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  __local double svals[256];
  uint loopMyParity = (uint)(0u);
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  double loopMyVal = as_double(0x0000000000000000ul);
  uint loopPartnerIdx = (uint)(0u);
  double loopPartnerVal = as_double(0x0000000000000000ul);
  uint rowIdx = ((uint)((int)(get_group_id(0))));
  uint colIdx = ((uint)((int)(get_local_id(0))));
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  if ((rowIdx < mVal)) {
    bool inBounds = (colIdx < nVal);
    uint linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      double val = A[linearIdx];
      svals[colIdx] = val;
    } else {
      svals[colIdx] = as_double(0x7feffffffffffffful);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int phase = 0; phase < (int)(nVal); ++phase) {
      uint phaseU32 = ((uint)(phase));
      loopMyParity = (colIdx % (uint)(2u));
      uint myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % (uint)(2u)));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((colIdx + (uint)(1u)) < nVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[colIdx];
        double myValRead = loopMyVal;
        loopPartnerIdx = (colIdx + (uint)(1u));
        uint partnerIdxVal = loopPartnerIdx;
        loopPartnerVal = svals[partnerIdxVal];
        double partnerValRead = loopPartnerVal;
        if ((myValRead > partnerValRead)) {
          svals[colIdx] = partnerValRead;
          svals[partnerIdxVal] = myValRead;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (inBounds) {
      double sortedVal = svals[colIdx];
      _out[linearIdx] = sortedVal;
    }
  }
}

)kernel";
const char* k_opencl_navatala_linalg_sort_cols_f32 = R"kernel(
__kernel void navatala_linalg_sort_cols_f32(__global const float* A, __global const uint* m, __global const uint* n, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  __local float svals[256];
  uint loopMyParity = (uint)(0u);
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  float loopMyVal = as_float(0x00000000u);
  uint loopPartnerIdx = (uint)(0u);
  float loopPartnerVal = as_float(0x00000000u);
  uint colIdx = ((uint)((int)(get_group_id(0))));
  uint rowIdx = ((uint)((int)(get_local_id(0))));
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  if ((colIdx < nVal)) {
    bool inBounds = (rowIdx < mVal);
    uint linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      float val = A[linearIdx];
      svals[rowIdx] = val;
    } else {
      svals[rowIdx] = as_float(0x7f7fffffu);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int phase = 0; phase < (int)(mVal); ++phase) {
      uint phaseU32 = ((uint)(phase));
      loopMyParity = (rowIdx % (uint)(2u));
      uint myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % (uint)(2u)));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((rowIdx + (uint)(1u)) < mVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[rowIdx];
        float myValRead = loopMyVal;
        loopPartnerIdx = (rowIdx + (uint)(1u));
        uint partnerIdxVal = loopPartnerIdx;
        loopPartnerVal = svals[partnerIdxVal];
        float partnerValRead = loopPartnerVal;
        if ((myValRead > partnerValRead)) {
          svals[rowIdx] = partnerValRead;
          svals[partnerIdxVal] = myValRead;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (inBounds) {
      float sortedVal = svals[rowIdx];
      _out[linearIdx] = sortedVal;
    }
  }
}

)kernel";
const char* k_opencl_navatala_linalg_sort_cols_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_sort_cols_f64(__global const double* A, __global const uint* m, __global const uint* n, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  __local double svals[256];
  uint loopMyParity = (uint)(0u);
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  double loopMyVal = as_double(0x0000000000000000ul);
  uint loopPartnerIdx = (uint)(0u);
  double loopPartnerVal = as_double(0x0000000000000000ul);
  uint colIdx = ((uint)((int)(get_group_id(0))));
  uint rowIdx = ((uint)((int)(get_local_id(0))));
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  if ((colIdx < nVal)) {
    bool inBounds = (rowIdx < mVal);
    uint linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      double val = A[linearIdx];
      svals[rowIdx] = val;
    } else {
      svals[rowIdx] = as_double(0x7feffffffffffffful);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int phase = 0; phase < (int)(mVal); ++phase) {
      uint phaseU32 = ((uint)(phase));
      loopMyParity = (rowIdx % (uint)(2u));
      uint myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % (uint)(2u)));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((rowIdx + (uint)(1u)) < mVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[rowIdx];
        double myValRead = loopMyVal;
        loopPartnerIdx = (rowIdx + (uint)(1u));
        uint partnerIdxVal = loopPartnerIdx;
        loopPartnerVal = svals[partnerIdxVal];
        double partnerValRead = loopPartnerVal;
        if ((myValRead > partnerValRead)) {
          svals[rowIdx] = partnerValRead;
          svals[partnerIdxVal] = myValRead;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (inBounds) {
      double sortedVal = svals[rowIdx];
      _out[linearIdx] = sortedVal;
    }
  }
}

)kernel";
const char* k_opencl_navatala_linalg_argsort_rows_f32 = R"kernel(
__kernel void navatala_linalg_argsort_rows_f32(__global const float* A, __global const uint* m, __global const uint* n, __global uint* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  __local float svals[256];
  __local uint sidxs[256];
  uint loopMyParity = (uint)(0u);
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  float loopMyVal = as_float(0x00000000u);
  uint loopMyIdx = (uint)(0u);
  uint loopPartnerPos = (uint)(0u);
  float loopPartnerVal = as_float(0x00000000u);
  uint loopPartnerIdx = (uint)(0u);
  uint rowIdx = ((uint)((int)(get_group_id(0))));
  uint colIdx = ((uint)((int)(get_local_id(0))));
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  if ((rowIdx < mVal)) {
    bool inBounds = (colIdx < nVal);
    uint linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      float val = A[linearIdx];
      svals[colIdx] = val;
      sidxs[colIdx] = colIdx;
    } else {
      svals[colIdx] = as_float(0x7f7fffffu);
      sidxs[colIdx] = colIdx;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int phase = 0; phase < (int)(nVal); ++phase) {
      uint phaseU32 = ((uint)(phase));
      loopMyParity = (colIdx % (uint)(2u));
      uint myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % (uint)(2u)));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((colIdx + (uint)(1u)) < nVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[colIdx];
        float myValRead = loopMyVal;
        loopMyIdx = sidxs[colIdx];
        uint myIdxRead = loopMyIdx;
        loopPartnerPos = (colIdx + (uint)(1u));
        uint partnerPosVal = loopPartnerPos;
        loopPartnerVal = svals[partnerPosVal];
        float partnerValRead = loopPartnerVal;
        loopPartnerIdx = sidxs[partnerPosVal];
        uint partnerIdxRead = loopPartnerIdx;
        if ((myValRead > partnerValRead)) {
          svals[colIdx] = partnerValRead;
          sidxs[colIdx] = partnerIdxRead;
          svals[partnerPosVal] = myValRead;
          sidxs[partnerPosVal] = myIdxRead;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (inBounds) {
      uint sortedIdx = sidxs[colIdx];
      _out[linearIdx] = sortedIdx;
    }
  }
}

)kernel";
const char* k_opencl_navatala_linalg_argsort_rows_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_argsort_rows_f64(__global const double* A, __global const uint* m, __global const uint* n, __global uint* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  __local double svals[256];
  __local uint sidxs[256];
  uint loopMyParity = (uint)(0u);
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  double loopMyVal = as_double(0x0000000000000000ul);
  uint loopMyIdx = (uint)(0u);
  uint loopPartnerPos = (uint)(0u);
  double loopPartnerVal = as_double(0x0000000000000000ul);
  uint loopPartnerIdx = (uint)(0u);
  uint rowIdx = ((uint)((int)(get_group_id(0))));
  uint colIdx = ((uint)((int)(get_local_id(0))));
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  if ((rowIdx < mVal)) {
    bool inBounds = (colIdx < nVal);
    uint linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      double val = A[linearIdx];
      svals[colIdx] = val;
      sidxs[colIdx] = colIdx;
    } else {
      svals[colIdx] = as_double(0x7feffffffffffffful);
      sidxs[colIdx] = colIdx;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int phase = 0; phase < (int)(nVal); ++phase) {
      uint phaseU32 = ((uint)(phase));
      loopMyParity = (colIdx % (uint)(2u));
      uint myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % (uint)(2u)));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((colIdx + (uint)(1u)) < nVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[colIdx];
        double myValRead = loopMyVal;
        loopMyIdx = sidxs[colIdx];
        uint myIdxRead = loopMyIdx;
        loopPartnerPos = (colIdx + (uint)(1u));
        uint partnerPosVal = loopPartnerPos;
        loopPartnerVal = svals[partnerPosVal];
        double partnerValRead = loopPartnerVal;
        loopPartnerIdx = sidxs[partnerPosVal];
        uint partnerIdxRead = loopPartnerIdx;
        if ((myValRead > partnerValRead)) {
          svals[colIdx] = partnerValRead;
          sidxs[colIdx] = partnerIdxRead;
          svals[partnerPosVal] = myValRead;
          sidxs[partnerPosVal] = myIdxRead;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (inBounds) {
      uint sortedIdx = sidxs[colIdx];
      _out[linearIdx] = sortedIdx;
    }
  }
}

)kernel";
const char* k_opencl_navatala_linalg_argsort_cols_f32 = R"kernel(
__kernel void navatala_linalg_argsort_cols_f32(__global const float* A, __global const uint* m, __global const uint* n, __global uint* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  __local float svals[256];
  __local uint sidxs[256];
  uint loopMyParity = (uint)(0u);
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  float loopMyVal = as_float(0x00000000u);
  uint loopMyIdx = (uint)(0u);
  uint loopPartnerPos = (uint)(0u);
  float loopPartnerVal = as_float(0x00000000u);
  uint loopPartnerIdx = (uint)(0u);
  uint colIdx = ((uint)((int)(get_group_id(0))));
  uint rowIdx = ((uint)((int)(get_local_id(0))));
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  if ((colIdx < nVal)) {
    bool inBounds = (rowIdx < mVal);
    uint linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      float val = A[linearIdx];
      svals[rowIdx] = val;
      sidxs[rowIdx] = rowIdx;
    } else {
      svals[rowIdx] = as_float(0x7f7fffffu);
      sidxs[rowIdx] = rowIdx;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int phase = 0; phase < (int)(mVal); ++phase) {
      uint phaseU32 = ((uint)(phase));
      loopMyParity = (rowIdx % (uint)(2u));
      uint myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % (uint)(2u)));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((rowIdx + (uint)(1u)) < mVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[rowIdx];
        float myValRead = loopMyVal;
        loopMyIdx = sidxs[rowIdx];
        uint myIdxRead = loopMyIdx;
        loopPartnerPos = (rowIdx + (uint)(1u));
        uint partnerPosVal = loopPartnerPos;
        loopPartnerVal = svals[partnerPosVal];
        float partnerValRead = loopPartnerVal;
        loopPartnerIdx = sidxs[partnerPosVal];
        uint partnerIdxRead = loopPartnerIdx;
        if ((myValRead > partnerValRead)) {
          svals[rowIdx] = partnerValRead;
          sidxs[rowIdx] = partnerIdxRead;
          svals[partnerPosVal] = myValRead;
          sidxs[partnerPosVal] = myIdxRead;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (inBounds) {
      uint sortedIdx = sidxs[rowIdx];
      _out[linearIdx] = sortedIdx;
    }
  }
}

)kernel";
const char* k_opencl_navatala_linalg_argsort_cols_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_linalg_argsort_cols_f64(__global const double* A, __global const uint* m, __global const uint* n, __global uint* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  __local double svals[256];
  __local uint sidxs[256];
  uint loopMyParity = (uint)(0u);
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  double loopMyVal = as_double(0x0000000000000000ul);
  uint loopMyIdx = (uint)(0u);
  uint loopPartnerPos = (uint)(0u);
  double loopPartnerVal = as_double(0x0000000000000000ul);
  uint loopPartnerIdx = (uint)(0u);
  uint colIdx = ((uint)((int)(get_group_id(0))));
  uint rowIdx = ((uint)((int)(get_local_id(0))));
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  if ((colIdx < nVal)) {
    bool inBounds = (rowIdx < mVal);
    uint linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      double val = A[linearIdx];
      svals[rowIdx] = val;
      sidxs[rowIdx] = rowIdx;
    } else {
      svals[rowIdx] = as_double(0x7feffffffffffffful);
      sidxs[rowIdx] = rowIdx;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int phase = 0; phase < (int)(mVal); ++phase) {
      uint phaseU32 = ((uint)(phase));
      loopMyParity = (rowIdx % (uint)(2u));
      uint myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % (uint)(2u)));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((rowIdx + (uint)(1u)) < mVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[rowIdx];
        double myValRead = loopMyVal;
        loopMyIdx = sidxs[rowIdx];
        uint myIdxRead = loopMyIdx;
        loopPartnerPos = (rowIdx + (uint)(1u));
        uint partnerPosVal = loopPartnerPos;
        loopPartnerVal = svals[partnerPosVal];
        double partnerValRead = loopPartnerVal;
        loopPartnerIdx = sidxs[partnerPosVal];
        uint partnerIdxRead = loopPartnerIdx;
        if ((myValRead > partnerValRead)) {
          svals[rowIdx] = partnerValRead;
          sidxs[rowIdx] = partnerIdxRead;
          svals[partnerPosVal] = myValRead;
          sidxs[partnerPosVal] = myIdxRead;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (inBounds) {
      uint sortedIdx = sidxs[rowIdx];
      _out[linearIdx] = sortedIdx;
    }
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_norm_linf_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_norm_linf_f32 = {
  1,
  "navatala_linalg_norm_linf_f32",
  "opencl",
  "navatala_linalg_norm_linf_f32",
  "kernel:opencl:navatala_linalg_norm_linf_f32",
  "abi-r1:opencl:navatala_linalg_norm_linf_f32",
  "abi-r1:opencl:navatala_linalg_norm_linf_f32",
  3,
  kAbiArgs_opencl_navatala_linalg_norm_linf_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_norm_linf_f64[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_norm_linf_f64 = {
  1,
  "navatala_linalg_norm_linf_f64",
  "opencl",
  "navatala_linalg_norm_linf_f64",
  "kernel:opencl:navatala_linalg_norm_linf_f64",
  "abi-r1:opencl:navatala_linalg_norm_linf_f64",
  "abi-r1:opencl:navatala_linalg_norm_linf_f64",
  3,
  kAbiArgs_opencl_navatala_linalg_norm_linf_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_eig_select_top_k_f32[] = {
  { "eigenvalues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "eigenvectors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "topKValues", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "topKVectors", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_eig_select_top_k_f32 = {
  1,
  "navatala_linalg_eig_select_top_k_f32",
  "opencl",
  "navatala_linalg_eig_select_top_k_f32",
  "kernel:opencl:navatala_linalg_eig_select_top_k_f32",
  "abi-r1:opencl:navatala_linalg_eig_select_top_k_f32",
  "abi-r1:opencl:navatala_linalg_eig_select_top_k_f32",
  6,
  kAbiArgs_opencl_navatala_linalg_eig_select_top_k_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_eig_select_top_k_f64[] = {
  { "eigenvalues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "eigenvectors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "topKValues", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "topKVectors", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_eig_select_top_k_f64 = {
  1,
  "navatala_linalg_eig_select_top_k_f64",
  "opencl",
  "navatala_linalg_eig_select_top_k_f64",
  "kernel:opencl:navatala_linalg_eig_select_top_k_f64",
  "abi-r1:opencl:navatala_linalg_eig_select_top_k_f64",
  "abi-r1:opencl:navatala_linalg_eig_select_top_k_f64",
  6,
  kAbiArgs_opencl_navatala_linalg_eig_select_top_k_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_cast_f32_to_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_cast_f32_to_f16 = {
  1,
  "navatala_linalg_cast_f32_to_f16",
  "opencl",
  "navatala_linalg_cast_f32_to_f16",
  "kernel:opencl:navatala_linalg_cast_f32_to_f16",
  "abi-r1:opencl:navatala_linalg_cast_f32_to_f16",
  "abi-r1:opencl:navatala_linalg_cast_f32_to_f16",
  3,
  kAbiArgs_opencl_navatala_linalg_cast_f32_to_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_cast_f64_to_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_cast_f64_to_f16 = {
  1,
  "navatala_linalg_cast_f64_to_f16",
  "opencl",
  "navatala_linalg_cast_f64_to_f16",
  "kernel:opencl:navatala_linalg_cast_f64_to_f16",
  "abi-r1:opencl:navatala_linalg_cast_f64_to_f16",
  "abi-r1:opencl:navatala_linalg_cast_f64_to_f16",
  3,
  kAbiArgs_opencl_navatala_linalg_cast_f64_to_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_cast_f16_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_cast_f16_to_f32 = {
  1,
  "navatala_linalg_cast_f16_to_f32",
  "opencl",
  "navatala_linalg_cast_f16_to_f32",
  "kernel:opencl:navatala_linalg_cast_f16_to_f32",
  "abi-r1:opencl:navatala_linalg_cast_f16_to_f32",
  "abi-r1:opencl:navatala_linalg_cast_f16_to_f32",
  3,
  kAbiArgs_opencl_navatala_linalg_cast_f16_to_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_cast_f16_to_f64[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_cast_f16_to_f64 = {
  1,
  "navatala_linalg_cast_f16_to_f64",
  "opencl",
  "navatala_linalg_cast_f16_to_f64",
  "kernel:opencl:navatala_linalg_cast_f16_to_f64",
  "abi-r1:opencl:navatala_linalg_cast_f16_to_f64",
  "abi-r1:opencl:navatala_linalg_cast_f16_to_f64",
  3,
  kAbiArgs_opencl_navatala_linalg_cast_f16_to_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_reduce_sum_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_reduce_sum_f16 = {
  1,
  "navatala_linalg_reduce_sum_f16",
  "opencl",
  "navatala_linalg_reduce_sum_f16",
  "kernel:opencl:navatala_linalg_reduce_sum_f16",
  "abi-r1:opencl:navatala_linalg_reduce_sum_f16",
  "abi-r1:opencl:navatala_linalg_reduce_sum_f16",
  3,
  kAbiArgs_opencl_navatala_linalg_reduce_sum_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_reduce_max_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_reduce_max_f16 = {
  1,
  "navatala_linalg_reduce_max_f16",
  "opencl",
  "navatala_linalg_reduce_max_f16",
  "kernel:opencl:navatala_linalg_reduce_max_f16",
  "abi-r1:opencl:navatala_linalg_reduce_max_f16",
  "abi-r1:opencl:navatala_linalg_reduce_max_f16",
  3,
  kAbiArgs_opencl_navatala_linalg_reduce_max_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_add_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_add_f32 = {
  1,
  "navatala_linalg_elt_add_f32",
  "opencl",
  "navatala_linalg_elt_add_f32",
  "kernel:opencl:navatala_linalg_elt_add_f32",
  "abi-r1:opencl:navatala_linalg_elt_add_f32",
  "abi-r1:opencl:navatala_linalg_elt_add_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_elt_add_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_add_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_add_f64 = {
  1,
  "navatala_linalg_elt_add_f64",
  "opencl",
  "navatala_linalg_elt_add_f64",
  "kernel:opencl:navatala_linalg_elt_add_f64",
  "abi-r1:opencl:navatala_linalg_elt_add_f64",
  "abi-r1:opencl:navatala_linalg_elt_add_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_elt_add_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_sub_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_sub_f32 = {
  1,
  "navatala_linalg_elt_sub_f32",
  "opencl",
  "navatala_linalg_elt_sub_f32",
  "kernel:opencl:navatala_linalg_elt_sub_f32",
  "abi-r1:opencl:navatala_linalg_elt_sub_f32",
  "abi-r1:opencl:navatala_linalg_elt_sub_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_elt_sub_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_sub_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_sub_f64 = {
  1,
  "navatala_linalg_elt_sub_f64",
  "opencl",
  "navatala_linalg_elt_sub_f64",
  "kernel:opencl:navatala_linalg_elt_sub_f64",
  "abi-r1:opencl:navatala_linalg_elt_sub_f64",
  "abi-r1:opencl:navatala_linalg_elt_sub_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_elt_sub_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_mul_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_mul_f32 = {
  1,
  "navatala_linalg_elt_mul_f32",
  "opencl",
  "navatala_linalg_elt_mul_f32",
  "kernel:opencl:navatala_linalg_elt_mul_f32",
  "abi-r1:opencl:navatala_linalg_elt_mul_f32",
  "abi-r1:opencl:navatala_linalg_elt_mul_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_elt_mul_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_mul_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_mul_f64 = {
  1,
  "navatala_linalg_elt_mul_f64",
  "opencl",
  "navatala_linalg_elt_mul_f64",
  "kernel:opencl:navatala_linalg_elt_mul_f64",
  "abi-r1:opencl:navatala_linalg_elt_mul_f64",
  "abi-r1:opencl:navatala_linalg_elt_mul_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_elt_mul_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_div_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_div_f32 = {
  1,
  "navatala_linalg_elt_div_f32",
  "opencl",
  "navatala_linalg_elt_div_f32",
  "kernel:opencl:navatala_linalg_elt_div_f32",
  "abi-r1:opencl:navatala_linalg_elt_div_f32",
  "abi-r1:opencl:navatala_linalg_elt_div_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_elt_div_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_div_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_div_f64 = {
  1,
  "navatala_linalg_elt_div_f64",
  "opencl",
  "navatala_linalg_elt_div_f64",
  "kernel:opencl:navatala_linalg_elt_div_f64",
  "abi-r1:opencl:navatala_linalg_elt_div_f64",
  "abi-r1:opencl:navatala_linalg_elt_div_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_elt_div_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_pow_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "p", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_pow_f32 = {
  1,
  "navatala_linalg_elt_pow_f32",
  "opencl",
  "navatala_linalg_elt_pow_f32",
  "kernel:opencl:navatala_linalg_elt_pow_f32",
  "abi-r1:opencl:navatala_linalg_elt_pow_f32",
  "abi-r1:opencl:navatala_linalg_elt_pow_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_elt_pow_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_pow_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "p", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_pow_f64 = {
  1,
  "navatala_linalg_elt_pow_f64",
  "opencl",
  "navatala_linalg_elt_pow_f64",
  "kernel:opencl:navatala_linalg_elt_pow_f64",
  "abi-r1:opencl:navatala_linalg_elt_pow_f64",
  "abi-r1:opencl:navatala_linalg_elt_pow_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_elt_pow_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_sqrt_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_sqrt_f32 = {
  1,
  "navatala_linalg_elt_sqrt_f32",
  "opencl",
  "navatala_linalg_elt_sqrt_f32",
  "kernel:opencl:navatala_linalg_elt_sqrt_f32",
  "abi-r1:opencl:navatala_linalg_elt_sqrt_f32",
  "abi-r1:opencl:navatala_linalg_elt_sqrt_f32",
  3,
  kAbiArgs_opencl_navatala_linalg_elt_sqrt_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_sqrt_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_sqrt_f64 = {
  1,
  "navatala_linalg_elt_sqrt_f64",
  "opencl",
  "navatala_linalg_elt_sqrt_f64",
  "kernel:opencl:navatala_linalg_elt_sqrt_f64",
  "abi-r1:opencl:navatala_linalg_elt_sqrt_f64",
  "abi-r1:opencl:navatala_linalg_elt_sqrt_f64",
  3,
  kAbiArgs_opencl_navatala_linalg_elt_sqrt_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_exp_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_exp_f32 = {
  1,
  "navatala_linalg_elt_exp_f32",
  "opencl",
  "navatala_linalg_elt_exp_f32",
  "kernel:opencl:navatala_linalg_elt_exp_f32",
  "abi-r1:opencl:navatala_linalg_elt_exp_f32",
  "abi-r1:opencl:navatala_linalg_elt_exp_f32",
  3,
  kAbiArgs_opencl_navatala_linalg_elt_exp_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_exp_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_exp_f64 = {
  1,
  "navatala_linalg_elt_exp_f64",
  "opencl",
  "navatala_linalg_elt_exp_f64",
  "kernel:opencl:navatala_linalg_elt_exp_f64",
  "abi-r1:opencl:navatala_linalg_elt_exp_f64",
  "abi-r1:opencl:navatala_linalg_elt_exp_f64",
  3,
  kAbiArgs_opencl_navatala_linalg_elt_exp_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_log_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_log_f32 = {
  1,
  "navatala_linalg_elt_log_f32",
  "opencl",
  "navatala_linalg_elt_log_f32",
  "kernel:opencl:navatala_linalg_elt_log_f32",
  "abi-r1:opencl:navatala_linalg_elt_log_f32",
  "abi-r1:opencl:navatala_linalg_elt_log_f32",
  3,
  kAbiArgs_opencl_navatala_linalg_elt_log_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_log_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_log_f64 = {
  1,
  "navatala_linalg_elt_log_f64",
  "opencl",
  "navatala_linalg_elt_log_f64",
  "kernel:opencl:navatala_linalg_elt_log_f64",
  "abi-r1:opencl:navatala_linalg_elt_log_f64",
  "abi-r1:opencl:navatala_linalg_elt_log_f64",
  3,
  kAbiArgs_opencl_navatala_linalg_elt_log_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_abs_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_abs_f32 = {
  1,
  "navatala_linalg_elt_abs_f32",
  "opencl",
  "navatala_linalg_elt_abs_f32",
  "kernel:opencl:navatala_linalg_elt_abs_f32",
  "abi-r1:opencl:navatala_linalg_elt_abs_f32",
  "abi-r1:opencl:navatala_linalg_elt_abs_f32",
  3,
  kAbiArgs_opencl_navatala_linalg_elt_abs_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_abs_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_abs_f64 = {
  1,
  "navatala_linalg_elt_abs_f64",
  "opencl",
  "navatala_linalg_elt_abs_f64",
  "kernel:opencl:navatala_linalg_elt_abs_f64",
  "abi-r1:opencl:navatala_linalg_elt_abs_f64",
  "abi-r1:opencl:navatala_linalg_elt_abs_f64",
  3,
  kAbiArgs_opencl_navatala_linalg_elt_abs_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_sign_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_sign_f32 = {
  1,
  "navatala_linalg_elt_sign_f32",
  "opencl",
  "navatala_linalg_elt_sign_f32",
  "kernel:opencl:navatala_linalg_elt_sign_f32",
  "abi-r1:opencl:navatala_linalg_elt_sign_f32",
  "abi-r1:opencl:navatala_linalg_elt_sign_f32",
  3,
  kAbiArgs_opencl_navatala_linalg_elt_sign_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_elt_sign_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_elt_sign_f64 = {
  1,
  "navatala_linalg_elt_sign_f64",
  "opencl",
  "navatala_linalg_elt_sign_f64",
  "kernel:opencl:navatala_linalg_elt_sign_f64",
  "abi-r1:opencl:navatala_linalg_elt_sign_f64",
  "abi-r1:opencl:navatala_linalg_elt_sign_f64",
  3,
  kAbiArgs_opencl_navatala_linalg_elt_sign_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_transpose_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_transpose_f32 = {
  1,
  "navatala_linalg_transpose_f32",
  "opencl",
  "navatala_linalg_transpose_f32",
  "kernel:opencl:navatala_linalg_transpose_f32",
  "abi-r1:opencl:navatala_linalg_transpose_f32",
  "abi-r1:opencl:navatala_linalg_transpose_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_transpose_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_transpose_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_transpose_f64 = {
  1,
  "navatala_linalg_transpose_f64",
  "opencl",
  "navatala_linalg_transpose_f64",
  "kernel:opencl:navatala_linalg_transpose_f64",
  "abi-r1:opencl:navatala_linalg_transpose_f64",
  "abi-r1:opencl:navatala_linalg_transpose_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_transpose_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_transpose_in_place_f32[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_transpose_in_place_f32 = {
  1,
  "navatala_linalg_transpose_in_place_f32",
  "opencl",
  "navatala_linalg_transpose_in_place_f32",
  "kernel:opencl:navatala_linalg_transpose_in_place_f32",
  "abi-r1:opencl:navatala_linalg_transpose_in_place_f32",
  "abi-r1:opencl:navatala_linalg_transpose_in_place_f32",
  2,
  kAbiArgs_opencl_navatala_linalg_transpose_in_place_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_transpose_in_place_f64[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_transpose_in_place_f64 = {
  1,
  "navatala_linalg_transpose_in_place_f64",
  "opencl",
  "navatala_linalg_transpose_in_place_f64",
  "kernel:opencl:navatala_linalg_transpose_in_place_f64",
  "abi-r1:opencl:navatala_linalg_transpose_in_place_f64",
  "abi-r1:opencl:navatala_linalg_transpose_in_place_f64",
  2,
  kAbiArgs_opencl_navatala_linalg_transpose_in_place_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_transpose_batched_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_transpose_batched_f32 = {
  1,
  "navatala_linalg_transpose_batched_f32",
  "opencl",
  "navatala_linalg_transpose_batched_f32",
  "kernel:opencl:navatala_linalg_transpose_batched_f32",
  "abi-r1:opencl:navatala_linalg_transpose_batched_f32",
  "abi-r1:opencl:navatala_linalg_transpose_batched_f32",
  5,
  kAbiArgs_opencl_navatala_linalg_transpose_batched_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_transpose_batched_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_transpose_batched_f64 = {
  1,
  "navatala_linalg_transpose_batched_f64",
  "opencl",
  "navatala_linalg_transpose_batched_f64",
  "kernel:opencl:navatala_linalg_transpose_batched_f64",
  "abi-r1:opencl:navatala_linalg_transpose_batched_f64",
  "abi-r1:opencl:navatala_linalg_transpose_batched_f64",
  5,
  kAbiArgs_opencl_navatala_linalg_transpose_batched_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_map_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_map_f32 = {
  1,
  "navatala_linalg_map_f32",
  "opencl",
  "navatala_linalg_map_f32",
  "kernel:opencl:navatala_linalg_map_f32",
  "abi-r1:opencl:navatala_linalg_map_f32",
  "abi-r1:opencl:navatala_linalg_map_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_map_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_map_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_map_f64 = {
  1,
  "navatala_linalg_map_f64",
  "opencl",
  "navatala_linalg_map_f64",
  "kernel:opencl:navatala_linalg_map_f64",
  "abi-r1:opencl:navatala_linalg_map_f64",
  "abi-r1:opencl:navatala_linalg_map_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_map_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_reduce_row_sum_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_reduce_row_sum_f32 = {
  1,
  "navatala_linalg_reduce_row_sum_f32",
  "opencl",
  "navatala_linalg_reduce_row_sum_f32",
  "kernel:opencl:navatala_linalg_reduce_row_sum_f32",
  "abi-r1:opencl:navatala_linalg_reduce_row_sum_f32",
  "abi-r1:opencl:navatala_linalg_reduce_row_sum_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_reduce_row_sum_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_reduce_row_sum_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_reduce_row_sum_f64 = {
  1,
  "navatala_linalg_reduce_row_sum_f64",
  "opencl",
  "navatala_linalg_reduce_row_sum_f64",
  "kernel:opencl:navatala_linalg_reduce_row_sum_f64",
  "abi-r1:opencl:navatala_linalg_reduce_row_sum_f64",
  "abi-r1:opencl:navatala_linalg_reduce_row_sum_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_reduce_row_sum_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_reduce_col_sum_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_reduce_col_sum_f32 = {
  1,
  "navatala_linalg_reduce_col_sum_f32",
  "opencl",
  "navatala_linalg_reduce_col_sum_f32",
  "kernel:opencl:navatala_linalg_reduce_col_sum_f32",
  "abi-r1:opencl:navatala_linalg_reduce_col_sum_f32",
  "abi-r1:opencl:navatala_linalg_reduce_col_sum_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_reduce_col_sum_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_reduce_col_sum_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_reduce_col_sum_f64 = {
  1,
  "navatala_linalg_reduce_col_sum_f64",
  "opencl",
  "navatala_linalg_reduce_col_sum_f64",
  "kernel:opencl:navatala_linalg_reduce_col_sum_f64",
  "abi-r1:opencl:navatala_linalg_reduce_col_sum_f64",
  "abi-r1:opencl:navatala_linalg_reduce_col_sum_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_reduce_col_sum_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_reduce_row_max_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_reduce_row_max_f32 = {
  1,
  "navatala_linalg_reduce_row_max_f32",
  "opencl",
  "navatala_linalg_reduce_row_max_f32",
  "kernel:opencl:navatala_linalg_reduce_row_max_f32",
  "abi-r1:opencl:navatala_linalg_reduce_row_max_f32",
  "abi-r1:opencl:navatala_linalg_reduce_row_max_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_reduce_row_max_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_reduce_row_max_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_reduce_row_max_f64 = {
  1,
  "navatala_linalg_reduce_row_max_f64",
  "opencl",
  "navatala_linalg_reduce_row_max_f64",
  "kernel:opencl:navatala_linalg_reduce_row_max_f64",
  "abi-r1:opencl:navatala_linalg_reduce_row_max_f64",
  "abi-r1:opencl:navatala_linalg_reduce_row_max_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_reduce_row_max_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_reduce_col_max_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_reduce_col_max_f32 = {
  1,
  "navatala_linalg_reduce_col_max_f32",
  "opencl",
  "navatala_linalg_reduce_col_max_f32",
  "kernel:opencl:navatala_linalg_reduce_col_max_f32",
  "abi-r1:opencl:navatala_linalg_reduce_col_max_f32",
  "abi-r1:opencl:navatala_linalg_reduce_col_max_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_reduce_col_max_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_reduce_col_max_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_reduce_col_max_f64 = {
  1,
  "navatala_linalg_reduce_col_max_f64",
  "opencl",
  "navatala_linalg_reduce_col_max_f64",
  "kernel:opencl:navatala_linalg_reduce_col_max_f64",
  "abi-r1:opencl:navatala_linalg_reduce_col_max_f64",
  "abi-r1:opencl:navatala_linalg_reduce_col_max_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_reduce_col_max_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_map_reduce_sum_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_map_reduce_sum_f32 = {
  1,
  "navatala_linalg_map_reduce_sum_f32",
  "opencl",
  "navatala_linalg_map_reduce_sum_f32",
  "kernel:opencl:navatala_linalg_map_reduce_sum_f32",
  "abi-r1:opencl:navatala_linalg_map_reduce_sum_f32",
  "abi-r1:opencl:navatala_linalg_map_reduce_sum_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_map_reduce_sum_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_map_reduce_sum_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_map_reduce_sum_f64 = {
  1,
  "navatala_linalg_map_reduce_sum_f64",
  "opencl",
  "navatala_linalg_map_reduce_sum_f64",
  "kernel:opencl:navatala_linalg_map_reduce_sum_f64",
  "abi-r1:opencl:navatala_linalg_map_reduce_sum_f64",
  "abi-r1:opencl:navatala_linalg_map_reduce_sum_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_map_reduce_sum_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_outer_product_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_outer_product_f32 = {
  1,
  "navatala_linalg_outer_product_f32",
  "opencl",
  "navatala_linalg_outer_product_f32",
  "kernel:opencl:navatala_linalg_outer_product_f32",
  "abi-r1:opencl:navatala_linalg_outer_product_f32",
  "abi-r1:opencl:navatala_linalg_outer_product_f32",
  5,
  kAbiArgs_opencl_navatala_linalg_outer_product_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_outer_product_f64[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_outer_product_f64 = {
  1,
  "navatala_linalg_outer_product_f64",
  "opencl",
  "navatala_linalg_outer_product_f64",
  "kernel:opencl:navatala_linalg_outer_product_f64",
  "abi-r1:opencl:navatala_linalg_outer_product_f64",
  "abi-r1:opencl:navatala_linalg_outer_product_f64",
  5,
  kAbiArgs_opencl_navatala_linalg_outer_product_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_batched_dot_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vecLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_batched_dot_f32 = {
  1,
  "navatala_linalg_batched_dot_f32",
  "opencl",
  "navatala_linalg_batched_dot_f32",
  "kernel:opencl:navatala_linalg_batched_dot_f32",
  "abi-r1:opencl:navatala_linalg_batched_dot_f32",
  "abi-r1:opencl:navatala_linalg_batched_dot_f32",
  5,
  kAbiArgs_opencl_navatala_linalg_batched_dot_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_batched_dot_f64[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vecLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_batched_dot_f64 = {
  1,
  "navatala_linalg_batched_dot_f64",
  "opencl",
  "navatala_linalg_batched_dot_f64",
  "kernel:opencl:navatala_linalg_batched_dot_f64",
  "abi-r1:opencl:navatala_linalg_batched_dot_f64",
  "abi-r1:opencl:navatala_linalg_batched_dot_f64",
  5,
  kAbiArgs_opencl_navatala_linalg_batched_dot_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_trace_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_trace_f32 = {
  1,
  "navatala_linalg_trace_f32",
  "opencl",
  "navatala_linalg_trace_f32",
  "kernel:opencl:navatala_linalg_trace_f32",
  "abi-r1:opencl:navatala_linalg_trace_f32",
  "abi-r1:opencl:navatala_linalg_trace_f32",
  3,
  kAbiArgs_opencl_navatala_linalg_trace_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_trace_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_trace_f64 = {
  1,
  "navatala_linalg_trace_f64",
  "opencl",
  "navatala_linalg_trace_f64",
  "kernel:opencl:navatala_linalg_trace_f64",
  "abi-r1:opencl:navatala_linalg_trace_f64",
  "abi-r1:opencl:navatala_linalg_trace_f64",
  3,
  kAbiArgs_opencl_navatala_linalg_trace_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_frobenius_norm_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_frobenius_norm_f32 = {
  1,
  "navatala_linalg_frobenius_norm_f32",
  "opencl",
  "navatala_linalg_frobenius_norm_f32",
  "kernel:opencl:navatala_linalg_frobenius_norm_f32",
  "abi-r1:opencl:navatala_linalg_frobenius_norm_f32",
  "abi-r1:opencl:navatala_linalg_frobenius_norm_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_frobenius_norm_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_frobenius_norm_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_frobenius_norm_f64 = {
  1,
  "navatala_linalg_frobenius_norm_f64",
  "opencl",
  "navatala_linalg_frobenius_norm_f64",
  "kernel:opencl:navatala_linalg_frobenius_norm_f64",
  "abi-r1:opencl:navatala_linalg_frobenius_norm_f64",
  "abi-r1:opencl:navatala_linalg_frobenius_norm_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_frobenius_norm_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_extract_upper_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_extract_upper_f32 = {
  1,
  "navatala_linalg_extract_upper_f32",
  "opencl",
  "navatala_linalg_extract_upper_f32",
  "kernel:opencl:navatala_linalg_extract_upper_f32",
  "abi-r1:opencl:navatala_linalg_extract_upper_f32",
  "abi-r1:opencl:navatala_linalg_extract_upper_f32",
  3,
  kAbiArgs_opencl_navatala_linalg_extract_upper_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_extract_upper_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_extract_upper_f64 = {
  1,
  "navatala_linalg_extract_upper_f64",
  "opencl",
  "navatala_linalg_extract_upper_f64",
  "kernel:opencl:navatala_linalg_extract_upper_f64",
  "abi-r1:opencl:navatala_linalg_extract_upper_f64",
  "abi-r1:opencl:navatala_linalg_extract_upper_f64",
  3,
  kAbiArgs_opencl_navatala_linalg_extract_upper_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_extract_lower_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_extract_lower_f32 = {
  1,
  "navatala_linalg_extract_lower_f32",
  "opencl",
  "navatala_linalg_extract_lower_f32",
  "kernel:opencl:navatala_linalg_extract_lower_f32",
  "abi-r1:opencl:navatala_linalg_extract_lower_f32",
  "abi-r1:opencl:navatala_linalg_extract_lower_f32",
  3,
  kAbiArgs_opencl_navatala_linalg_extract_lower_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_extract_lower_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_extract_lower_f64 = {
  1,
  "navatala_linalg_extract_lower_f64",
  "opencl",
  "navatala_linalg_extract_lower_f64",
  "kernel:opencl:navatala_linalg_extract_lower_f64",
  "abi-r1:opencl:navatala_linalg_extract_lower_f64",
  "abi-r1:opencl:navatala_linalg_extract_lower_f64",
  3,
  kAbiArgs_opencl_navatala_linalg_extract_lower_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_set_upper_f32[] = {
  { "val", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_set_upper_f32 = {
  1,
  "navatala_linalg_set_upper_f32",
  "opencl",
  "navatala_linalg_set_upper_f32",
  "kernel:opencl:navatala_linalg_set_upper_f32",
  "abi-r1:opencl:navatala_linalg_set_upper_f32",
  "abi-r1:opencl:navatala_linalg_set_upper_f32",
  3,
  kAbiArgs_opencl_navatala_linalg_set_upper_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_set_upper_f64[] = {
  { "val", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_set_upper_f64 = {
  1,
  "navatala_linalg_set_upper_f64",
  "opencl",
  "navatala_linalg_set_upper_f64",
  "kernel:opencl:navatala_linalg_set_upper_f64",
  "abi-r1:opencl:navatala_linalg_set_upper_f64",
  "abi-r1:opencl:navatala_linalg_set_upper_f64",
  3,
  kAbiArgs_opencl_navatala_linalg_set_upper_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_set_lower_f32[] = {
  { "val", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_set_lower_f32 = {
  1,
  "navatala_linalg_set_lower_f32",
  "opencl",
  "navatala_linalg_set_lower_f32",
  "kernel:opencl:navatala_linalg_set_lower_f32",
  "abi-r1:opencl:navatala_linalg_set_lower_f32",
  "abi-r1:opencl:navatala_linalg_set_lower_f32",
  3,
  kAbiArgs_opencl_navatala_linalg_set_lower_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_set_lower_f64[] = {
  { "val", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_set_lower_f64 = {
  1,
  "navatala_linalg_set_lower_f64",
  "opencl",
  "navatala_linalg_set_lower_f64",
  "kernel:opencl:navatala_linalg_set_lower_f64",
  "abi-r1:opencl:navatala_linalg_set_lower_f64",
  "abi-r1:opencl:navatala_linalg_set_lower_f64",
  3,
  kAbiArgs_opencl_navatala_linalg_set_lower_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_gather_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_gather_rows_f32 = {
  1,
  "navatala_linalg_gather_rows_f32",
  "opencl",
  "navatala_linalg_gather_rows_f32",
  "kernel:opencl:navatala_linalg_gather_rows_f32",
  "abi-r1:opencl:navatala_linalg_gather_rows_f32",
  "abi-r1:opencl:navatala_linalg_gather_rows_f32",
  5,
  kAbiArgs_opencl_navatala_linalg_gather_rows_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_gather_rows_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_gather_rows_f64 = {
  1,
  "navatala_linalg_gather_rows_f64",
  "opencl",
  "navatala_linalg_gather_rows_f64",
  "kernel:opencl:navatala_linalg_gather_rows_f64",
  "abi-r1:opencl:navatala_linalg_gather_rows_f64",
  "abi-r1:opencl:navatala_linalg_gather_rows_f64",
  5,
  kAbiArgs_opencl_navatala_linalg_gather_rows_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_gather_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kCount", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_gather_cols_f32 = {
  1,
  "navatala_linalg_gather_cols_f32",
  "opencl",
  "navatala_linalg_gather_cols_f32",
  "kernel:opencl:navatala_linalg_gather_cols_f32",
  "abi-r1:opencl:navatala_linalg_gather_cols_f32",
  "abi-r1:opencl:navatala_linalg_gather_cols_f32",
  6,
  kAbiArgs_opencl_navatala_linalg_gather_cols_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_gather_cols_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kCount", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_gather_cols_f64 = {
  1,
  "navatala_linalg_gather_cols_f64",
  "opencl",
  "navatala_linalg_gather_cols_f64",
  "kernel:opencl:navatala_linalg_gather_cols_f64",
  "abi-r1:opencl:navatala_linalg_gather_cols_f64",
  "abi-r1:opencl:navatala_linalg_gather_cols_f64",
  6,
  kAbiArgs_opencl_navatala_linalg_gather_cols_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_gather_elements_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_gather_elements_f32 = {
  1,
  "navatala_linalg_gather_elements_f32",
  "opencl",
  "navatala_linalg_gather_elements_f32",
  "kernel:opencl:navatala_linalg_gather_elements_f32",
  "abi-r1:opencl:navatala_linalg_gather_elements_f32",
  "abi-r1:opencl:navatala_linalg_gather_elements_f32",
  6,
  kAbiArgs_opencl_navatala_linalg_gather_elements_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_gather_elements_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_gather_elements_f64 = {
  1,
  "navatala_linalg_gather_elements_f64",
  "opencl",
  "navatala_linalg_gather_elements_f64",
  "kernel:opencl:navatala_linalg_gather_elements_f64",
  "abi-r1:opencl:navatala_linalg_gather_elements_f64",
  "abi-r1:opencl:navatala_linalg_gather_elements_f64",
  6,
  kAbiArgs_opencl_navatala_linalg_gather_elements_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_gather_batched_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kCount", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_gather_batched_f32 = {
  1,
  "navatala_linalg_gather_batched_f32",
  "opencl",
  "navatala_linalg_gather_batched_f32",
  "kernel:opencl:navatala_linalg_gather_batched_f32",
  "abi-r1:opencl:navatala_linalg_gather_batched_f32",
  "abi-r1:opencl:navatala_linalg_gather_batched_f32",
  7,
  kAbiArgs_opencl_navatala_linalg_gather_batched_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_gather_batched_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kCount", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_gather_batched_f64 = {
  1,
  "navatala_linalg_gather_batched_f64",
  "opencl",
  "navatala_linalg_gather_batched_f64",
  "kernel:opencl:navatala_linalg_gather_batched_f64",
  "abi-r1:opencl:navatala_linalg_gather_batched_f64",
  "abi-r1:opencl:navatala_linalg_gather_batched_f64",
  7,
  kAbiArgs_opencl_navatala_linalg_gather_batched_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_scatter_rows_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_scatter_rows_f32 = {
  1,
  "navatala_linalg_scatter_rows_f32",
  "opencl",
  "navatala_linalg_scatter_rows_f32",
  "kernel:opencl:navatala_linalg_scatter_rows_f32",
  "abi-r1:opencl:navatala_linalg_scatter_rows_f32",
  "abi-r1:opencl:navatala_linalg_scatter_rows_f32",
  5,
  kAbiArgs_opencl_navatala_linalg_scatter_rows_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_scatter_rows_f64[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_scatter_rows_f64 = {
  1,
  "navatala_linalg_scatter_rows_f64",
  "opencl",
  "navatala_linalg_scatter_rows_f64",
  "kernel:opencl:navatala_linalg_scatter_rows_f64",
  "abi-r1:opencl:navatala_linalg_scatter_rows_f64",
  "abi-r1:opencl:navatala_linalg_scatter_rows_f64",
  5,
  kAbiArgs_opencl_navatala_linalg_scatter_rows_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_scatter_cols_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_scatter_cols_f32 = {
  1,
  "navatala_linalg_scatter_cols_f32",
  "opencl",
  "navatala_linalg_scatter_cols_f32",
  "kernel:opencl:navatala_linalg_scatter_cols_f32",
  "abi-r1:opencl:navatala_linalg_scatter_cols_f32",
  "abi-r1:opencl:navatala_linalg_scatter_cols_f32",
  6,
  kAbiArgs_opencl_navatala_linalg_scatter_cols_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_scatter_cols_f64[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_scatter_cols_f64 = {
  1,
  "navatala_linalg_scatter_cols_f64",
  "opencl",
  "navatala_linalg_scatter_cols_f64",
  "kernel:opencl:navatala_linalg_scatter_cols_f64",
  "abi-r1:opencl:navatala_linalg_scatter_cols_f64",
  "abi-r1:opencl:navatala_linalg_scatter_cols_f64",
  6,
  kAbiArgs_opencl_navatala_linalg_scatter_cols_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_scatter_add_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_scatter_add_f32 = {
  1,
  "navatala_linalg_scatter_add_f32",
  "opencl",
  "navatala_linalg_scatter_add_f32",
  "kernel:opencl:navatala_linalg_scatter_add_f32",
  "abi-r1:opencl:navatala_linalg_scatter_add_f32",
  "abi-r1:opencl:navatala_linalg_scatter_add_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_scatter_add_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_scatter_add_f64[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_scatter_add_f64 = {
  1,
  "navatala_linalg_scatter_add_f64",
  "opencl",
  "navatala_linalg_scatter_add_f64",
  "kernel:opencl:navatala_linalg_scatter_add_f64",
  "abi-r1:opencl:navatala_linalg_scatter_add_f64",
  "abi-r1:opencl:navatala_linalg_scatter_add_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_scatter_add_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_scatter_max_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_scatter_max_f32 = {
  1,
  "navatala_linalg_scatter_max_f32",
  "opencl",
  "navatala_linalg_scatter_max_f32",
  "kernel:opencl:navatala_linalg_scatter_max_f32",
  "abi-r1:opencl:navatala_linalg_scatter_max_f32",
  "abi-r1:opencl:navatala_linalg_scatter_max_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_scatter_max_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_scatter_max_f64[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_scatter_max_f64 = {
  1,
  "navatala_linalg_scatter_max_f64",
  "opencl",
  "navatala_linalg_scatter_max_f64",
  "kernel:opencl:navatala_linalg_scatter_max_f64",
  "abi-r1:opencl:navatala_linalg_scatter_max_f64",
  "abi-r1:opencl:navatala_linalg_scatter_max_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_scatter_max_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_slice_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowStart", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowEnd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numCols", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_slice_rows_f32 = {
  1,
  "navatala_linalg_slice_rows_f32",
  "opencl",
  "navatala_linalg_slice_rows_f32",
  "kernel:opencl:navatala_linalg_slice_rows_f32",
  "abi-r1:opencl:navatala_linalg_slice_rows_f32",
  "abi-r1:opencl:navatala_linalg_slice_rows_f32",
  5,
  kAbiArgs_opencl_navatala_linalg_slice_rows_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_slice_rows_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowStart", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowEnd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numCols", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_slice_rows_f64 = {
  1,
  "navatala_linalg_slice_rows_f64",
  "opencl",
  "navatala_linalg_slice_rows_f64",
  "kernel:opencl:navatala_linalg_slice_rows_f64",
  "abi-r1:opencl:navatala_linalg_slice_rows_f64",
  "abi-r1:opencl:navatala_linalg_slice_rows_f64",
  5,
  kAbiArgs_opencl_navatala_linalg_slice_rows_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_slice_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colStart", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colEnd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcCols", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_slice_cols_f32 = {
  1,
  "navatala_linalg_slice_cols_f32",
  "opencl",
  "navatala_linalg_slice_cols_f32",
  "kernel:opencl:navatala_linalg_slice_cols_f32",
  "abi-r1:opencl:navatala_linalg_slice_cols_f32",
  "abi-r1:opencl:navatala_linalg_slice_cols_f32",
  6,
  kAbiArgs_opencl_navatala_linalg_slice_cols_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_slice_cols_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colStart", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colEnd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcCols", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_slice_cols_f64 = {
  1,
  "navatala_linalg_slice_cols_f64",
  "opencl",
  "navatala_linalg_slice_cols_f64",
  "kernel:opencl:navatala_linalg_slice_cols_f64",
  "abi-r1:opencl:navatala_linalg_slice_cols_f64",
  "abi-r1:opencl:navatala_linalg_slice_cols_f64",
  6,
  kAbiArgs_opencl_navatala_linalg_slice_cols_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_slice_block_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowStart", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowEnd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colStart", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colEnd", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcCols", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_slice_block_f32 = {
  1,
  "navatala_linalg_slice_block_f32",
  "opencl",
  "navatala_linalg_slice_block_f32",
  "kernel:opencl:navatala_linalg_slice_block_f32",
  "abi-r1:opencl:navatala_linalg_slice_block_f32",
  "abi-r1:opencl:navatala_linalg_slice_block_f32",
  7,
  kAbiArgs_opencl_navatala_linalg_slice_block_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_slice_block_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowStart", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowEnd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colStart", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colEnd", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcCols", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_slice_block_f64 = {
  1,
  "navatala_linalg_slice_block_f64",
  "opencl",
  "navatala_linalg_slice_block_f64",
  "kernel:opencl:navatala_linalg_slice_block_f64",
  "abi-r1:opencl:navatala_linalg_slice_block_f64",
  "abi-r1:opencl:navatala_linalg_slice_block_f64",
  7,
  kAbiArgs_opencl_navatala_linalg_slice_block_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_strided_slice_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "start", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stride", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_strided_slice_f32 = {
  1,
  "navatala_linalg_strided_slice_f32",
  "opencl",
  "navatala_linalg_strided_slice_f32",
  "kernel:opencl:navatala_linalg_strided_slice_f32",
  "abi-r1:opencl:navatala_linalg_strided_slice_f32",
  "abi-r1:opencl:navatala_linalg_strided_slice_f32",
  5,
  kAbiArgs_opencl_navatala_linalg_strided_slice_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_strided_slice_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "start", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stride", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_strided_slice_f64 = {
  1,
  "navatala_linalg_strided_slice_f64",
  "opencl",
  "navatala_linalg_strided_slice_f64",
  "kernel:opencl:navatala_linalg_strided_slice_f64",
  "abi-r1:opencl:navatala_linalg_strided_slice_f64",
  "abi-r1:opencl:navatala_linalg_strided_slice_f64",
  5,
  kAbiArgs_opencl_navatala_linalg_strided_slice_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_extract_diagonal_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_extract_diagonal_f32 = {
  1,
  "navatala_linalg_extract_diagonal_f32",
  "opencl",
  "navatala_linalg_extract_diagonal_f32",
  "kernel:opencl:navatala_linalg_extract_diagonal_f32",
  "abi-r1:opencl:navatala_linalg_extract_diagonal_f32",
  "abi-r1:opencl:navatala_linalg_extract_diagonal_f32",
  3,
  kAbiArgs_opencl_navatala_linalg_extract_diagonal_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_extract_diagonal_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_extract_diagonal_f64 = {
  1,
  "navatala_linalg_extract_diagonal_f64",
  "opencl",
  "navatala_linalg_extract_diagonal_f64",
  "kernel:opencl:navatala_linalg_extract_diagonal_f64",
  "abi-r1:opencl:navatala_linalg_extract_diagonal_f64",
  "abi-r1:opencl:navatala_linalg_extract_diagonal_f64",
  3,
  kAbiArgs_opencl_navatala_linalg_extract_diagonal_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_extract_diagonal_k_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_extract_diagonal_k_f32 = {
  1,
  "navatala_linalg_extract_diagonal_k_f32",
  "opencl",
  "navatala_linalg_extract_diagonal_k_f32",
  "kernel:opencl:navatala_linalg_extract_diagonal_k_f32",
  "abi-r1:opencl:navatala_linalg_extract_diagonal_k_f32",
  "abi-r1:opencl:navatala_linalg_extract_diagonal_k_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_extract_diagonal_k_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_extract_diagonal_k_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_extract_diagonal_k_f64 = {
  1,
  "navatala_linalg_extract_diagonal_k_f64",
  "opencl",
  "navatala_linalg_extract_diagonal_k_f64",
  "kernel:opencl:navatala_linalg_extract_diagonal_k_f64",
  "abi-r1:opencl:navatala_linalg_extract_diagonal_k_f64",
  "abi-r1:opencl:navatala_linalg_extract_diagonal_k_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_extract_diagonal_k_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_set_diagonal_f32[] = {
  { "val", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_set_diagonal_f32 = {
  1,
  "navatala_linalg_set_diagonal_f32",
  "opencl",
  "navatala_linalg_set_diagonal_f32",
  "kernel:opencl:navatala_linalg_set_diagonal_f32",
  "abi-r1:opencl:navatala_linalg_set_diagonal_f32",
  "abi-r1:opencl:navatala_linalg_set_diagonal_f32",
  3,
  kAbiArgs_opencl_navatala_linalg_set_diagonal_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_set_diagonal_f64[] = {
  { "val", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_set_diagonal_f64 = {
  1,
  "navatala_linalg_set_diagonal_f64",
  "opencl",
  "navatala_linalg_set_diagonal_f64",
  "kernel:opencl:navatala_linalg_set_diagonal_f64",
  "abi-r1:opencl:navatala_linalg_set_diagonal_f64",
  "abi-r1:opencl:navatala_linalg_set_diagonal_f64",
  3,
  kAbiArgs_opencl_navatala_linalg_set_diagonal_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_diag_to_matrix_f32[] = {
  { "v", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_diag_to_matrix_f32 = {
  1,
  "navatala_linalg_diag_to_matrix_f32",
  "opencl",
  "navatala_linalg_diag_to_matrix_f32",
  "kernel:opencl:navatala_linalg_diag_to_matrix_f32",
  "abi-r1:opencl:navatala_linalg_diag_to_matrix_f32",
  "abi-r1:opencl:navatala_linalg_diag_to_matrix_f32",
  3,
  kAbiArgs_opencl_navatala_linalg_diag_to_matrix_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_diag_to_matrix_f64[] = {
  { "v", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_diag_to_matrix_f64 = {
  1,
  "navatala_linalg_diag_to_matrix_f64",
  "opencl",
  "navatala_linalg_diag_to_matrix_f64",
  "kernel:opencl:navatala_linalg_diag_to_matrix_f64",
  "abi-r1:opencl:navatala_linalg_diag_to_matrix_f64",
  "abi-r1:opencl:navatala_linalg_diag_to_matrix_f64",
  3,
  kAbiArgs_opencl_navatala_linalg_diag_to_matrix_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_shift_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_shift_rows_f32 = {
  1,
  "navatala_linalg_shift_rows_f32",
  "opencl",
  "navatala_linalg_shift_rows_f32",
  "kernel:opencl:navatala_linalg_shift_rows_f32",
  "abi-r1:opencl:navatala_linalg_shift_rows_f32",
  "abi-r1:opencl:navatala_linalg_shift_rows_f32",
  5,
  kAbiArgs_opencl_navatala_linalg_shift_rows_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_shift_rows_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_shift_rows_f64 = {
  1,
  "navatala_linalg_shift_rows_f64",
  "opencl",
  "navatala_linalg_shift_rows_f64",
  "kernel:opencl:navatala_linalg_shift_rows_f64",
  "abi-r1:opencl:navatala_linalg_shift_rows_f64",
  "abi-r1:opencl:navatala_linalg_shift_rows_f64",
  5,
  kAbiArgs_opencl_navatala_linalg_shift_rows_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_shift_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_shift_cols_f32 = {
  1,
  "navatala_linalg_shift_cols_f32",
  "opencl",
  "navatala_linalg_shift_cols_f32",
  "kernel:opencl:navatala_linalg_shift_cols_f32",
  "abi-r1:opencl:navatala_linalg_shift_cols_f32",
  "abi-r1:opencl:navatala_linalg_shift_cols_f32",
  5,
  kAbiArgs_opencl_navatala_linalg_shift_cols_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_shift_cols_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_shift_cols_f64 = {
  1,
  "navatala_linalg_shift_cols_f64",
  "opencl",
  "navatala_linalg_shift_cols_f64",
  "kernel:opencl:navatala_linalg_shift_cols_f64",
  "abi-r1:opencl:navatala_linalg_shift_cols_f64",
  "abi-r1:opencl:navatala_linalg_shift_cols_f64",
  5,
  kAbiArgs_opencl_navatala_linalg_shift_cols_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_roll_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_roll_rows_f32 = {
  1,
  "navatala_linalg_roll_rows_f32",
  "opencl",
  "navatala_linalg_roll_rows_f32",
  "kernel:opencl:navatala_linalg_roll_rows_f32",
  "abi-r1:opencl:navatala_linalg_roll_rows_f32",
  "abi-r1:opencl:navatala_linalg_roll_rows_f32",
  5,
  kAbiArgs_opencl_navatala_linalg_roll_rows_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_roll_rows_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_roll_rows_f64 = {
  1,
  "navatala_linalg_roll_rows_f64",
  "opencl",
  "navatala_linalg_roll_rows_f64",
  "kernel:opencl:navatala_linalg_roll_rows_f64",
  "abi-r1:opencl:navatala_linalg_roll_rows_f64",
  "abi-r1:opencl:navatala_linalg_roll_rows_f64",
  5,
  kAbiArgs_opencl_navatala_linalg_roll_rows_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_roll_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_roll_cols_f32 = {
  1,
  "navatala_linalg_roll_cols_f32",
  "opencl",
  "navatala_linalg_roll_cols_f32",
  "kernel:opencl:navatala_linalg_roll_cols_f32",
  "abi-r1:opencl:navatala_linalg_roll_cols_f32",
  "abi-r1:opencl:navatala_linalg_roll_cols_f32",
  5,
  kAbiArgs_opencl_navatala_linalg_roll_cols_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_roll_cols_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_roll_cols_f64 = {
  1,
  "navatala_linalg_roll_cols_f64",
  "opencl",
  "navatala_linalg_roll_cols_f64",
  "kernel:opencl:navatala_linalg_roll_cols_f64",
  "abi-r1:opencl:navatala_linalg_roll_cols_f64",
  "abi-r1:opencl:navatala_linalg_roll_cols_f64",
  5,
  kAbiArgs_opencl_navatala_linalg_roll_cols_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_sort_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_sort_rows_f32 = {
  1,
  "navatala_linalg_sort_rows_f32",
  "opencl",
  "navatala_linalg_sort_rows_f32",
  "kernel:opencl:navatala_linalg_sort_rows_f32",
  "abi-r1:opencl:navatala_linalg_sort_rows_f32",
  "abi-r1:opencl:navatala_linalg_sort_rows_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_sort_rows_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_sort_rows_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_sort_rows_f64 = {
  1,
  "navatala_linalg_sort_rows_f64",
  "opencl",
  "navatala_linalg_sort_rows_f64",
  "kernel:opencl:navatala_linalg_sort_rows_f64",
  "abi-r1:opencl:navatala_linalg_sort_rows_f64",
  "abi-r1:opencl:navatala_linalg_sort_rows_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_sort_rows_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_sort_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_sort_cols_f32 = {
  1,
  "navatala_linalg_sort_cols_f32",
  "opencl",
  "navatala_linalg_sort_cols_f32",
  "kernel:opencl:navatala_linalg_sort_cols_f32",
  "abi-r1:opencl:navatala_linalg_sort_cols_f32",
  "abi-r1:opencl:navatala_linalg_sort_cols_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_sort_cols_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_sort_cols_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_sort_cols_f64 = {
  1,
  "navatala_linalg_sort_cols_f64",
  "opencl",
  "navatala_linalg_sort_cols_f64",
  "kernel:opencl:navatala_linalg_sort_cols_f64",
  "abi-r1:opencl:navatala_linalg_sort_cols_f64",
  "abi-r1:opencl:navatala_linalg_sort_cols_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_sort_cols_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_argsort_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_argsort_rows_f32 = {
  1,
  "navatala_linalg_argsort_rows_f32",
  "opencl",
  "navatala_linalg_argsort_rows_f32",
  "kernel:opencl:navatala_linalg_argsort_rows_f32",
  "abi-r1:opencl:navatala_linalg_argsort_rows_f32",
  "abi-r1:opencl:navatala_linalg_argsort_rows_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_argsort_rows_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_argsort_rows_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_argsort_rows_f64 = {
  1,
  "navatala_linalg_argsort_rows_f64",
  "opencl",
  "navatala_linalg_argsort_rows_f64",
  "kernel:opencl:navatala_linalg_argsort_rows_f64",
  "abi-r1:opencl:navatala_linalg_argsort_rows_f64",
  "abi-r1:opencl:navatala_linalg_argsort_rows_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_argsort_rows_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_argsort_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_argsort_cols_f32 = {
  1,
  "navatala_linalg_argsort_cols_f32",
  "opencl",
  "navatala_linalg_argsort_cols_f32",
  "kernel:opencl:navatala_linalg_argsort_cols_f32",
  "abi-r1:opencl:navatala_linalg_argsort_cols_f32",
  "abi-r1:opencl:navatala_linalg_argsort_cols_f32",
  4,
  kAbiArgs_opencl_navatala_linalg_argsort_cols_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_linalg_argsort_cols_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_linalg_argsort_cols_f64 = {
  1,
  "navatala_linalg_argsort_cols_f64",
  "opencl",
  "navatala_linalg_argsort_cols_f64",
  "kernel:opencl:navatala_linalg_argsort_cols_f64",
  "abi-r1:opencl:navatala_linalg_argsort_cols_f64",
  "abi-r1:opencl:navatala_linalg_argsort_cols_f64",
  4,
  kAbiArgs_opencl_navatala_linalg_argsort_cols_f64
};

bool tryGetKernelAbiManifest_opencl_linalg(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "opencl" && kernelName == "navatala_linalg_norm_linf_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_norm_linf_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_norm_linf_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_norm_linf_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_eig_select_top_k_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_eig_select_top_k_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_eig_select_top_k_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_eig_select_top_k_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_cast_f32_to_f16") {
    out = &kAbiManifest_opencl_navatala_linalg_cast_f32_to_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_cast_f64_to_f16") {
    out = &kAbiManifest_opencl_navatala_linalg_cast_f64_to_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_cast_f16_to_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_cast_f16_to_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_cast_f16_to_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_cast_f16_to_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_sum_f16") {
    out = &kAbiManifest_opencl_navatala_linalg_reduce_sum_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_max_f16") {
    out = &kAbiManifest_opencl_navatala_linalg_reduce_max_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_add_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_add_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_add_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_add_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_sub_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_sub_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_sub_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_sub_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_mul_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_mul_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_mul_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_mul_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_div_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_div_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_div_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_div_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_pow_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_pow_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_pow_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_pow_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_sqrt_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_sqrt_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_sqrt_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_sqrt_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_exp_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_exp_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_exp_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_exp_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_log_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_log_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_log_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_log_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_abs_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_abs_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_abs_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_abs_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_sign_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_sign_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_sign_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_elt_sign_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_transpose_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_transpose_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_transpose_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_transpose_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_transpose_in_place_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_transpose_in_place_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_transpose_in_place_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_transpose_in_place_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_transpose_batched_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_transpose_batched_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_transpose_batched_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_transpose_batched_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_map_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_map_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_map_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_map_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_row_sum_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_reduce_row_sum_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_row_sum_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_reduce_row_sum_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_col_sum_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_reduce_col_sum_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_col_sum_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_reduce_col_sum_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_row_max_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_reduce_row_max_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_row_max_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_reduce_row_max_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_col_max_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_reduce_col_max_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_col_max_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_reduce_col_max_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_map_reduce_sum_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_map_reduce_sum_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_map_reduce_sum_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_map_reduce_sum_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_outer_product_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_outer_product_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_outer_product_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_outer_product_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_batched_dot_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_batched_dot_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_batched_dot_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_batched_dot_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_trace_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_trace_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_trace_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_trace_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_frobenius_norm_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_frobenius_norm_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_frobenius_norm_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_frobenius_norm_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_extract_upper_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_extract_upper_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_extract_upper_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_extract_upper_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_extract_lower_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_extract_lower_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_extract_lower_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_extract_lower_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_set_upper_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_set_upper_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_set_upper_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_set_upper_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_set_lower_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_set_lower_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_set_lower_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_set_lower_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_gather_rows_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_gather_rows_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_gather_rows_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_gather_rows_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_gather_cols_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_gather_cols_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_gather_cols_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_gather_cols_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_gather_elements_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_gather_elements_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_gather_elements_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_gather_elements_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_gather_batched_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_gather_batched_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_gather_batched_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_gather_batched_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_scatter_rows_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_scatter_rows_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_scatter_rows_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_scatter_rows_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_scatter_cols_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_scatter_cols_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_scatter_cols_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_scatter_cols_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_scatter_add_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_scatter_add_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_scatter_add_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_scatter_add_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_scatter_max_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_scatter_max_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_scatter_max_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_scatter_max_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_slice_rows_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_slice_rows_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_slice_rows_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_slice_rows_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_slice_cols_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_slice_cols_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_slice_cols_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_slice_cols_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_slice_block_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_slice_block_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_slice_block_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_slice_block_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_strided_slice_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_strided_slice_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_strided_slice_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_strided_slice_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_extract_diagonal_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_extract_diagonal_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_extract_diagonal_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_extract_diagonal_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_extract_diagonal_k_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_extract_diagonal_k_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_extract_diagonal_k_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_extract_diagonal_k_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_set_diagonal_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_set_diagonal_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_set_diagonal_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_set_diagonal_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_diag_to_matrix_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_diag_to_matrix_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_diag_to_matrix_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_diag_to_matrix_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_shift_rows_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_shift_rows_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_shift_rows_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_shift_rows_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_shift_cols_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_shift_cols_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_shift_cols_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_shift_cols_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_roll_rows_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_roll_rows_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_roll_rows_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_roll_rows_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_roll_cols_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_roll_cols_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_roll_cols_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_roll_cols_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_sort_rows_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_sort_rows_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_sort_rows_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_sort_rows_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_sort_cols_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_sort_cols_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_sort_cols_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_sort_cols_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_argsort_rows_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_argsort_rows_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_argsort_rows_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_argsort_rows_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_argsort_cols_f32") {
    out = &kAbiManifest_opencl_navatala_linalg_argsort_cols_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_argsort_cols_f64") {
    out = &kAbiManifest_opencl_navatala_linalg_argsort_cols_f64;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_opencl_linalg(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "opencl" && kernelName == "navatala_linalg_norm_linf_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_norm_linf_f32";
    std::string_view sv(k_opencl_navatala_linalg_norm_linf_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_norm_linf_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_norm_linf_f64";
    std::string_view sv(k_opencl_navatala_linalg_norm_linf_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_eig_select_top_k_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_eig_select_top_k_f32";
    std::string_view sv(k_opencl_navatala_linalg_eig_select_top_k_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_eig_select_top_k_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_eig_select_top_k_f64";
    std::string_view sv(k_opencl_navatala_linalg_eig_select_top_k_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_cast_f32_to_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_cast_f32_to_f16";
    std::string_view sv(k_opencl_navatala_linalg_cast_f32_to_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_cast_f64_to_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_cast_f64_to_f16";
    std::string_view sv(k_opencl_navatala_linalg_cast_f64_to_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_cast_f16_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_cast_f16_to_f32";
    std::string_view sv(k_opencl_navatala_linalg_cast_f16_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_cast_f16_to_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_cast_f16_to_f64";
    std::string_view sv(k_opencl_navatala_linalg_cast_f16_to_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_sum_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_reduce_sum_f16";
    std::string_view sv(k_opencl_navatala_linalg_reduce_sum_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_max_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_reduce_max_f16";
    std::string_view sv(k_opencl_navatala_linalg_reduce_max_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_add_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_add_f32";
    std::string_view sv(k_opencl_navatala_linalg_elt_add_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_add_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_add_f64";
    std::string_view sv(k_opencl_navatala_linalg_elt_add_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_sub_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_sub_f32";
    std::string_view sv(k_opencl_navatala_linalg_elt_sub_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_sub_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_sub_f64";
    std::string_view sv(k_opencl_navatala_linalg_elt_sub_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_mul_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_mul_f32";
    std::string_view sv(k_opencl_navatala_linalg_elt_mul_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_mul_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_mul_f64";
    std::string_view sv(k_opencl_navatala_linalg_elt_mul_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_div_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_div_f32";
    std::string_view sv(k_opencl_navatala_linalg_elt_div_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_div_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_div_f64";
    std::string_view sv(k_opencl_navatala_linalg_elt_div_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_pow_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_pow_f32";
    std::string_view sv(k_opencl_navatala_linalg_elt_pow_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_pow_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_pow_f64";
    std::string_view sv(k_opencl_navatala_linalg_elt_pow_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_sqrt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_sqrt_f32";
    std::string_view sv(k_opencl_navatala_linalg_elt_sqrt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_sqrt_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_sqrt_f64";
    std::string_view sv(k_opencl_navatala_linalg_elt_sqrt_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_exp_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_exp_f32";
    std::string_view sv(k_opencl_navatala_linalg_elt_exp_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_exp_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_exp_f64";
    std::string_view sv(k_opencl_navatala_linalg_elt_exp_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_log_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_log_f32";
    std::string_view sv(k_opencl_navatala_linalg_elt_log_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_log_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_log_f64";
    std::string_view sv(k_opencl_navatala_linalg_elt_log_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_abs_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_abs_f32";
    std::string_view sv(k_opencl_navatala_linalg_elt_abs_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_abs_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_abs_f64";
    std::string_view sv(k_opencl_navatala_linalg_elt_abs_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_sign_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_sign_f32";
    std::string_view sv(k_opencl_navatala_linalg_elt_sign_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_elt_sign_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_elt_sign_f64";
    std::string_view sv(k_opencl_navatala_linalg_elt_sign_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_transpose_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_transpose_f32";
    std::string_view sv(k_opencl_navatala_linalg_transpose_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_transpose_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_transpose_f64";
    std::string_view sv(k_opencl_navatala_linalg_transpose_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_transpose_in_place_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_transpose_in_place_f32";
    std::string_view sv(k_opencl_navatala_linalg_transpose_in_place_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_transpose_in_place_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_transpose_in_place_f64";
    std::string_view sv(k_opencl_navatala_linalg_transpose_in_place_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_transpose_batched_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_transpose_batched_f32";
    std::string_view sv(k_opencl_navatala_linalg_transpose_batched_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_transpose_batched_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_transpose_batched_f64";
    std::string_view sv(k_opencl_navatala_linalg_transpose_batched_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_map_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_map_f32";
    std::string_view sv(k_opencl_navatala_linalg_map_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_map_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_map_f64";
    std::string_view sv(k_opencl_navatala_linalg_map_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_row_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_reduce_row_sum_f32";
    std::string_view sv(k_opencl_navatala_linalg_reduce_row_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_row_sum_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_reduce_row_sum_f64";
    std::string_view sv(k_opencl_navatala_linalg_reduce_row_sum_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_col_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_reduce_col_sum_f32";
    std::string_view sv(k_opencl_navatala_linalg_reduce_col_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_col_sum_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_reduce_col_sum_f64";
    std::string_view sv(k_opencl_navatala_linalg_reduce_col_sum_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_row_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_reduce_row_max_f32";
    std::string_view sv(k_opencl_navatala_linalg_reduce_row_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_row_max_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_reduce_row_max_f64";
    std::string_view sv(k_opencl_navatala_linalg_reduce_row_max_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_col_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_reduce_col_max_f32";
    std::string_view sv(k_opencl_navatala_linalg_reduce_col_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_reduce_col_max_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_reduce_col_max_f64";
    std::string_view sv(k_opencl_navatala_linalg_reduce_col_max_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_map_reduce_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_map_reduce_sum_f32";
    std::string_view sv(k_opencl_navatala_linalg_map_reduce_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_map_reduce_sum_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_map_reduce_sum_f64";
    std::string_view sv(k_opencl_navatala_linalg_map_reduce_sum_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_outer_product_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_outer_product_f32";
    std::string_view sv(k_opencl_navatala_linalg_outer_product_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_outer_product_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_outer_product_f64";
    std::string_view sv(k_opencl_navatala_linalg_outer_product_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_batched_dot_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_batched_dot_f32";
    std::string_view sv(k_opencl_navatala_linalg_batched_dot_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_batched_dot_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_batched_dot_f64";
    std::string_view sv(k_opencl_navatala_linalg_batched_dot_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_trace_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_trace_f32";
    std::string_view sv(k_opencl_navatala_linalg_trace_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_trace_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_trace_f64";
    std::string_view sv(k_opencl_navatala_linalg_trace_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_frobenius_norm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_frobenius_norm_f32";
    std::string_view sv(k_opencl_navatala_linalg_frobenius_norm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_frobenius_norm_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_frobenius_norm_f64";
    std::string_view sv(k_opencl_navatala_linalg_frobenius_norm_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_extract_upper_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_extract_upper_f32";
    std::string_view sv(k_opencl_navatala_linalg_extract_upper_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_extract_upper_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_extract_upper_f64";
    std::string_view sv(k_opencl_navatala_linalg_extract_upper_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_extract_lower_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_extract_lower_f32";
    std::string_view sv(k_opencl_navatala_linalg_extract_lower_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_extract_lower_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_extract_lower_f64";
    std::string_view sv(k_opencl_navatala_linalg_extract_lower_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_set_upper_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_set_upper_f32";
    std::string_view sv(k_opencl_navatala_linalg_set_upper_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_set_upper_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_set_upper_f64";
    std::string_view sv(k_opencl_navatala_linalg_set_upper_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_set_lower_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_set_lower_f32";
    std::string_view sv(k_opencl_navatala_linalg_set_lower_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_set_lower_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_set_lower_f64";
    std::string_view sv(k_opencl_navatala_linalg_set_lower_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_gather_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_gather_rows_f32";
    std::string_view sv(k_opencl_navatala_linalg_gather_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_gather_rows_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_gather_rows_f64";
    std::string_view sv(k_opencl_navatala_linalg_gather_rows_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_gather_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_gather_cols_f32";
    std::string_view sv(k_opencl_navatala_linalg_gather_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_gather_cols_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_gather_cols_f64";
    std::string_view sv(k_opencl_navatala_linalg_gather_cols_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_gather_elements_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_gather_elements_f32";
    std::string_view sv(k_opencl_navatala_linalg_gather_elements_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_gather_elements_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_gather_elements_f64";
    std::string_view sv(k_opencl_navatala_linalg_gather_elements_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_gather_batched_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_gather_batched_f32";
    std::string_view sv(k_opencl_navatala_linalg_gather_batched_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_gather_batched_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_gather_batched_f64";
    std::string_view sv(k_opencl_navatala_linalg_gather_batched_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_scatter_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_scatter_rows_f32";
    std::string_view sv(k_opencl_navatala_linalg_scatter_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_scatter_rows_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_scatter_rows_f64";
    std::string_view sv(k_opencl_navatala_linalg_scatter_rows_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_scatter_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_scatter_cols_f32";
    std::string_view sv(k_opencl_navatala_linalg_scatter_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_scatter_cols_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_scatter_cols_f64";
    std::string_view sv(k_opencl_navatala_linalg_scatter_cols_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_scatter_add_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_scatter_add_f32";
    std::string_view sv(k_opencl_navatala_linalg_scatter_add_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_scatter_add_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_scatter_add_f64";
    std::string_view sv(k_opencl_navatala_linalg_scatter_add_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_scatter_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_scatter_max_f32";
    std::string_view sv(k_opencl_navatala_linalg_scatter_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_scatter_max_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_scatter_max_f64";
    std::string_view sv(k_opencl_navatala_linalg_scatter_max_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_slice_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_slice_rows_f32";
    std::string_view sv(k_opencl_navatala_linalg_slice_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_slice_rows_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_slice_rows_f64";
    std::string_view sv(k_opencl_navatala_linalg_slice_rows_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_slice_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_slice_cols_f32";
    std::string_view sv(k_opencl_navatala_linalg_slice_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_slice_cols_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_slice_cols_f64";
    std::string_view sv(k_opencl_navatala_linalg_slice_cols_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_slice_block_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_slice_block_f32";
    std::string_view sv(k_opencl_navatala_linalg_slice_block_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_slice_block_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_slice_block_f64";
    std::string_view sv(k_opencl_navatala_linalg_slice_block_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_strided_slice_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_strided_slice_f32";
    std::string_view sv(k_opencl_navatala_linalg_strided_slice_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_strided_slice_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_strided_slice_f64";
    std::string_view sv(k_opencl_navatala_linalg_strided_slice_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_extract_diagonal_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_extract_diagonal_f32";
    std::string_view sv(k_opencl_navatala_linalg_extract_diagonal_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_extract_diagonal_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_extract_diagonal_f64";
    std::string_view sv(k_opencl_navatala_linalg_extract_diagonal_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_extract_diagonal_k_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_extract_diagonal_k_f32";
    std::string_view sv(k_opencl_navatala_linalg_extract_diagonal_k_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_extract_diagonal_k_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_extract_diagonal_k_f64";
    std::string_view sv(k_opencl_navatala_linalg_extract_diagonal_k_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_set_diagonal_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_set_diagonal_f32";
    std::string_view sv(k_opencl_navatala_linalg_set_diagonal_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_set_diagonal_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_set_diagonal_f64";
    std::string_view sv(k_opencl_navatala_linalg_set_diagonal_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_diag_to_matrix_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_diag_to_matrix_f32";
    std::string_view sv(k_opencl_navatala_linalg_diag_to_matrix_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_diag_to_matrix_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_diag_to_matrix_f64";
    std::string_view sv(k_opencl_navatala_linalg_diag_to_matrix_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_shift_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_shift_rows_f32";
    std::string_view sv(k_opencl_navatala_linalg_shift_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_shift_rows_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_shift_rows_f64";
    std::string_view sv(k_opencl_navatala_linalg_shift_rows_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_shift_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_shift_cols_f32";
    std::string_view sv(k_opencl_navatala_linalg_shift_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_shift_cols_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_shift_cols_f64";
    std::string_view sv(k_opencl_navatala_linalg_shift_cols_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_roll_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_roll_rows_f32";
    std::string_view sv(k_opencl_navatala_linalg_roll_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_roll_rows_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_roll_rows_f64";
    std::string_view sv(k_opencl_navatala_linalg_roll_rows_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_roll_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_roll_cols_f32";
    std::string_view sv(k_opencl_navatala_linalg_roll_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_roll_cols_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_roll_cols_f64";
    std::string_view sv(k_opencl_navatala_linalg_roll_cols_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_sort_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_sort_rows_f32";
    std::string_view sv(k_opencl_navatala_linalg_sort_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_sort_rows_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_sort_rows_f64";
    std::string_view sv(k_opencl_navatala_linalg_sort_rows_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_sort_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_sort_cols_f32";
    std::string_view sv(k_opencl_navatala_linalg_sort_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_sort_cols_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_sort_cols_f64";
    std::string_view sv(k_opencl_navatala_linalg_sort_cols_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_argsort_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_argsort_rows_f32";
    std::string_view sv(k_opencl_navatala_linalg_argsort_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_argsort_rows_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_argsort_rows_f64";
    std::string_view sv(k_opencl_navatala_linalg_argsort_rows_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_argsort_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_argsort_cols_f32";
    std::string_view sv(k_opencl_navatala_linalg_argsort_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_linalg_argsort_cols_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_linalg_argsort_cols_f64";
    std::string_view sv(k_opencl_navatala_linalg_argsort_cols_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

