// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `cuda_linalg` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_cuda_navatala_linalg_norm_linf_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_norm_linf_f32(const float* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  float val = abs(_input[gid]);
  if (inBounds) {
    sdata[lid] = val;
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int linfF32RedStride = 128u;
  for (int linfF32RedStep = 0; linfF32RedStep < (int)(8); ++linfF32RedStep) {
    unsigned int linfF32Stride = linfF32RedStride;
    if (lid < linfF32Stride) {
      float other = sdata[(lid + linfF32Stride)];
      float mine = sdata[lid];
      bool mineGtOther = (mine > other);
      float maxVal = ((mineGtOther) ? (mine) : (other));
      sdata[lid] = maxVal;
    }
    unsigned int linfF32StrideToHalve = linfF32RedStride;
    unsigned int linfF32NextStride = (linfF32StrideToHalve >> 1u);
    linfF32RedStride = linfF32NextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    result[0] = sdata[0];
  }
}

)kernel";
const char* k_cuda_navatala_linalg_norm_linf_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_norm_linf_f64(const double* _input, const unsigned int* count, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  __shared__ double sdata[256];
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  double val = abs(_input[gid]);
  if (inBounds) {
    sdata[lid] = val;
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int linfF64RedStride = 128u;
  for (int linfF64RedStep = 0; linfF64RedStep < (int)(8); ++linfF64RedStep) {
    unsigned int linfF64Stride = linfF64RedStride;
    if (lid < linfF64Stride) {
      double other = sdata[(lid + linfF64Stride)];
      double mine = sdata[lid];
      bool mineGtOther = (mine > other);
      double maxVal = ((mineGtOther) ? (mine) : (other));
      sdata[lid] = maxVal;
    }
    unsigned int linfF64StrideToHalve = linfF64RedStride;
    unsigned int linfF64NextStride = (linfF64StrideToHalve >> 1u);
    linfF64RedStride = linfF64NextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    result[0] = sdata[0];
  }
}

)kernel";
const char* k_cuda_navatala_linalg_eig_select_top_k_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_eig_select_top_k_f32(const float* eigenvalues, const float* eigenvectors, const unsigned int* k, const unsigned int* n, float* topKValues, float* topKVectors) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int kVal = k[0];
  unsigned int nVal = n[0];
  unsigned int totalWork = (kVal * nVal);
  bool inBounds = (gid < totalWork);
  if (inBounds) {
    unsigned int col = (gid / nVal);
    unsigned int row = (gid % nVal);
    bool isFirstRow = (row == 0u);
    if (isFirstRow) {
      float eigenval = eigenvalues[col];
      topKValues[col] = eigenval;
    }
    unsigned int srcIdx = ((col * nVal) + row);
    float vecElem = eigenvectors[srcIdx];
    topKVectors[srcIdx] = vecElem;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_eig_select_top_k_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_eig_select_top_k_f64(const double* eigenvalues, const double* eigenvectors, const unsigned int* k, const unsigned int* n, double* topKValues, double* topKVectors) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int kVal = k[0];
  unsigned int nVal = n[0];
  unsigned int totalWork = (kVal * nVal);
  bool inBounds = (gid < totalWork);
  if (inBounds) {
    unsigned int col = (gid / nVal);
    unsigned int row = (gid % nVal);
    bool isFirstRow = (row == 0u);
    if (isFirstRow) {
      double eigenval = eigenvalues[col];
      topKValues[col] = eigenval;
    }
    unsigned int srcIdx = ((col * nVal) + row);
    double vecElem = eigenvectors[srcIdx];
    topKVectors[srcIdx] = vecElem;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_cast_f32_to_f16 = R"kernel(
#include <cuda_runtime.h>
#include <cuda_fp16.h>
extern "C" __global__ void navatala_linalg_cast_f32_to_f16(const float* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    float x = _input[gid];
    __half y = ((__half)(x));
    _output[gid] = y;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_cast_f64_to_f16 = R"kernel(
#include <cuda_runtime.h>
#include <cuda_fp16.h>
extern "C" __global__ void navatala_linalg_cast_f64_to_f16(const double* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    double x = _input[gid];
    float xf32 = ((float)(x));
    __half y = ((__half)(xf32));
    _output[gid] = y;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_cast_f16_to_f32 = R"kernel(
#include <cuda_runtime.h>
#include <cuda_fp16.h>
extern "C" __global__ void navatala_linalg_cast_f16_to_f32(const __half* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    __half x = _input[gid];
    float y = ((float)(x));
    _output[gid] = y;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_cast_f16_to_f64 = R"kernel(
#include <cuda_runtime.h>
#include <cuda_fp16.h>
extern "C" __global__ void navatala_linalg_cast_f16_to_f64(const __half* _input, const unsigned int* count, double* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    __half x = _input[gid];
    float xf32 = ((float)(x));
    double y = ((double)(xf32));
    _output[gid] = y;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_cast_f32_to_bf16 = R"kernel(
#include <cuda_runtime.h>
#include <cuda_bf16.h>
extern "C" __global__ void navatala_linalg_cast_f32_to_bf16(const float* _input, const unsigned int* count, __nv_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    float x = _input[gid];
    __nv_bfloat16 y = ((__nv_bfloat16)(x));
    _output[gid] = y;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_cast_f64_to_bf16 = R"kernel(
#include <cuda_runtime.h>
#include <cuda_bf16.h>
extern "C" __global__ void navatala_linalg_cast_f64_to_bf16(const double* _input, const unsigned int* count, __nv_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    double x = _input[gid];
    float xf32 = ((float)(x));
    __nv_bfloat16 y = ((__nv_bfloat16)(xf32));
    _output[gid] = y;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_cast_b_f16_to_f32 = R"kernel(
#include <cuda_runtime.h>
#include <cuda_bf16.h>
extern "C" __global__ void navatala_linalg_cast_b_f16_to_f32(const __nv_bfloat16* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    __nv_bfloat16 x = _input[gid];
    float y = ((float)(x));
    _output[gid] = y;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_cast_b_f16_to_f64 = R"kernel(
#include <cuda_runtime.h>
#include <cuda_bf16.h>
extern "C" __global__ void navatala_linalg_cast_b_f16_to_f64(const __nv_bfloat16* _input, const unsigned int* count, double* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    __nv_bfloat16 x = _input[gid];
    float xf32 = ((float)(x));
    double y = ((double)(xf32));
    _output[gid] = y;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_reduce_sum_f16 = R"kernel(
#include <cuda_runtime.h>
#include <cuda_fp16.h>
extern "C" __global__ void navatala_linalg_reduce_sum_f16(const __half* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    __half xf16 = _input[gid];
    float xf32 = ((float)(xf16));
    sdata[lid] = xf32;
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int sumF16ReductionStride = 128u;
  for (int sumF16ReductionStep = 0; sumF16ReductionStep < (int)(8); ++sumF16ReductionStep) {
    unsigned int sumF16Stride = sumF16ReductionStride;
    if (lid < sumF16Stride) {
      float other = sdata[(lid + sumF16Stride)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    unsigned int sumF16StrideToHalve = sumF16ReductionStride;
    unsigned int sumF16NextStride = (sumF16StrideToHalve >> 1u);
    sumF16ReductionStride = sumF16NextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float partialSum = sdata[0];
    result[0] = partialSum;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_reduce_sum_bf16 = R"kernel(
#include <cuda_runtime.h>
#include <cuda_bf16.h>
extern "C" __global__ void navatala_linalg_reduce_sum_bf16(const __nv_bfloat16* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    __nv_bfloat16 xbf16 = _input[gid];
    float xf32 = ((float)(xbf16));
    sdata[lid] = xf32;
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int sumBF16ReductionStride = 128u;
  for (int sumBF16ReductionStep = 0; sumBF16ReductionStep < (int)(8); ++sumBF16ReductionStep) {
    unsigned int sumBF16Stride = sumBF16ReductionStride;
    if (lid < sumBF16Stride) {
      float other = sdata[(lid + sumBF16Stride)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    unsigned int sumBF16StrideToHalve = sumBF16ReductionStride;
    unsigned int sumBF16NextStride = (sumBF16StrideToHalve >> 1u);
    sumBF16ReductionStride = sumBF16NextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float partialSum = sdata[0];
    result[0] = partialSum;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_reduce_max_f16 = R"kernel(
#include <cuda_runtime.h>
#include <cuda_fp16.h>
extern "C" __global__ void navatala_linalg_reduce_max_f16(const __half* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    __half xf16 = _input[gid];
    float xf32 = ((float)(xf16));
    sdata[lid] = xf32;
  } else {
    sdata[lid] = __uint_as_float(0xff7ffffdu);
  }
  __syncthreads();
  unsigned int maxF16ReductionStride = 128u;
  for (int maxF16ReductionStep = 0; maxF16ReductionStep < (int)(8); ++maxF16ReductionStep) {
    unsigned int maxF16Stride = maxF16ReductionStride;
    if (lid < maxF16Stride) {
      float other = sdata[(lid + maxF16Stride)];
      float mine = sdata[lid];
      bool isGreater = (mine > other);
      float maxVal = ((isGreater) ? (mine) : (other));
      sdata[lid] = maxVal;
    }
    unsigned int maxF16StrideToHalve = maxF16ReductionStride;
    unsigned int maxF16NextStride = (maxF16StrideToHalve >> 1u);
    maxF16ReductionStride = maxF16NextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float maxResult = sdata[0];
    result[0] = maxResult;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_reduce_max_bf16 = R"kernel(
#include <cuda_runtime.h>
#include <cuda_bf16.h>
extern "C" __global__ void navatala_linalg_reduce_max_bf16(const __nv_bfloat16* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    __nv_bfloat16 xbf16 = _input[gid];
    float xf32 = ((float)(xbf16));
    sdata[lid] = xf32;
  } else {
    sdata[lid] = __uint_as_float(0xff7ffffdu);
  }
  __syncthreads();
  unsigned int maxBF16ReductionStride = 128u;
  for (int maxBF16ReductionStep = 0; maxBF16ReductionStep < (int)(8); ++maxBF16ReductionStep) {
    unsigned int maxBF16Stride = maxBF16ReductionStride;
    if (lid < maxBF16Stride) {
      float other = sdata[(lid + maxBF16Stride)];
      float mine = sdata[lid];
      bool isGreater = (mine > other);
      float maxVal = ((isGreater) ? (mine) : (other));
      sdata[lid] = maxVal;
    }
    unsigned int maxBF16StrideToHalve = maxBF16ReductionStride;
    unsigned int maxBF16NextStride = (maxBF16StrideToHalve >> 1u);
    maxBF16ReductionStride = maxBF16NextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float maxResult = sdata[0];
    result[0] = maxResult;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_add_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_add_f32(const float* A, const float* B, const unsigned int* n, float* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    float a = A[i];
    float b = B[i];
    C[i] = (a + b);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_add_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_add_f64(const double* A, const double* B, const unsigned int* n, double* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    double a = A[i];
    double b = B[i];
    C[i] = (a + b);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_sub_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_sub_f32(const float* A, const float* B, const unsigned int* n, float* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    float a = A[i];
    float b = B[i];
    C[i] = (a - b);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_sub_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_sub_f64(const double* A, const double* B, const unsigned int* n, double* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    double a = A[i];
    double b = B[i];
    C[i] = (a - b);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_mul_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_mul_f32(const float* A, const float* B, const unsigned int* n, float* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    float a = A[i];
    float b = B[i];
    C[i] = (a * b);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_mul_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_mul_f64(const double* A, const double* B, const unsigned int* n, double* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    double a = A[i];
    double b = B[i];
    C[i] = (a * b);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_div_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_div_f32(const float* A, const float* B, const unsigned int* n, float* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    float a = A[i];
    float b = B[i];
    C[i] = (a / b);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_div_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_div_f64(const double* A, const double* B, const unsigned int* n, double* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    double a = A[i];
    double b = B[i];
    C[i] = (a / b);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_pow_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_pow_f32(const float* A, const float* p, const unsigned int* n, float* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  float pVal = p[0u];
  if (i < n[0u]) {
    float a = A[i];
    float logA = log(a);
    float pLogA = (pVal * logA);
    C[i] = exp(pLogA);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_pow_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_pow_f64(const double* A, const double* p, const unsigned int* n, double* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  double pVal = p[0u];
  if (i < n[0u]) {
    double a = A[i];
    double logA = log(a);
    double pLogA = (pVal * logA);
    C[i] = exp(pLogA);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_sqrt_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_sqrt_f32(const float* A, const unsigned int* n, float* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    float a = A[i];
    C[i] = sqrt(a);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_sqrt_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_sqrt_f64(const double* A, const unsigned int* n, double* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    double a = A[i];
    C[i] = sqrt(a);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_exp_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_exp_f32(const float* A, const unsigned int* n, float* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    float a = A[i];
    C[i] = exp(a);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_exp_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_exp_f64(const double* A, const unsigned int* n, double* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    double a = A[i];
    C[i] = exp(a);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_log_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_log_f32(const float* A, const unsigned int* n, float* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    float a = A[i];
    C[i] = log(a);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_log_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_log_f64(const double* A, const unsigned int* n, double* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    double a = A[i];
    C[i] = log(a);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_abs_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_abs_f32(const float* A, const unsigned int* n, float* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    float a = A[i];
    C[i] = abs(a);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_abs_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_abs_f64(const double* A, const unsigned int* n, double* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    double a = A[i];
    C[i] = abs(a);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_sign_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_sign_f32(const float* A, const unsigned int* n, float* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    float a = A[i];
    bool isNeg = (a < __uint_as_float(0x00000000u));
    float negOrZero = ((isNeg) ? (__uint_as_float(0xbf800000u)) : (__uint_as_float(0x00000000u)));
    bool isPos = (a > __uint_as_float(0x00000000u));
    float result = ((isPos) ? (__uint_as_float(0x3f800000u)) : (negOrZero));
    C[i] = result;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_elt_sign_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_elt_sign_f64(const double* A, const unsigned int* n, double* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int i = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < n[0u]) {
    double a = A[i];
    bool isNeg = (a < __longlong_as_double(0x0000000000000000ull));
    double negOrZero = ((isNeg) ? (__longlong_as_double(0xbff0000000000000ull)) : (__longlong_as_double(0x0000000000000000ull)));
    bool isPos = (a > __longlong_as_double(0x0000000000000000ull));
    double result = ((isPos) ? (__longlong_as_double(0x3ff0000000000000ull)) : (negOrZero));
    C[i] = result;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_transpose_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_transpose_f32(const float* A, const unsigned int* m, const unsigned int* n, float* B) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int rows = m[0];
  unsigned int cols = n[0];
  unsigned int totalElems = (rows * cols);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int i = (gid / cols);
    unsigned int j = (gid % cols);
    unsigned int srcIdx = ((i * cols) + j);
    unsigned int dstIdx = ((j * rows) + i);
    float val = A[srcIdx];
    B[dstIdx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_transpose_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_transpose_f64(const double* A, const unsigned int* m, const unsigned int* n, double* B) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int rows = m[0];
  unsigned int cols = n[0];
  unsigned int totalElems = (rows * cols);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int i = (gid / cols);
    unsigned int j = (gid % cols);
    unsigned int srcIdx = ((i * cols) + j);
    unsigned int dstIdx = ((j * rows) + i);
    double val = A[srcIdx];
    B[dstIdx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_transpose_in_place_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_transpose_in_place_f32(const unsigned int* n, float* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int dim = n[0];
  unsigned int numSwaps = ((dim * (dim - 1u)) / 2u);
  bool inBounds = (gid < numSwaps);
  if (inBounds) {
    float gidF = ((float)(gid));
    float dimF = ((float)(dim));
    float twoNm1 = ((__uint_as_float(0x40000000u) * dimF) - __uint_as_float(0x3f800000u));
    float disc = ((twoNm1 * twoNm1) - (__uint_as_float(0x41000000u) * gidF));
    float sqrtDisc = sqrt(disc);
    float iF = floor(((twoNm1 - sqrtDisc) / __uint_as_float(0x40000000u)));
    unsigned int i = ((unsigned int)(iF));
    unsigned int rowStart = ((i * (((2u * dim) - i) - 1u)) / 2u);
    unsigned int offset = (gid - rowStart);
    unsigned int j = ((i + 1u) + offset);
    unsigned int idx_ij = ((i * dim) + j);
    unsigned int idx_ji = ((j * dim) + i);
    float val_ij = A[idx_ij];
    float val_ji = A[idx_ji];
    A[idx_ij] = val_ji;
    A[idx_ji] = val_ij;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_transpose_in_place_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_transpose_in_place_f64(const unsigned int* n, double* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int dim = n[0];
  unsigned int numSwaps = ((dim * (dim - 1u)) / 2u);
  bool inBounds = (gid < numSwaps);
  if (inBounds) {
    double gidF = ((double)(gid));
    double dimF = ((double)(dim));
    double twoNm1 = ((__longlong_as_double(0x4000000000000000ull) * dimF) - __longlong_as_double(0x3ff0000000000000ull));
    double disc = ((twoNm1 * twoNm1) - (__longlong_as_double(0x4020000000000000ull) * gidF));
    double sqrtDisc = sqrt(disc);
    double iF = floor(((twoNm1 - sqrtDisc) / __longlong_as_double(0x4000000000000000ull)));
    unsigned int i = ((unsigned int)(iF));
    unsigned int rowStart = ((i * (((2u * dim) - i) - 1u)) / 2u);
    unsigned int offset = (gid - rowStart);
    unsigned int j = ((i + 1u) + offset);
    unsigned int idx_ij = ((i * dim) + j);
    unsigned int idx_ji = ((j * dim) + i);
    double val_ij = A[idx_ij];
    double val_ji = A[idx_ji];
    A[idx_ij] = val_ji;
    A[idx_ji] = val_ij;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_transpose_batched_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_transpose_batched_f32(const float* A, const unsigned int* m, const unsigned int* n, const unsigned int* batchSize, float* B) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int rows = m[0];
  unsigned int cols = n[0];
  unsigned int numBatches = batchSize[0];
  unsigned int matSize = (rows * cols);
  unsigned int totalElems = (matSize * numBatches);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int batchIdx = (gid / matSize);
    unsigned int localIdx = (gid % matSize);
    unsigned int i = (localIdx / cols);
    unsigned int j = (localIdx % cols);
    unsigned int srcIdx = ((batchIdx * matSize) + ((i * cols) + j));
    unsigned int dstIdx = ((batchIdx * matSize) + ((j * rows) + i));
    float val = A[srcIdx];
    B[dstIdx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_transpose_batched_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_transpose_batched_f64(const double* A, const unsigned int* m, const unsigned int* n, const unsigned int* batchSize, double* B) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int rows = m[0];
  unsigned int cols = n[0];
  unsigned int numBatches = batchSize[0];
  unsigned int matSize = (rows * cols);
  unsigned int totalElems = (matSize * numBatches);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int batchIdx = (gid / matSize);
    unsigned int localIdx = (gid % matSize);
    unsigned int i = (localIdx / cols);
    unsigned int j = (localIdx % cols);
    unsigned int srcIdx = ((batchIdx * matSize) + ((i * cols) + j));
    unsigned int dstIdx = ((batchIdx * matSize) + ((j * rows) + i));
    double val = A[srcIdx];
    B[dstIdx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_map_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_map_f32(const float* A, const unsigned int* m, const unsigned int* n, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  if (idx < (m[0u] * n[0u])) {
    float val = A[idx];
    result[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_map_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_map_f64(const double* A, const unsigned int* m, const unsigned int* n, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  if (idx < (m[0u] * n[0u])) {
    double val = A[idx];
    result[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_reduce_row_sum_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_reduce_row_sum_f32(const float* A, const unsigned int* m, const unsigned int* n, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int row = gid;
  if (row < m[0u]) {
    float sumAccum = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)(n[0u]); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int idx = ((row * n[0u]) + jU32);
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
const char* k_cuda_navatala_linalg_reduce_row_sum_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_reduce_row_sum_f64(const double* A, const unsigned int* m, const unsigned int* n, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int row = gid;
  if (row < m[0u]) {
    double sumAccum = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)(n[0u]); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int idx = ((row * n[0u]) + jU32);
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
const char* k_cuda_navatala_linalg_reduce_col_sum_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_reduce_col_sum_f32(const float* A, const unsigned int* m, const unsigned int* n, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int col = gid;
  if (col < n[0u]) {
    float sumAccum = __uint_as_float(0x00000000u);
    for (int i = 0; i < (int)(m[0u]); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int idx = ((iU32 * n[0u]) + col);
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
const char* k_cuda_navatala_linalg_reduce_col_sum_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_reduce_col_sum_f64(const double* A, const unsigned int* m, const unsigned int* n, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int col = gid;
  if (col < n[0u]) {
    double sumAccum = __longlong_as_double(0x0000000000000000ull);
    for (int i = 0; i < (int)(m[0u]); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int idx = ((iU32 * n[0u]) + col);
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
const char* k_cuda_navatala_linalg_reduce_row_max_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_reduce_row_max_f32(const float* A, const unsigned int* m, const unsigned int* n, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int row = gid;
  if (row < m[0u]) {
    unsigned int firstIdx = (row * n[0u]);
    float initMaxVal = A[firstIdx];
    float maxAccum = initMaxVal;
    for (int j = 0; j < (int)(n[0u]); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int idx = ((row * n[0u]) + jU32);
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
const char* k_cuda_navatala_linalg_reduce_row_max_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_reduce_row_max_f64(const double* A, const unsigned int* m, const unsigned int* n, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int row = gid;
  if (row < m[0u]) {
    unsigned int firstIdx = (row * n[0u]);
    double initMaxVal = A[firstIdx];
    double maxAccum = initMaxVal;
    for (int j = 0; j < (int)(n[0u]); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int idx = ((row * n[0u]) + jU32);
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
const char* k_cuda_navatala_linalg_reduce_col_max_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_reduce_col_max_f32(const float* A, const unsigned int* m, const unsigned int* n, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int col = gid;
  if (col < n[0u]) {
    float initMaxVal = A[col];
    float maxAccum = initMaxVal;
    for (int i = 0; i < (int)(m[0u]); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int idx = ((iU32 * n[0u]) + col);
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
const char* k_cuda_navatala_linalg_reduce_col_max_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_reduce_col_max_f64(const double* A, const unsigned int* m, const unsigned int* n, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int col = gid;
  if (col < n[0u]) {
    double initMaxVal = A[col];
    double maxAccum = initMaxVal;
    for (int i = 0; i < (int)(m[0u]); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int idx = ((iU32 * n[0u]) + col);
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
const char* k_cuda_navatala_linalg_map_reduce_sum_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_map_reduce_sum_f32(const float* A, const unsigned int* m, const unsigned int* n, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  if (idx < (m[0u] * n[0u])) {
    float val = A[idx];
    atomicAdd(&result[0u], val);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_map_reduce_sum_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_map_reduce_sum_f64(const double* A, const unsigned int* m, const unsigned int* n, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  if (idx < (m[0u] * n[0u])) {
    double val = A[idx];
    atomicAdd(&result[0u], val);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_outer_product_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_outer_product_f32(const float* a, const float* b, const unsigned int* m, const unsigned int* n, float* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int rows = m[0u];
  unsigned int cols = n[0u];
  unsigned int totalElems = (rows * cols);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int i = (gid / cols);
    unsigned int j = (gid % cols);
    float aVal = a[i];
    float bVal = b[j];
    unsigned int outIdx = ((i * cols) + j);
    float product = (aVal * bVal);
    C[outIdx] = product;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_outer_product_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_outer_product_f64(const double* a, const double* b, const unsigned int* m, const unsigned int* n, double* C) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int rows = m[0u];
  unsigned int cols = n[0u];
  unsigned int totalElems = (rows * cols);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int i = (gid / cols);
    unsigned int j = (gid % cols);
    double aVal = a[i];
    double bVal = b[j];
    unsigned int outIdx = ((i * cols) + j);
    double product = (aVal * bVal);
    C[outIdx] = product;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_batched_dot_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_batched_dot_f32(const float* a, const float* b, const unsigned int* vecLen, const unsigned int* batchSize, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int len = vecLen[0u];
  unsigned int numBatches = batchSize[0u];
  unsigned int k = gid;
  bool inBounds = (k < numBatches);
  if (inBounds) {
    unsigned int baseOffset = (k * len);
    float sumAccum = __uint_as_float(0x00000000u);
    for (int i = 0; i < (int)(len); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int idx = (baseOffset + iU32);
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
const char* k_cuda_navatala_linalg_batched_dot_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_batched_dot_f64(const double* a, const double* b, const unsigned int* vecLen, const unsigned int* batchSize, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int len = vecLen[0u];
  unsigned int numBatches = batchSize[0u];
  unsigned int k = gid;
  bool inBounds = (k < numBatches);
  if (inBounds) {
    unsigned int baseOffset = (k * len);
    double sumAccum = __longlong_as_double(0x0000000000000000ull);
    for (int i = 0; i < (int)(len); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int idx = (baseOffset + iU32);
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
const char* k_cuda_navatala_linalg_trace_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_trace_f32(const float* A, const unsigned int* n, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int dim = n[0u];
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    float traceAccum = __uint_as_float(0x00000000u);
    for (int i = 0; i < (int)(dim); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int diagIdx = ((iU32 * dim) + iU32);
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
const char* k_cuda_navatala_linalg_trace_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_trace_f64(const double* A, const unsigned int* n, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int dim = n[0u];
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    double traceAccum = __longlong_as_double(0x0000000000000000ull);
    for (int i = 0; i < (int)(dim); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int diagIdx = ((iU32 * dim) + iU32);
      double diagVal = A[diagIdx];
      double currentTrace = traceAccum;
      double nextTrace = (currentTrace + diagVal);
      traceAccum = nextTrace;
    }
    double finalTrace = traceAccum;
    result[0u] = finalTrace;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_frobenius_norm_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_frobenius_norm_f32(const float* A, const unsigned int* m, const unsigned int* n, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int rows = m[0u];
  unsigned int cols = n[0u];
  unsigned int totalElems = (rows * cols);
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    float sumSqAccum = __uint_as_float(0x00000000u);
    for (int idx = 0; idx < (int)(totalElems); ++idx) {
      unsigned int idxU32 = ((unsigned int)(idx));
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
const char* k_cuda_navatala_linalg_frobenius_norm_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_frobenius_norm_f64(const double* A, const unsigned int* m, const unsigned int* n, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int rows = m[0u];
  unsigned int cols = n[0u];
  unsigned int totalElems = (rows * cols);
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    double sumSqAccum = __longlong_as_double(0x0000000000000000ull);
    for (int idx = 0; idx < (int)(totalElems); ++idx) {
      unsigned int idxU32 = ((unsigned int)(idx));
      double val = A[idxU32];
      double valSq = (val * val);
      double currentSumSq = sumSqAccum;
      double nextSumSq = (currentSumSq + valSq);
      sumSqAccum = nextSumSq;
    }
    double finalSumSq = sumSqAccum;
    double norm = sqrt(finalSumSq);
    result[0u] = norm;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_extract_upper_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_extract_upper_f32(const float* A, const unsigned int* n, float* B) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int idx = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0u];
  unsigned int totalElements = (nVal * nVal);
  if (idx < totalElements) {
    unsigned int i = (idx / nVal);
    unsigned int j = (idx % nVal);
    float aVal = A[idx];
    bool isUpper = (i <= j);
    float result = ((isUpper) ? (aVal) : (__uint_as_float(0x00000000u)));
    B[idx] = result;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_extract_upper_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_extract_upper_f64(const double* A, const unsigned int* n, double* B) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int idx = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0u];
  unsigned int totalElements = (nVal * nVal);
  if (idx < totalElements) {
    unsigned int i = (idx / nVal);
    unsigned int j = (idx % nVal);
    double aVal = A[idx];
    bool isUpper = (i <= j);
    double result = ((isUpper) ? (aVal) : (__longlong_as_double(0x0000000000000000ull)));
    B[idx] = result;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_extract_lower_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_extract_lower_f32(const float* A, const unsigned int* n, float* B) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int idx = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0u];
  unsigned int totalElements = (nVal * nVal);
  if (idx < totalElements) {
    unsigned int i = (idx / nVal);
    unsigned int j = (idx % nVal);
    float aVal = A[idx];
    bool isLower = (i >= j);
    float result = ((isLower) ? (aVal) : (__uint_as_float(0x00000000u)));
    B[idx] = result;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_extract_lower_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_extract_lower_f64(const double* A, const unsigned int* n, double* B) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int idx = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0u];
  unsigned int totalElements = (nVal * nVal);
  if (idx < totalElements) {
    unsigned int i = (idx / nVal);
    unsigned int j = (idx % nVal);
    double aVal = A[idx];
    bool isLower = (i >= j);
    double result = ((isLower) ? (aVal) : (__longlong_as_double(0x0000000000000000ull)));
    B[idx] = result;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_set_upper_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_set_upper_f32(const float* val, const unsigned int* n, float* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int idx = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0u];
  float fillVal = val[0u];
  unsigned int totalElements = (nVal * nVal);
  if (idx < totalElements) {
    unsigned int i = (idx / nVal);
    unsigned int j = (idx % nVal);
    bool isStrictUpper = (i < j);
    if (isStrictUpper) {
      A[idx] = fillVal;
    }
  }
}

)kernel";
const char* k_cuda_navatala_linalg_set_upper_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_set_upper_f64(const double* val, const unsigned int* n, double* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int idx = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0u];
  double fillVal = val[0u];
  unsigned int totalElements = (nVal * nVal);
  if (idx < totalElements) {
    unsigned int i = (idx / nVal);
    unsigned int j = (idx % nVal);
    bool isStrictUpper = (i < j);
    if (isStrictUpper) {
      A[idx] = fillVal;
    }
  }
}

)kernel";
const char* k_cuda_navatala_linalg_set_lower_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_set_lower_f32(const float* val, const unsigned int* n, float* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int idx = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0u];
  float fillVal = val[0u];
  unsigned int totalElements = (nVal * nVal);
  if (idx < totalElements) {
    unsigned int i = (idx / nVal);
    unsigned int j = (idx % nVal);
    bool isStrictLower = (i > j);
    if (isStrictLower) {
      A[idx] = fillVal;
    }
  }
}

)kernel";
const char* k_cuda_navatala_linalg_set_lower_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_set_lower_f64(const double* val, const unsigned int* n, double* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int idx = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0u];
  double fillVal = val[0u];
  unsigned int totalElements = (nVal * nVal);
  if (idx < totalElements) {
    unsigned int i = (idx / nVal);
    unsigned int j = (idx % nVal);
    bool isStrictLower = (i > j);
    if (isStrictLower) {
      A[idx] = fillVal;
    }
  }
}

)kernel";
const char* k_cuda_navatala_linalg_gather_rows_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_gather_rows_f32(const float* A, const unsigned int* indices, const unsigned int* k, const unsigned int* n, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int kVal = k[0u];
  unsigned int nVal = n[0u];
  unsigned int totalElems = (kVal * nVal);
  if (tid < totalElems) {
    unsigned int rowIdx = (tid / nVal);
    unsigned int colIdx = (tid % nVal);
    unsigned int srcRow = indices[rowIdx];
    unsigned int srcIdx = ((srcRow * nVal) + colIdx);
    float val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_gather_rows_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_gather_rows_f64(const double* A, const unsigned int* indices, const unsigned int* k, const unsigned int* n, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int kVal = k[0u];
  unsigned int nVal = n[0u];
  unsigned int totalElems = (kVal * nVal);
  if (tid < totalElems) {
    unsigned int rowIdx = (tid / nVal);
    unsigned int colIdx = (tid % nVal);
    unsigned int srcRow = indices[rowIdx];
    unsigned int srcIdx = ((srcRow * nVal) + colIdx);
    double val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_gather_cols_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_gather_cols_f32(const float* A, const unsigned int* indices, const unsigned int* m, const unsigned int* n, const unsigned int* kCount, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  unsigned int kVal = kCount[0u];
  unsigned int totalElems = (mVal * kVal);
  if (tid < totalElems) {
    unsigned int rowIdx = (tid / kVal);
    unsigned int outColIdx = (tid % kVal);
    unsigned int srcCol = indices[outColIdx];
    unsigned int srcIdx = ((rowIdx * nVal) + srcCol);
    float val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_gather_cols_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_gather_cols_f64(const double* A, const unsigned int* indices, const unsigned int* m, const unsigned int* n, const unsigned int* kCount, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  unsigned int kVal = kCount[0u];
  unsigned int totalElems = (mVal * kVal);
  if (tid < totalElems) {
    unsigned int rowIdx = (tid / kVal);
    unsigned int outColIdx = (tid % kVal);
    unsigned int srcCol = indices[outColIdx];
    unsigned int srcIdx = ((rowIdx * nVal) + srcCol);
    double val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_gather_elements_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_gather_elements_f32(const float* A, const unsigned int* rowIdx, const unsigned int* colIdx, const unsigned int* count, const unsigned int* n, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int countVal = count[0u];
  unsigned int nVal = n[0u];
  if (tid < countVal) {
    unsigned int row = rowIdx[tid];
    unsigned int col = colIdx[tid];
    unsigned int srcIdx = ((row * nVal) + col);
    float val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_gather_elements_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_gather_elements_f64(const double* A, const unsigned int* rowIdx, const unsigned int* colIdx, const unsigned int* count, const unsigned int* n, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int countVal = count[0u];
  unsigned int nVal = n[0u];
  if (tid < countVal) {
    unsigned int row = rowIdx[tid];
    unsigned int col = colIdx[tid];
    unsigned int srcIdx = ((row * nVal) + col);
    double val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_gather_batched_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_gather_batched_f32(const float* A, const unsigned int* indices, const unsigned int* batchSize, const unsigned int* m, const unsigned int* n, const unsigned int* kCount, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int batchVal = batchSize[0u];
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  unsigned int kVal = kCount[0u];
  unsigned int outBatchStride = (kVal * nVal);
  unsigned int totalElems = (batchVal * outBatchStride);
  if (tid < totalElems) {
    unsigned int batchIdx = (tid / outBatchStride);
    unsigned int withinBatch = (tid % outBatchStride);
    unsigned int rowIdx = (withinBatch / nVal);
    unsigned int colIdx = (withinBatch % nVal);
    unsigned int indicesIdx = ((batchIdx * kVal) + rowIdx);
    unsigned int srcRow = indices[indicesIdx];
    unsigned int srcBatchStride = (mVal * nVal);
    unsigned int srcIdx = (((batchIdx * srcBatchStride) + (srcRow * nVal)) + colIdx);
    float val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_gather_batched_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_gather_batched_f64(const double* A, const unsigned int* indices, const unsigned int* batchSize, const unsigned int* m, const unsigned int* n, const unsigned int* kCount, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int batchVal = batchSize[0u];
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  unsigned int kVal = kCount[0u];
  unsigned int outBatchStride = (kVal * nVal);
  unsigned int totalElems = (batchVal * outBatchStride);
  if (tid < totalElems) {
    unsigned int batchIdx = (tid / outBatchStride);
    unsigned int withinBatch = (tid % outBatchStride);
    unsigned int rowIdx = (withinBatch / nVal);
    unsigned int colIdx = (withinBatch % nVal);
    unsigned int indicesIdx = ((batchIdx * kVal) + rowIdx);
    unsigned int srcRow = indices[indicesIdx];
    unsigned int srcBatchStride = (mVal * nVal);
    unsigned int srcIdx = (((batchIdx * srcBatchStride) + (srcRow * nVal)) + colIdx);
    double val = A[srcIdx];
    _out[tid] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_scatter_rows_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_scatter_rows_f32(const float* src, const unsigned int* indices, const unsigned int* k, const unsigned int* n, float* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int kVal = k[0u];
  unsigned int nVal = n[0u];
  unsigned int totalElems = (kVal * nVal);
  if (tid < totalElems) {
    unsigned int rowIdx = (tid / nVal);
    unsigned int colIdx = (tid % nVal);
    unsigned int dstRow = indices[rowIdx];
    unsigned int dstIdx = ((dstRow * nVal) + colIdx);
    float val = src[tid];
    A[dstIdx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_scatter_rows_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_scatter_rows_f64(const double* src, const unsigned int* indices, const unsigned int* k, const unsigned int* n, double* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int kVal = k[0u];
  unsigned int nVal = n[0u];
  unsigned int totalElems = (kVal * nVal);
  if (tid < totalElems) {
    unsigned int rowIdx = (tid / nVal);
    unsigned int colIdx = (tid % nVal);
    unsigned int dstRow = indices[rowIdx];
    unsigned int dstIdx = ((dstRow * nVal) + colIdx);
    double val = src[tid];
    A[dstIdx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_scatter_cols_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_scatter_cols_f32(const float* src, const unsigned int* indices, const unsigned int* m, const unsigned int* k, const unsigned int* n, float* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int mVal = m[0u];
  unsigned int kVal = k[0u];
  unsigned int nVal = n[0u];
  unsigned int totalElems = (mVal * kVal);
  if (tid < totalElems) {
    unsigned int row = (tid / kVal);
    unsigned int srcColIdx = (tid % kVal);
    unsigned int dstCol = indices[srcColIdx];
    unsigned int dstIdx = ((row * nVal) + dstCol);
    float val = src[tid];
    A[dstIdx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_scatter_cols_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_scatter_cols_f64(const double* src, const unsigned int* indices, const unsigned int* m, const unsigned int* k, const unsigned int* n, double* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int mVal = m[0u];
  unsigned int kVal = k[0u];
  unsigned int nVal = n[0u];
  unsigned int totalElems = (mVal * kVal);
  if (tid < totalElems) {
    unsigned int row = (tid / kVal);
    unsigned int srcColIdx = (tid % kVal);
    unsigned int dstCol = indices[srcColIdx];
    unsigned int dstIdx = ((row * nVal) + dstCol);
    double val = src[tid];
    A[dstIdx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_scatter_add_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_scatter_add_f32(const float* src, const unsigned int* indices, const unsigned int* count, float* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int n = count[0u];
  if (tid < n) {
    unsigned int dstIdx = indices[tid];
    float val = src[tid];
    atomicAdd(&A[dstIdx], val);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_scatter_add_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_scatter_add_f64(const double* src, const unsigned int* indices, const unsigned int* count, double* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int n = count[0u];
  if (tid < n) {
    unsigned int dstIdx = indices[tid];
    double val = src[tid];
    atomicAdd(&A[dstIdx], val);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_scatter_max_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_scatter_max_f32(const float* src, const unsigned int* indices, const unsigned int* count, float* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int n = count[0u];
  if (tid < n) {
    unsigned int dstIdx = indices[tid];
    float val = src[tid];
    atomicMax(&A[dstIdx], val);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_scatter_max_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_scatter_max_f64(const double* src, const unsigned int* indices, const unsigned int* count, double* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int n = count[0u];
  if (tid < n) {
    unsigned int dstIdx = indices[tid];
    double val = src[tid];
    atomicMax(&A[dstIdx], val);
  }
}

)kernel";
const char* k_cuda_navatala_linalg_slice_rows_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_slice_rows_f32(const float* A, const unsigned int* rowStart, const unsigned int* rowEnd, const unsigned int* numCols, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int rs = rowStart[0u];
  unsigned int re = rowEnd[0u];
  unsigned int n = numCols[0u];
  unsigned int numRows = (re - rs);
  unsigned int totalElems = (numRows * n);
  if (idx < totalElems) {
    unsigned int outRow = (idx / n);
    unsigned int outCol = (idx % n);
    unsigned int srcRow = (rs + outRow);
    unsigned int srcIdx = ((srcRow * n) + outCol);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_slice_rows_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_slice_rows_f64(const double* A, const unsigned int* rowStart, const unsigned int* rowEnd, const unsigned int* numCols, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int rs = rowStart[0u];
  unsigned int re = rowEnd[0u];
  unsigned int n = numCols[0u];
  unsigned int numRows = (re - rs);
  unsigned int totalElems = (numRows * n);
  if (idx < totalElems) {
    unsigned int outRow = (idx / n);
    unsigned int outCol = (idx % n);
    unsigned int srcRow = (rs + outRow);
    unsigned int srcIdx = ((srcRow * n) + outCol);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_slice_cols_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_slice_cols_f32(const float* A, const unsigned int* colStart, const unsigned int* colEnd, const unsigned int* numRows, const unsigned int* srcCols, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int cs = colStart[0u];
  unsigned int ce = colEnd[0u];
  unsigned int m = numRows[0u];
  unsigned int n = srcCols[0u];
  unsigned int outCols = (ce - cs);
  unsigned int totalElems = (m * outCols);
  if (idx < totalElems) {
    unsigned int outRow = (idx / outCols);
    unsigned int outCol = (idx % outCols);
    unsigned int srcCol = (cs + outCol);
    unsigned int srcIdx = ((outRow * n) + srcCol);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_slice_cols_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_slice_cols_f64(const double* A, const unsigned int* colStart, const unsigned int* colEnd, const unsigned int* numRows, const unsigned int* srcCols, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int cs = colStart[0u];
  unsigned int ce = colEnd[0u];
  unsigned int m = numRows[0u];
  unsigned int n = srcCols[0u];
  unsigned int outCols = (ce - cs);
  unsigned int totalElems = (m * outCols);
  if (idx < totalElems) {
    unsigned int outRow = (idx / outCols);
    unsigned int outCol = (idx % outCols);
    unsigned int srcCol = (cs + outCol);
    unsigned int srcIdx = ((outRow * n) + srcCol);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_slice_block_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_slice_block_f32(const float* A, const unsigned int* rowStart, const unsigned int* rowEnd, const unsigned int* colStart, const unsigned int* colEnd, const unsigned int* srcCols, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int rs = rowStart[0u];
  unsigned int re = rowEnd[0u];
  unsigned int cs = colStart[0u];
  unsigned int ce = colEnd[0u];
  unsigned int n = srcCols[0u];
  unsigned int outRows = (re - rs);
  unsigned int outCols = (ce - cs);
  unsigned int totalElems = (outRows * outCols);
  if (idx < totalElems) {
    unsigned int outRow = (idx / outCols);
    unsigned int outCol = (idx % outCols);
    unsigned int srcRow = (rs + outRow);
    unsigned int srcCol = (cs + outCol);
    unsigned int srcIdx = ((srcRow * n) + srcCol);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_slice_block_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_slice_block_f64(const double* A, const unsigned int* rowStart, const unsigned int* rowEnd, const unsigned int* colStart, const unsigned int* colEnd, const unsigned int* srcCols, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int rs = rowStart[0u];
  unsigned int re = rowEnd[0u];
  unsigned int cs = colStart[0u];
  unsigned int ce = colEnd[0u];
  unsigned int n = srcCols[0u];
  unsigned int outRows = (re - rs);
  unsigned int outCols = (ce - cs);
  unsigned int totalElems = (outRows * outCols);
  if (idx < totalElems) {
    unsigned int outRow = (idx / outCols);
    unsigned int outCol = (idx % outCols);
    unsigned int srcRow = (rs + outRow);
    unsigned int srcCol = (cs + outCol);
    unsigned int srcIdx = ((srcRow * n) + srcCol);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_strided_slice_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_strided_slice_f32(const float* A, const unsigned int* start, const unsigned int* stride, const unsigned int* count, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int s = start[0u];
  unsigned int st = stride[0u];
  unsigned int n = count[0u];
  if (idx < n) {
    unsigned int srcIdx = (s + (idx * st));
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_strided_slice_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_strided_slice_f64(const double* A, const unsigned int* start, const unsigned int* stride, const unsigned int* count, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int s = start[0u];
  unsigned int st = stride[0u];
  unsigned int n = count[0u];
  if (idx < n) {
    unsigned int srcIdx = (s + (idx * st));
    double val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_extract_diagonal_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_extract_diagonal_f32(const float* A, const unsigned int* n, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int size = n[0u];
  if (idx < size) {
    unsigned int stride = (size + 1u);
    unsigned int srcIdx = (idx * stride);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_extract_diagonal_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_extract_diagonal_f64(const double* A, const unsigned int* n, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int size = n[0u];
  if (idx < size) {
    unsigned int stride = (size + 1u);
    unsigned int srcIdx = (idx * stride);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_extract_diagonal_k_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_extract_diagonal_k_f32(const float* A, const unsigned int* n, const unsigned int* k, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int size = n[0u];
  unsigned int offset = k[0u];
  unsigned int diagLen = (size - offset);
  if (idx < diagLen) {
    unsigned int row = idx;
    unsigned int col = (idx + offset);
    unsigned int srcIdx = ((row * size) + col);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_extract_diagonal_k_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_extract_diagonal_k_f64(const double* A, const unsigned int* n, const unsigned int* k, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int size = n[0u];
  unsigned int offset = k[0u];
  unsigned int diagLen = (size - offset);
  if (idx < diagLen) {
    unsigned int row = idx;
    unsigned int col = (idx + offset);
    unsigned int srcIdx = ((row * size) + col);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_set_diagonal_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_set_diagonal_f32(const float* val, const unsigned int* n, float* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int size = n[0u];
  if (idx < size) {
    unsigned int stride = (size + 1u);
    unsigned int destIdx = (idx * stride);
    float v = val[idx];
    A[destIdx] = v;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_set_diagonal_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_set_diagonal_f64(const double* val, const unsigned int* n, double* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int size = n[0u];
  if (idx < size) {
    unsigned int stride = (size + 1u);
    unsigned int destIdx = (idx * stride);
    double v = val[idx];
    A[destIdx] = v;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_diag_to_matrix_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_diag_to_matrix_f32(const float* v, const unsigned int* n, float* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int size = n[0u];
  unsigned int totalElems = (size * size);
  if (idx < totalElems) {
    unsigned int row = (idx / size);
    unsigned int col = (idx % size);
    bool isDiag = (row == col);
    float diagVal = v[row];
    float result = ((isDiag) ? (diagVal) : (__uint_as_float(0x00000000u)));
    A[idx] = result;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_diag_to_matrix_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_diag_to_matrix_f64(const double* v, const unsigned int* n, double* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int size = n[0u];
  unsigned int totalElems = (size * size);
  if (idx < totalElems) {
    unsigned int row = (idx / size);
    unsigned int col = (idx % size);
    bool isDiag = (row == col);
    double diagVal = v[row];
    double result = ((isDiag) ? (diagVal) : (__longlong_as_double(0x0000000000000000ull)));
    A[idx] = result;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_shift_rows_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_shift_rows_f32(const float* A, const unsigned int* k, const unsigned int* m, const unsigned int* n, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int kVal = k[0u];
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  unsigned int totalElems = (mVal * nVal);
  if (idx < totalElems) {
    unsigned int row = (idx / nVal);
    unsigned int col = (idx % nVal);
    bool inBounds = (row >= kVal);
    unsigned int srcRow = (row - kVal);
    unsigned int srcIdx = ((srcRow * nVal) + col);
    float srcVal = A[srcIdx];
    float result = ((inBounds) ? (srcVal) : (__uint_as_float(0x00000000u)));
    _out[idx] = result;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_shift_rows_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_shift_rows_f64(const double* A, const unsigned int* k, const unsigned int* m, const unsigned int* n, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int kVal = k[0u];
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  unsigned int totalElems = (mVal * nVal);
  if (idx < totalElems) {
    unsigned int row = (idx / nVal);
    unsigned int col = (idx % nVal);
    bool inBounds = (row >= kVal);
    unsigned int srcRow = (row - kVal);
    unsigned int srcIdx = ((srcRow * nVal) + col);
    double srcVal = A[srcIdx];
    double result = ((inBounds) ? (srcVal) : (__longlong_as_double(0x0000000000000000ull)));
    _out[idx] = result;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_shift_cols_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_shift_cols_f32(const float* A, const unsigned int* k, const unsigned int* m, const unsigned int* n, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int kVal = k[0u];
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  unsigned int totalElems = (mVal * nVal);
  if (idx < totalElems) {
    unsigned int row = (idx / nVal);
    unsigned int col = (idx % nVal);
    bool inBounds = (col >= kVal);
    unsigned int srcCol = (col - kVal);
    unsigned int srcIdx = ((row * nVal) + srcCol);
    float srcVal = A[srcIdx];
    float result = ((inBounds) ? (srcVal) : (__uint_as_float(0x00000000u)));
    _out[idx] = result;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_shift_cols_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_shift_cols_f64(const double* A, const unsigned int* k, const unsigned int* m, const unsigned int* n, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int kVal = k[0u];
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  unsigned int totalElems = (mVal * nVal);
  if (idx < totalElems) {
    unsigned int row = (idx / nVal);
    unsigned int col = (idx % nVal);
    bool inBounds = (col >= kVal);
    unsigned int srcCol = (col - kVal);
    unsigned int srcIdx = ((row * nVal) + srcCol);
    double srcVal = A[srcIdx];
    double result = ((inBounds) ? (srcVal) : (__longlong_as_double(0x0000000000000000ull)));
    _out[idx] = result;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_roll_rows_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_roll_rows_f32(const float* A, const unsigned int* k, const unsigned int* m, const unsigned int* n, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int kVal = k[0u];
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  unsigned int totalElems = (mVal * nVal);
  if (idx < totalElems) {
    unsigned int row = (idx / nVal);
    unsigned int col = (idx % nVal);
    unsigned int kMod = (kVal % mVal);
    unsigned int rowPlusM = (row + mVal);
    unsigned int rowMinusK = (rowPlusM - kMod);
    unsigned int srcRow = (rowMinusK % mVal);
    unsigned int srcIdx = ((srcRow * nVal) + col);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_roll_rows_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_roll_rows_f64(const double* A, const unsigned int* k, const unsigned int* m, const unsigned int* n, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int kVal = k[0u];
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  unsigned int totalElems = (mVal * nVal);
  if (idx < totalElems) {
    unsigned int row = (idx / nVal);
    unsigned int col = (idx % nVal);
    unsigned int kMod = (kVal % mVal);
    unsigned int rowPlusM = (row + mVal);
    unsigned int rowMinusK = (rowPlusM - kMod);
    unsigned int srcRow = (rowMinusK % mVal);
    unsigned int srcIdx = ((srcRow * nVal) + col);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_roll_cols_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_roll_cols_f32(const float* A, const unsigned int* k, const unsigned int* m, const unsigned int* n, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int kVal = k[0u];
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  unsigned int totalElems = (mVal * nVal);
  if (idx < totalElems) {
    unsigned int row = (idx / nVal);
    unsigned int col = (idx % nVal);
    unsigned int kMod = (kVal % nVal);
    unsigned int colPlusN = (col + nVal);
    unsigned int colMinusK = (colPlusN - kMod);
    unsigned int srcCol = (colMinusK % nVal);
    unsigned int srcIdx = ((row * nVal) + srcCol);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_roll_cols_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_roll_cols_f64(const double* A, const unsigned int* k, const unsigned int* m, const unsigned int* n, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int kVal = k[0u];
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  unsigned int totalElems = (mVal * nVal);
  if (idx < totalElems) {
    unsigned int row = (idx / nVal);
    unsigned int col = (idx % nVal);
    unsigned int kMod = (kVal % nVal);
    unsigned int colPlusN = (col + nVal);
    unsigned int colMinusK = (colPlusN - kMod);
    unsigned int srcCol = (colMinusK % nVal);
    unsigned int srcIdx = ((row * nVal) + srcCol);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}

)kernel";
const char* k_cuda_navatala_linalg_sort_rows_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_sort_rows_f32(const float* A, const unsigned int* m, const unsigned int* n, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  __shared__ float svals[256];
  unsigned int loopMyParity = 0u;
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  float loopMyVal = __uint_as_float(0x00000000u);
  unsigned int loopPartnerIdx = 0u;
  float loopPartnerVal = __uint_as_float(0x00000000u);
  unsigned int rowIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int colIdx = ((unsigned int)((int)(threadIdx.x)));
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  if (rowIdx < mVal) {
    bool inBounds = (colIdx < nVal);
    unsigned int linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      float val = A[linearIdx];
      svals[colIdx] = val;
    } else {
      svals[colIdx] = __uint_as_float(0x7f7fffffu);
    }
    __syncthreads();
    for (int phase = 0; phase < (int)(nVal); ++phase) {
      unsigned int phaseU32 = ((unsigned int)(phase));
      loopMyParity = (colIdx % 2u);
      unsigned int myParityVal = loopMyParity;
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
        unsigned int partnerIdxVal = loopPartnerIdx;
        loopPartnerVal = svals[partnerIdxVal];
        float partnerValRead = loopPartnerVal;
        if (myValRead > partnerValRead) {
          svals[colIdx] = partnerValRead;
          svals[partnerIdxVal] = myValRead;
        }
      }
      __syncthreads();
    }
    if (inBounds) {
      float sortedVal = svals[colIdx];
      _out[linearIdx] = sortedVal;
    }
  }
}

)kernel";
const char* k_cuda_navatala_linalg_sort_rows_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_sort_rows_f64(const double* A, const unsigned int* m, const unsigned int* n, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  __shared__ double svals[256];
  unsigned int loopMyParity = 0u;
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  double loopMyVal = __longlong_as_double(0x0000000000000000ull);
  unsigned int loopPartnerIdx = 0u;
  double loopPartnerVal = __longlong_as_double(0x0000000000000000ull);
  unsigned int rowIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int colIdx = ((unsigned int)((int)(threadIdx.x)));
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  if (rowIdx < mVal) {
    bool inBounds = (colIdx < nVal);
    unsigned int linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      double val = A[linearIdx];
      svals[colIdx] = val;
    } else {
      svals[colIdx] = __longlong_as_double(0x7fefffffffffffffull);
    }
    __syncthreads();
    for (int phase = 0; phase < (int)(nVal); ++phase) {
      unsigned int phaseU32 = ((unsigned int)(phase));
      loopMyParity = (colIdx % 2u);
      unsigned int myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % 2u));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((colIdx + 1u) < nVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[colIdx];
        double myValRead = loopMyVal;
        loopPartnerIdx = (colIdx + 1u);
        unsigned int partnerIdxVal = loopPartnerIdx;
        loopPartnerVal = svals[partnerIdxVal];
        double partnerValRead = loopPartnerVal;
        if (myValRead > partnerValRead) {
          svals[colIdx] = partnerValRead;
          svals[partnerIdxVal] = myValRead;
        }
      }
      __syncthreads();
    }
    if (inBounds) {
      double sortedVal = svals[colIdx];
      _out[linearIdx] = sortedVal;
    }
  }
}

)kernel";
const char* k_cuda_navatala_linalg_sort_cols_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_sort_cols_f32(const float* A, const unsigned int* m, const unsigned int* n, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  __shared__ float svals[256];
  unsigned int loopMyParity = 0u;
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  float loopMyVal = __uint_as_float(0x00000000u);
  unsigned int loopPartnerIdx = 0u;
  float loopPartnerVal = __uint_as_float(0x00000000u);
  unsigned int colIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int rowIdx = ((unsigned int)((int)(threadIdx.x)));
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  if (colIdx < nVal) {
    bool inBounds = (rowIdx < mVal);
    unsigned int linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      float val = A[linearIdx];
      svals[rowIdx] = val;
    } else {
      svals[rowIdx] = __uint_as_float(0x7f7fffffu);
    }
    __syncthreads();
    for (int phase = 0; phase < (int)(mVal); ++phase) {
      unsigned int phaseU32 = ((unsigned int)(phase));
      loopMyParity = (rowIdx % 2u);
      unsigned int myParityVal = loopMyParity;
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
        unsigned int partnerIdxVal = loopPartnerIdx;
        loopPartnerVal = svals[partnerIdxVal];
        float partnerValRead = loopPartnerVal;
        if (myValRead > partnerValRead) {
          svals[rowIdx] = partnerValRead;
          svals[partnerIdxVal] = myValRead;
        }
      }
      __syncthreads();
    }
    if (inBounds) {
      float sortedVal = svals[rowIdx];
      _out[linearIdx] = sortedVal;
    }
  }
}

)kernel";
const char* k_cuda_navatala_linalg_sort_cols_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_sort_cols_f64(const double* A, const unsigned int* m, const unsigned int* n, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  __shared__ double svals[256];
  unsigned int loopMyParity = 0u;
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  double loopMyVal = __longlong_as_double(0x0000000000000000ull);
  unsigned int loopPartnerIdx = 0u;
  double loopPartnerVal = __longlong_as_double(0x0000000000000000ull);
  unsigned int colIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int rowIdx = ((unsigned int)((int)(threadIdx.x)));
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  if (colIdx < nVal) {
    bool inBounds = (rowIdx < mVal);
    unsigned int linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      double val = A[linearIdx];
      svals[rowIdx] = val;
    } else {
      svals[rowIdx] = __longlong_as_double(0x7fefffffffffffffull);
    }
    __syncthreads();
    for (int phase = 0; phase < (int)(mVal); ++phase) {
      unsigned int phaseU32 = ((unsigned int)(phase));
      loopMyParity = (rowIdx % 2u);
      unsigned int myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % 2u));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((rowIdx + 1u) < mVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[rowIdx];
        double myValRead = loopMyVal;
        loopPartnerIdx = (rowIdx + 1u);
        unsigned int partnerIdxVal = loopPartnerIdx;
        loopPartnerVal = svals[partnerIdxVal];
        double partnerValRead = loopPartnerVal;
        if (myValRead > partnerValRead) {
          svals[rowIdx] = partnerValRead;
          svals[partnerIdxVal] = myValRead;
        }
      }
      __syncthreads();
    }
    if (inBounds) {
      double sortedVal = svals[rowIdx];
      _out[linearIdx] = sortedVal;
    }
  }
}

)kernel";
const char* k_cuda_navatala_linalg_argsort_rows_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_argsort_rows_f32(const float* A, const unsigned int* m, const unsigned int* n, unsigned int* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  __shared__ float svals[256];
  __shared__ unsigned int sidxs[256];
  unsigned int loopMyParity = 0u;
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  float loopMyVal = __uint_as_float(0x00000000u);
  unsigned int loopMyIdx = 0u;
  unsigned int loopPartnerPos = 0u;
  float loopPartnerVal = __uint_as_float(0x00000000u);
  unsigned int loopPartnerIdx = 0u;
  unsigned int rowIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int colIdx = ((unsigned int)((int)(threadIdx.x)));
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  if (rowIdx < mVal) {
    bool inBounds = (colIdx < nVal);
    unsigned int linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      float val = A[linearIdx];
      svals[colIdx] = val;
      sidxs[colIdx] = colIdx;
    } else {
      svals[colIdx] = __uint_as_float(0x7f7fffffu);
      sidxs[colIdx] = colIdx;
    }
    __syncthreads();
    for (int phase = 0; phase < (int)(nVal); ++phase) {
      unsigned int phaseU32 = ((unsigned int)(phase));
      loopMyParity = (colIdx % 2u);
      unsigned int myParityVal = loopMyParity;
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
        unsigned int myIdxRead = loopMyIdx;
        loopPartnerPos = (colIdx + 1u);
        unsigned int partnerPosVal = loopPartnerPos;
        loopPartnerVal = svals[partnerPosVal];
        float partnerValRead = loopPartnerVal;
        loopPartnerIdx = sidxs[partnerPosVal];
        unsigned int partnerIdxRead = loopPartnerIdx;
        if (myValRead > partnerValRead) {
          svals[colIdx] = partnerValRead;
          sidxs[colIdx] = partnerIdxRead;
          svals[partnerPosVal] = myValRead;
          sidxs[partnerPosVal] = myIdxRead;
        }
      }
      __syncthreads();
    }
    if (inBounds) {
      unsigned int sortedIdx = sidxs[colIdx];
      _out[linearIdx] = sortedIdx;
    }
  }
}

)kernel";
const char* k_cuda_navatala_linalg_argsort_rows_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_argsort_rows_f64(const double* A, const unsigned int* m, const unsigned int* n, unsigned int* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  __shared__ double svals[256];
  __shared__ unsigned int sidxs[256];
  unsigned int loopMyParity = 0u;
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  double loopMyVal = __longlong_as_double(0x0000000000000000ull);
  unsigned int loopMyIdx = 0u;
  unsigned int loopPartnerPos = 0u;
  double loopPartnerVal = __longlong_as_double(0x0000000000000000ull);
  unsigned int loopPartnerIdx = 0u;
  unsigned int rowIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int colIdx = ((unsigned int)((int)(threadIdx.x)));
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  if (rowIdx < mVal) {
    bool inBounds = (colIdx < nVal);
    unsigned int linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      double val = A[linearIdx];
      svals[colIdx] = val;
      sidxs[colIdx] = colIdx;
    } else {
      svals[colIdx] = __longlong_as_double(0x7fefffffffffffffull);
      sidxs[colIdx] = colIdx;
    }
    __syncthreads();
    for (int phase = 0; phase < (int)(nVal); ++phase) {
      unsigned int phaseU32 = ((unsigned int)(phase));
      loopMyParity = (colIdx % 2u);
      unsigned int myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % 2u));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((colIdx + 1u) < nVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[colIdx];
        double myValRead = loopMyVal;
        loopMyIdx = sidxs[colIdx];
        unsigned int myIdxRead = loopMyIdx;
        loopPartnerPos = (colIdx + 1u);
        unsigned int partnerPosVal = loopPartnerPos;
        loopPartnerVal = svals[partnerPosVal];
        double partnerValRead = loopPartnerVal;
        loopPartnerIdx = sidxs[partnerPosVal];
        unsigned int partnerIdxRead = loopPartnerIdx;
        if (myValRead > partnerValRead) {
          svals[colIdx] = partnerValRead;
          sidxs[colIdx] = partnerIdxRead;
          svals[partnerPosVal] = myValRead;
          sidxs[partnerPosVal] = myIdxRead;
        }
      }
      __syncthreads();
    }
    if (inBounds) {
      unsigned int sortedIdx = sidxs[colIdx];
      _out[linearIdx] = sortedIdx;
    }
  }
}

)kernel";
const char* k_cuda_navatala_linalg_argsort_cols_f32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_argsort_cols_f32(const float* A, const unsigned int* m, const unsigned int* n, unsigned int* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  __shared__ float svals[256];
  __shared__ unsigned int sidxs[256];
  unsigned int loopMyParity = 0u;
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  float loopMyVal = __uint_as_float(0x00000000u);
  unsigned int loopMyIdx = 0u;
  unsigned int loopPartnerPos = 0u;
  float loopPartnerVal = __uint_as_float(0x00000000u);
  unsigned int loopPartnerIdx = 0u;
  unsigned int colIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int rowIdx = ((unsigned int)((int)(threadIdx.x)));
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  if (colIdx < nVal) {
    bool inBounds = (rowIdx < mVal);
    unsigned int linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      float val = A[linearIdx];
      svals[rowIdx] = val;
      sidxs[rowIdx] = rowIdx;
    } else {
      svals[rowIdx] = __uint_as_float(0x7f7fffffu);
      sidxs[rowIdx] = rowIdx;
    }
    __syncthreads();
    for (int phase = 0; phase < (int)(mVal); ++phase) {
      unsigned int phaseU32 = ((unsigned int)(phase));
      loopMyParity = (rowIdx % 2u);
      unsigned int myParityVal = loopMyParity;
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
        unsigned int myIdxRead = loopMyIdx;
        loopPartnerPos = (rowIdx + 1u);
        unsigned int partnerPosVal = loopPartnerPos;
        loopPartnerVal = svals[partnerPosVal];
        float partnerValRead = loopPartnerVal;
        loopPartnerIdx = sidxs[partnerPosVal];
        unsigned int partnerIdxRead = loopPartnerIdx;
        if (myValRead > partnerValRead) {
          svals[rowIdx] = partnerValRead;
          sidxs[rowIdx] = partnerIdxRead;
          svals[partnerPosVal] = myValRead;
          sidxs[partnerPosVal] = myIdxRead;
        }
      }
      __syncthreads();
    }
    if (inBounds) {
      unsigned int sortedIdx = sidxs[rowIdx];
      _out[linearIdx] = sortedIdx;
    }
  }
}

)kernel";
const char* k_cuda_navatala_linalg_argsort_cols_f64 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_linalg_argsort_cols_f64(const double* A, const unsigned int* m, const unsigned int* n, unsigned int* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  __shared__ double svals[256];
  __shared__ unsigned int sidxs[256];
  unsigned int loopMyParity = 0u;
  bool loopParityMatch = false;
  bool loopHasPartner = false;
  bool loopShouldCompare = false;
  double loopMyVal = __longlong_as_double(0x0000000000000000ull);
  unsigned int loopMyIdx = 0u;
  unsigned int loopPartnerPos = 0u;
  double loopPartnerVal = __longlong_as_double(0x0000000000000000ull);
  unsigned int loopPartnerIdx = 0u;
  unsigned int colIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int rowIdx = ((unsigned int)((int)(threadIdx.x)));
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  if (colIdx < nVal) {
    bool inBounds = (rowIdx < mVal);
    unsigned int linearIdx = ((rowIdx * nVal) + colIdx);
    if (inBounds) {
      double val = A[linearIdx];
      svals[rowIdx] = val;
      sidxs[rowIdx] = rowIdx;
    } else {
      svals[rowIdx] = __longlong_as_double(0x7fefffffffffffffull);
      sidxs[rowIdx] = rowIdx;
    }
    __syncthreads();
    for (int phase = 0; phase < (int)(mVal); ++phase) {
      unsigned int phaseU32 = ((unsigned int)(phase));
      loopMyParity = (rowIdx % 2u);
      unsigned int myParityVal = loopMyParity;
      loopParityMatch = (myParityVal == (phaseU32 % 2u));
      bool parityMatchVal = loopParityMatch;
      loopHasPartner = ((rowIdx + 1u) < mVal);
      bool hasPartnerVal = loopHasPartner;
      loopShouldCompare = ((inBounds && parityMatchVal) && hasPartnerVal);
      bool shouldCompareVal = loopShouldCompare;
      if (shouldCompareVal) {
        loopMyVal = svals[rowIdx];
        double myValRead = loopMyVal;
        loopMyIdx = sidxs[rowIdx];
        unsigned int myIdxRead = loopMyIdx;
        loopPartnerPos = (rowIdx + 1u);
        unsigned int partnerPosVal = loopPartnerPos;
        loopPartnerVal = svals[partnerPosVal];
        double partnerValRead = loopPartnerVal;
        loopPartnerIdx = sidxs[partnerPosVal];
        unsigned int partnerIdxRead = loopPartnerIdx;
        if (myValRead > partnerValRead) {
          svals[rowIdx] = partnerValRead;
          sidxs[rowIdx] = partnerIdxRead;
          svals[partnerPosVal] = myValRead;
          sidxs[partnerPosVal] = myIdxRead;
        }
      }
      __syncthreads();
    }
    if (inBounds) {
      unsigned int sortedIdx = sidxs[rowIdx];
      _out[linearIdx] = sortedIdx;
    }
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_norm_linf_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_norm_linf_f32 = {
  1,
  "navatala_linalg_norm_linf_f32",
  "cuda",
  "navatala_linalg_norm_linf_f32",
  "kernel:cuda:navatala_linalg_norm_linf_f32",
  "abi-r1:cuda:navatala_linalg_norm_linf_f32",
  "abi-r1:cuda:navatala_linalg_norm_linf_f32",
  3,
  kAbiArgs_cuda_navatala_linalg_norm_linf_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_norm_linf_f64[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_norm_linf_f64 = {
  1,
  "navatala_linalg_norm_linf_f64",
  "cuda",
  "navatala_linalg_norm_linf_f64",
  "kernel:cuda:navatala_linalg_norm_linf_f64",
  "abi-r1:cuda:navatala_linalg_norm_linf_f64",
  "abi-r1:cuda:navatala_linalg_norm_linf_f64",
  3,
  kAbiArgs_cuda_navatala_linalg_norm_linf_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_eig_select_top_k_f32[] = {
  { "eigenvalues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "eigenvectors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "topKValues", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "topKVectors", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_eig_select_top_k_f32 = {
  1,
  "navatala_linalg_eig_select_top_k_f32",
  "cuda",
  "navatala_linalg_eig_select_top_k_f32",
  "kernel:cuda:navatala_linalg_eig_select_top_k_f32",
  "abi-r1:cuda:navatala_linalg_eig_select_top_k_f32",
  "abi-r1:cuda:navatala_linalg_eig_select_top_k_f32",
  6,
  kAbiArgs_cuda_navatala_linalg_eig_select_top_k_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_eig_select_top_k_f64[] = {
  { "eigenvalues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "eigenvectors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "topKValues", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "topKVectors", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_eig_select_top_k_f64 = {
  1,
  "navatala_linalg_eig_select_top_k_f64",
  "cuda",
  "navatala_linalg_eig_select_top_k_f64",
  "kernel:cuda:navatala_linalg_eig_select_top_k_f64",
  "abi-r1:cuda:navatala_linalg_eig_select_top_k_f64",
  "abi-r1:cuda:navatala_linalg_eig_select_top_k_f64",
  6,
  kAbiArgs_cuda_navatala_linalg_eig_select_top_k_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_cast_f32_to_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_cast_f32_to_f16 = {
  1,
  "navatala_linalg_cast_f32_to_f16",
  "cuda",
  "navatala_linalg_cast_f32_to_f16",
  "kernel:cuda:navatala_linalg_cast_f32_to_f16",
  "abi-r1:cuda:navatala_linalg_cast_f32_to_f16",
  "abi-r1:cuda:navatala_linalg_cast_f32_to_f16",
  3,
  kAbiArgs_cuda_navatala_linalg_cast_f32_to_f16
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_cast_f64_to_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_cast_f64_to_f16 = {
  1,
  "navatala_linalg_cast_f64_to_f16",
  "cuda",
  "navatala_linalg_cast_f64_to_f16",
  "kernel:cuda:navatala_linalg_cast_f64_to_f16",
  "abi-r1:cuda:navatala_linalg_cast_f64_to_f16",
  "abi-r1:cuda:navatala_linalg_cast_f64_to_f16",
  3,
  kAbiArgs_cuda_navatala_linalg_cast_f64_to_f16
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_cast_f16_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_cast_f16_to_f32 = {
  1,
  "navatala_linalg_cast_f16_to_f32",
  "cuda",
  "navatala_linalg_cast_f16_to_f32",
  "kernel:cuda:navatala_linalg_cast_f16_to_f32",
  "abi-r1:cuda:navatala_linalg_cast_f16_to_f32",
  "abi-r1:cuda:navatala_linalg_cast_f16_to_f32",
  3,
  kAbiArgs_cuda_navatala_linalg_cast_f16_to_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_cast_f16_to_f64[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_cast_f16_to_f64 = {
  1,
  "navatala_linalg_cast_f16_to_f64",
  "cuda",
  "navatala_linalg_cast_f16_to_f64",
  "kernel:cuda:navatala_linalg_cast_f16_to_f64",
  "abi-r1:cuda:navatala_linalg_cast_f16_to_f64",
  "abi-r1:cuda:navatala_linalg_cast_f16_to_f64",
  3,
  kAbiArgs_cuda_navatala_linalg_cast_f16_to_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_cast_f32_to_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_cast_f32_to_bf16 = {
  1,
  "navatala_linalg_cast_f32_to_bf16",
  "cuda",
  "navatala_linalg_cast_f32_to_bf16",
  "kernel:cuda:navatala_linalg_cast_f32_to_bf16",
  "abi-r1:cuda:navatala_linalg_cast_f32_to_bf16",
  "abi-r1:cuda:navatala_linalg_cast_f32_to_bf16",
  3,
  kAbiArgs_cuda_navatala_linalg_cast_f32_to_bf16
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_cast_f64_to_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_cast_f64_to_bf16 = {
  1,
  "navatala_linalg_cast_f64_to_bf16",
  "cuda",
  "navatala_linalg_cast_f64_to_bf16",
  "kernel:cuda:navatala_linalg_cast_f64_to_bf16",
  "abi-r1:cuda:navatala_linalg_cast_f64_to_bf16",
  "abi-r1:cuda:navatala_linalg_cast_f64_to_bf16",
  3,
  kAbiArgs_cuda_navatala_linalg_cast_f64_to_bf16
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_cast_b_f16_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_cast_b_f16_to_f32 = {
  1,
  "navatala_linalg_cast_b_f16_to_f32",
  "cuda",
  "navatala_linalg_cast_b_f16_to_f32",
  "kernel:cuda:navatala_linalg_cast_b_f16_to_f32",
  "abi-r1:cuda:navatala_linalg_cast_b_f16_to_f32",
  "abi-r1:cuda:navatala_linalg_cast_b_f16_to_f32",
  3,
  kAbiArgs_cuda_navatala_linalg_cast_b_f16_to_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_cast_b_f16_to_f64[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_cast_b_f16_to_f64 = {
  1,
  "navatala_linalg_cast_b_f16_to_f64",
  "cuda",
  "navatala_linalg_cast_b_f16_to_f64",
  "kernel:cuda:navatala_linalg_cast_b_f16_to_f64",
  "abi-r1:cuda:navatala_linalg_cast_b_f16_to_f64",
  "abi-r1:cuda:navatala_linalg_cast_b_f16_to_f64",
  3,
  kAbiArgs_cuda_navatala_linalg_cast_b_f16_to_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_reduce_sum_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_reduce_sum_f16 = {
  1,
  "navatala_linalg_reduce_sum_f16",
  "cuda",
  "navatala_linalg_reduce_sum_f16",
  "kernel:cuda:navatala_linalg_reduce_sum_f16",
  "abi-r1:cuda:navatala_linalg_reduce_sum_f16",
  "abi-r1:cuda:navatala_linalg_reduce_sum_f16",
  3,
  kAbiArgs_cuda_navatala_linalg_reduce_sum_f16
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_reduce_sum_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_reduce_sum_bf16 = {
  1,
  "navatala_linalg_reduce_sum_bf16",
  "cuda",
  "navatala_linalg_reduce_sum_bf16",
  "kernel:cuda:navatala_linalg_reduce_sum_bf16",
  "abi-r1:cuda:navatala_linalg_reduce_sum_bf16",
  "abi-r1:cuda:navatala_linalg_reduce_sum_bf16",
  3,
  kAbiArgs_cuda_navatala_linalg_reduce_sum_bf16
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_reduce_max_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_reduce_max_f16 = {
  1,
  "navatala_linalg_reduce_max_f16",
  "cuda",
  "navatala_linalg_reduce_max_f16",
  "kernel:cuda:navatala_linalg_reduce_max_f16",
  "abi-r1:cuda:navatala_linalg_reduce_max_f16",
  "abi-r1:cuda:navatala_linalg_reduce_max_f16",
  3,
  kAbiArgs_cuda_navatala_linalg_reduce_max_f16
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_reduce_max_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_reduce_max_bf16 = {
  1,
  "navatala_linalg_reduce_max_bf16",
  "cuda",
  "navatala_linalg_reduce_max_bf16",
  "kernel:cuda:navatala_linalg_reduce_max_bf16",
  "abi-r1:cuda:navatala_linalg_reduce_max_bf16",
  "abi-r1:cuda:navatala_linalg_reduce_max_bf16",
  3,
  kAbiArgs_cuda_navatala_linalg_reduce_max_bf16
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_add_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_add_f32 = {
  1,
  "navatala_linalg_elt_add_f32",
  "cuda",
  "navatala_linalg_elt_add_f32",
  "kernel:cuda:navatala_linalg_elt_add_f32",
  "abi-r1:cuda:navatala_linalg_elt_add_f32",
  "abi-r1:cuda:navatala_linalg_elt_add_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_elt_add_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_add_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_add_f64 = {
  1,
  "navatala_linalg_elt_add_f64",
  "cuda",
  "navatala_linalg_elt_add_f64",
  "kernel:cuda:navatala_linalg_elt_add_f64",
  "abi-r1:cuda:navatala_linalg_elt_add_f64",
  "abi-r1:cuda:navatala_linalg_elt_add_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_elt_add_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_sub_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_sub_f32 = {
  1,
  "navatala_linalg_elt_sub_f32",
  "cuda",
  "navatala_linalg_elt_sub_f32",
  "kernel:cuda:navatala_linalg_elt_sub_f32",
  "abi-r1:cuda:navatala_linalg_elt_sub_f32",
  "abi-r1:cuda:navatala_linalg_elt_sub_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_elt_sub_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_sub_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_sub_f64 = {
  1,
  "navatala_linalg_elt_sub_f64",
  "cuda",
  "navatala_linalg_elt_sub_f64",
  "kernel:cuda:navatala_linalg_elt_sub_f64",
  "abi-r1:cuda:navatala_linalg_elt_sub_f64",
  "abi-r1:cuda:navatala_linalg_elt_sub_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_elt_sub_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_mul_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_mul_f32 = {
  1,
  "navatala_linalg_elt_mul_f32",
  "cuda",
  "navatala_linalg_elt_mul_f32",
  "kernel:cuda:navatala_linalg_elt_mul_f32",
  "abi-r1:cuda:navatala_linalg_elt_mul_f32",
  "abi-r1:cuda:navatala_linalg_elt_mul_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_elt_mul_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_mul_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_mul_f64 = {
  1,
  "navatala_linalg_elt_mul_f64",
  "cuda",
  "navatala_linalg_elt_mul_f64",
  "kernel:cuda:navatala_linalg_elt_mul_f64",
  "abi-r1:cuda:navatala_linalg_elt_mul_f64",
  "abi-r1:cuda:navatala_linalg_elt_mul_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_elt_mul_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_div_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_div_f32 = {
  1,
  "navatala_linalg_elt_div_f32",
  "cuda",
  "navatala_linalg_elt_div_f32",
  "kernel:cuda:navatala_linalg_elt_div_f32",
  "abi-r1:cuda:navatala_linalg_elt_div_f32",
  "abi-r1:cuda:navatala_linalg_elt_div_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_elt_div_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_div_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "B", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_div_f64 = {
  1,
  "navatala_linalg_elt_div_f64",
  "cuda",
  "navatala_linalg_elt_div_f64",
  "kernel:cuda:navatala_linalg_elt_div_f64",
  "abi-r1:cuda:navatala_linalg_elt_div_f64",
  "abi-r1:cuda:navatala_linalg_elt_div_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_elt_div_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_pow_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "p", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_pow_f32 = {
  1,
  "navatala_linalg_elt_pow_f32",
  "cuda",
  "navatala_linalg_elt_pow_f32",
  "kernel:cuda:navatala_linalg_elt_pow_f32",
  "abi-r1:cuda:navatala_linalg_elt_pow_f32",
  "abi-r1:cuda:navatala_linalg_elt_pow_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_elt_pow_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_pow_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "p", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_pow_f64 = {
  1,
  "navatala_linalg_elt_pow_f64",
  "cuda",
  "navatala_linalg_elt_pow_f64",
  "kernel:cuda:navatala_linalg_elt_pow_f64",
  "abi-r1:cuda:navatala_linalg_elt_pow_f64",
  "abi-r1:cuda:navatala_linalg_elt_pow_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_elt_pow_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_sqrt_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_sqrt_f32 = {
  1,
  "navatala_linalg_elt_sqrt_f32",
  "cuda",
  "navatala_linalg_elt_sqrt_f32",
  "kernel:cuda:navatala_linalg_elt_sqrt_f32",
  "abi-r1:cuda:navatala_linalg_elt_sqrt_f32",
  "abi-r1:cuda:navatala_linalg_elt_sqrt_f32",
  3,
  kAbiArgs_cuda_navatala_linalg_elt_sqrt_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_sqrt_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_sqrt_f64 = {
  1,
  "navatala_linalg_elt_sqrt_f64",
  "cuda",
  "navatala_linalg_elt_sqrt_f64",
  "kernel:cuda:navatala_linalg_elt_sqrt_f64",
  "abi-r1:cuda:navatala_linalg_elt_sqrt_f64",
  "abi-r1:cuda:navatala_linalg_elt_sqrt_f64",
  3,
  kAbiArgs_cuda_navatala_linalg_elt_sqrt_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_exp_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_exp_f32 = {
  1,
  "navatala_linalg_elt_exp_f32",
  "cuda",
  "navatala_linalg_elt_exp_f32",
  "kernel:cuda:navatala_linalg_elt_exp_f32",
  "abi-r1:cuda:navatala_linalg_elt_exp_f32",
  "abi-r1:cuda:navatala_linalg_elt_exp_f32",
  3,
  kAbiArgs_cuda_navatala_linalg_elt_exp_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_exp_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_exp_f64 = {
  1,
  "navatala_linalg_elt_exp_f64",
  "cuda",
  "navatala_linalg_elt_exp_f64",
  "kernel:cuda:navatala_linalg_elt_exp_f64",
  "abi-r1:cuda:navatala_linalg_elt_exp_f64",
  "abi-r1:cuda:navatala_linalg_elt_exp_f64",
  3,
  kAbiArgs_cuda_navatala_linalg_elt_exp_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_log_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_log_f32 = {
  1,
  "navatala_linalg_elt_log_f32",
  "cuda",
  "navatala_linalg_elt_log_f32",
  "kernel:cuda:navatala_linalg_elt_log_f32",
  "abi-r1:cuda:navatala_linalg_elt_log_f32",
  "abi-r1:cuda:navatala_linalg_elt_log_f32",
  3,
  kAbiArgs_cuda_navatala_linalg_elt_log_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_log_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_log_f64 = {
  1,
  "navatala_linalg_elt_log_f64",
  "cuda",
  "navatala_linalg_elt_log_f64",
  "kernel:cuda:navatala_linalg_elt_log_f64",
  "abi-r1:cuda:navatala_linalg_elt_log_f64",
  "abi-r1:cuda:navatala_linalg_elt_log_f64",
  3,
  kAbiArgs_cuda_navatala_linalg_elt_log_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_abs_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_abs_f32 = {
  1,
  "navatala_linalg_elt_abs_f32",
  "cuda",
  "navatala_linalg_elt_abs_f32",
  "kernel:cuda:navatala_linalg_elt_abs_f32",
  "abi-r1:cuda:navatala_linalg_elt_abs_f32",
  "abi-r1:cuda:navatala_linalg_elt_abs_f32",
  3,
  kAbiArgs_cuda_navatala_linalg_elt_abs_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_abs_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_abs_f64 = {
  1,
  "navatala_linalg_elt_abs_f64",
  "cuda",
  "navatala_linalg_elt_abs_f64",
  "kernel:cuda:navatala_linalg_elt_abs_f64",
  "abi-r1:cuda:navatala_linalg_elt_abs_f64",
  "abi-r1:cuda:navatala_linalg_elt_abs_f64",
  3,
  kAbiArgs_cuda_navatala_linalg_elt_abs_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_sign_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_sign_f32 = {
  1,
  "navatala_linalg_elt_sign_f32",
  "cuda",
  "navatala_linalg_elt_sign_f32",
  "kernel:cuda:navatala_linalg_elt_sign_f32",
  "abi-r1:cuda:navatala_linalg_elt_sign_f32",
  "abi-r1:cuda:navatala_linalg_elt_sign_f32",
  3,
  kAbiArgs_cuda_navatala_linalg_elt_sign_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_elt_sign_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_elt_sign_f64 = {
  1,
  "navatala_linalg_elt_sign_f64",
  "cuda",
  "navatala_linalg_elt_sign_f64",
  "kernel:cuda:navatala_linalg_elt_sign_f64",
  "abi-r1:cuda:navatala_linalg_elt_sign_f64",
  "abi-r1:cuda:navatala_linalg_elt_sign_f64",
  3,
  kAbiArgs_cuda_navatala_linalg_elt_sign_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_transpose_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_transpose_f32 = {
  1,
  "navatala_linalg_transpose_f32",
  "cuda",
  "navatala_linalg_transpose_f32",
  "kernel:cuda:navatala_linalg_transpose_f32",
  "abi-r1:cuda:navatala_linalg_transpose_f32",
  "abi-r1:cuda:navatala_linalg_transpose_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_transpose_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_transpose_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_transpose_f64 = {
  1,
  "navatala_linalg_transpose_f64",
  "cuda",
  "navatala_linalg_transpose_f64",
  "kernel:cuda:navatala_linalg_transpose_f64",
  "abi-r1:cuda:navatala_linalg_transpose_f64",
  "abi-r1:cuda:navatala_linalg_transpose_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_transpose_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_transpose_in_place_f32[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_transpose_in_place_f32 = {
  1,
  "navatala_linalg_transpose_in_place_f32",
  "cuda",
  "navatala_linalg_transpose_in_place_f32",
  "kernel:cuda:navatala_linalg_transpose_in_place_f32",
  "abi-r1:cuda:navatala_linalg_transpose_in_place_f32",
  "abi-r1:cuda:navatala_linalg_transpose_in_place_f32",
  2,
  kAbiArgs_cuda_navatala_linalg_transpose_in_place_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_transpose_in_place_f64[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_transpose_in_place_f64 = {
  1,
  "navatala_linalg_transpose_in_place_f64",
  "cuda",
  "navatala_linalg_transpose_in_place_f64",
  "kernel:cuda:navatala_linalg_transpose_in_place_f64",
  "abi-r1:cuda:navatala_linalg_transpose_in_place_f64",
  "abi-r1:cuda:navatala_linalg_transpose_in_place_f64",
  2,
  kAbiArgs_cuda_navatala_linalg_transpose_in_place_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_transpose_batched_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_transpose_batched_f32 = {
  1,
  "navatala_linalg_transpose_batched_f32",
  "cuda",
  "navatala_linalg_transpose_batched_f32",
  "kernel:cuda:navatala_linalg_transpose_batched_f32",
  "abi-r1:cuda:navatala_linalg_transpose_batched_f32",
  "abi-r1:cuda:navatala_linalg_transpose_batched_f32",
  5,
  kAbiArgs_cuda_navatala_linalg_transpose_batched_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_transpose_batched_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_transpose_batched_f64 = {
  1,
  "navatala_linalg_transpose_batched_f64",
  "cuda",
  "navatala_linalg_transpose_batched_f64",
  "kernel:cuda:navatala_linalg_transpose_batched_f64",
  "abi-r1:cuda:navatala_linalg_transpose_batched_f64",
  "abi-r1:cuda:navatala_linalg_transpose_batched_f64",
  5,
  kAbiArgs_cuda_navatala_linalg_transpose_batched_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_map_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_map_f32 = {
  1,
  "navatala_linalg_map_f32",
  "cuda",
  "navatala_linalg_map_f32",
  "kernel:cuda:navatala_linalg_map_f32",
  "abi-r1:cuda:navatala_linalg_map_f32",
  "abi-r1:cuda:navatala_linalg_map_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_map_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_map_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_map_f64 = {
  1,
  "navatala_linalg_map_f64",
  "cuda",
  "navatala_linalg_map_f64",
  "kernel:cuda:navatala_linalg_map_f64",
  "abi-r1:cuda:navatala_linalg_map_f64",
  "abi-r1:cuda:navatala_linalg_map_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_map_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_reduce_row_sum_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_reduce_row_sum_f32 = {
  1,
  "navatala_linalg_reduce_row_sum_f32",
  "cuda",
  "navatala_linalg_reduce_row_sum_f32",
  "kernel:cuda:navatala_linalg_reduce_row_sum_f32",
  "abi-r1:cuda:navatala_linalg_reduce_row_sum_f32",
  "abi-r1:cuda:navatala_linalg_reduce_row_sum_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_reduce_row_sum_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_reduce_row_sum_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_reduce_row_sum_f64 = {
  1,
  "navatala_linalg_reduce_row_sum_f64",
  "cuda",
  "navatala_linalg_reduce_row_sum_f64",
  "kernel:cuda:navatala_linalg_reduce_row_sum_f64",
  "abi-r1:cuda:navatala_linalg_reduce_row_sum_f64",
  "abi-r1:cuda:navatala_linalg_reduce_row_sum_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_reduce_row_sum_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_reduce_col_sum_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_reduce_col_sum_f32 = {
  1,
  "navatala_linalg_reduce_col_sum_f32",
  "cuda",
  "navatala_linalg_reduce_col_sum_f32",
  "kernel:cuda:navatala_linalg_reduce_col_sum_f32",
  "abi-r1:cuda:navatala_linalg_reduce_col_sum_f32",
  "abi-r1:cuda:navatala_linalg_reduce_col_sum_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_reduce_col_sum_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_reduce_col_sum_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_reduce_col_sum_f64 = {
  1,
  "navatala_linalg_reduce_col_sum_f64",
  "cuda",
  "navatala_linalg_reduce_col_sum_f64",
  "kernel:cuda:navatala_linalg_reduce_col_sum_f64",
  "abi-r1:cuda:navatala_linalg_reduce_col_sum_f64",
  "abi-r1:cuda:navatala_linalg_reduce_col_sum_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_reduce_col_sum_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_reduce_row_max_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_reduce_row_max_f32 = {
  1,
  "navatala_linalg_reduce_row_max_f32",
  "cuda",
  "navatala_linalg_reduce_row_max_f32",
  "kernel:cuda:navatala_linalg_reduce_row_max_f32",
  "abi-r1:cuda:navatala_linalg_reduce_row_max_f32",
  "abi-r1:cuda:navatala_linalg_reduce_row_max_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_reduce_row_max_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_reduce_row_max_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_reduce_row_max_f64 = {
  1,
  "navatala_linalg_reduce_row_max_f64",
  "cuda",
  "navatala_linalg_reduce_row_max_f64",
  "kernel:cuda:navatala_linalg_reduce_row_max_f64",
  "abi-r1:cuda:navatala_linalg_reduce_row_max_f64",
  "abi-r1:cuda:navatala_linalg_reduce_row_max_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_reduce_row_max_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_reduce_col_max_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_reduce_col_max_f32 = {
  1,
  "navatala_linalg_reduce_col_max_f32",
  "cuda",
  "navatala_linalg_reduce_col_max_f32",
  "kernel:cuda:navatala_linalg_reduce_col_max_f32",
  "abi-r1:cuda:navatala_linalg_reduce_col_max_f32",
  "abi-r1:cuda:navatala_linalg_reduce_col_max_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_reduce_col_max_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_reduce_col_max_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_reduce_col_max_f64 = {
  1,
  "navatala_linalg_reduce_col_max_f64",
  "cuda",
  "navatala_linalg_reduce_col_max_f64",
  "kernel:cuda:navatala_linalg_reduce_col_max_f64",
  "abi-r1:cuda:navatala_linalg_reduce_col_max_f64",
  "abi-r1:cuda:navatala_linalg_reduce_col_max_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_reduce_col_max_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_map_reduce_sum_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_map_reduce_sum_f32 = {
  1,
  "navatala_linalg_map_reduce_sum_f32",
  "cuda",
  "navatala_linalg_map_reduce_sum_f32",
  "kernel:cuda:navatala_linalg_map_reduce_sum_f32",
  "abi-r1:cuda:navatala_linalg_map_reduce_sum_f32",
  "abi-r1:cuda:navatala_linalg_map_reduce_sum_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_map_reduce_sum_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_map_reduce_sum_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_map_reduce_sum_f64 = {
  1,
  "navatala_linalg_map_reduce_sum_f64",
  "cuda",
  "navatala_linalg_map_reduce_sum_f64",
  "kernel:cuda:navatala_linalg_map_reduce_sum_f64",
  "abi-r1:cuda:navatala_linalg_map_reduce_sum_f64",
  "abi-r1:cuda:navatala_linalg_map_reduce_sum_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_map_reduce_sum_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_outer_product_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_outer_product_f32 = {
  1,
  "navatala_linalg_outer_product_f32",
  "cuda",
  "navatala_linalg_outer_product_f32",
  "kernel:cuda:navatala_linalg_outer_product_f32",
  "abi-r1:cuda:navatala_linalg_outer_product_f32",
  "abi-r1:cuda:navatala_linalg_outer_product_f32",
  5,
  kAbiArgs_cuda_navatala_linalg_outer_product_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_outer_product_f64[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_outer_product_f64 = {
  1,
  "navatala_linalg_outer_product_f64",
  "cuda",
  "navatala_linalg_outer_product_f64",
  "kernel:cuda:navatala_linalg_outer_product_f64",
  "abi-r1:cuda:navatala_linalg_outer_product_f64",
  "abi-r1:cuda:navatala_linalg_outer_product_f64",
  5,
  kAbiArgs_cuda_navatala_linalg_outer_product_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_batched_dot_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vecLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_batched_dot_f32 = {
  1,
  "navatala_linalg_batched_dot_f32",
  "cuda",
  "navatala_linalg_batched_dot_f32",
  "kernel:cuda:navatala_linalg_batched_dot_f32",
  "abi-r1:cuda:navatala_linalg_batched_dot_f32",
  "abi-r1:cuda:navatala_linalg_batched_dot_f32",
  5,
  kAbiArgs_cuda_navatala_linalg_batched_dot_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_batched_dot_f64[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vecLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_batched_dot_f64 = {
  1,
  "navatala_linalg_batched_dot_f64",
  "cuda",
  "navatala_linalg_batched_dot_f64",
  "kernel:cuda:navatala_linalg_batched_dot_f64",
  "abi-r1:cuda:navatala_linalg_batched_dot_f64",
  "abi-r1:cuda:navatala_linalg_batched_dot_f64",
  5,
  kAbiArgs_cuda_navatala_linalg_batched_dot_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_trace_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_trace_f32 = {
  1,
  "navatala_linalg_trace_f32",
  "cuda",
  "navatala_linalg_trace_f32",
  "kernel:cuda:navatala_linalg_trace_f32",
  "abi-r1:cuda:navatala_linalg_trace_f32",
  "abi-r1:cuda:navatala_linalg_trace_f32",
  3,
  kAbiArgs_cuda_navatala_linalg_trace_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_trace_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_trace_f64 = {
  1,
  "navatala_linalg_trace_f64",
  "cuda",
  "navatala_linalg_trace_f64",
  "kernel:cuda:navatala_linalg_trace_f64",
  "abi-r1:cuda:navatala_linalg_trace_f64",
  "abi-r1:cuda:navatala_linalg_trace_f64",
  3,
  kAbiArgs_cuda_navatala_linalg_trace_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_frobenius_norm_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_frobenius_norm_f32 = {
  1,
  "navatala_linalg_frobenius_norm_f32",
  "cuda",
  "navatala_linalg_frobenius_norm_f32",
  "kernel:cuda:navatala_linalg_frobenius_norm_f32",
  "abi-r1:cuda:navatala_linalg_frobenius_norm_f32",
  "abi-r1:cuda:navatala_linalg_frobenius_norm_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_frobenius_norm_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_frobenius_norm_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_frobenius_norm_f64 = {
  1,
  "navatala_linalg_frobenius_norm_f64",
  "cuda",
  "navatala_linalg_frobenius_norm_f64",
  "kernel:cuda:navatala_linalg_frobenius_norm_f64",
  "abi-r1:cuda:navatala_linalg_frobenius_norm_f64",
  "abi-r1:cuda:navatala_linalg_frobenius_norm_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_frobenius_norm_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_extract_upper_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_extract_upper_f32 = {
  1,
  "navatala_linalg_extract_upper_f32",
  "cuda",
  "navatala_linalg_extract_upper_f32",
  "kernel:cuda:navatala_linalg_extract_upper_f32",
  "abi-r1:cuda:navatala_linalg_extract_upper_f32",
  "abi-r1:cuda:navatala_linalg_extract_upper_f32",
  3,
  kAbiArgs_cuda_navatala_linalg_extract_upper_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_extract_upper_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_extract_upper_f64 = {
  1,
  "navatala_linalg_extract_upper_f64",
  "cuda",
  "navatala_linalg_extract_upper_f64",
  "kernel:cuda:navatala_linalg_extract_upper_f64",
  "abi-r1:cuda:navatala_linalg_extract_upper_f64",
  "abi-r1:cuda:navatala_linalg_extract_upper_f64",
  3,
  kAbiArgs_cuda_navatala_linalg_extract_upper_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_extract_lower_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_extract_lower_f32 = {
  1,
  "navatala_linalg_extract_lower_f32",
  "cuda",
  "navatala_linalg_extract_lower_f32",
  "kernel:cuda:navatala_linalg_extract_lower_f32",
  "abi-r1:cuda:navatala_linalg_extract_lower_f32",
  "abi-r1:cuda:navatala_linalg_extract_lower_f32",
  3,
  kAbiArgs_cuda_navatala_linalg_extract_lower_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_extract_lower_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "B", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_extract_lower_f64 = {
  1,
  "navatala_linalg_extract_lower_f64",
  "cuda",
  "navatala_linalg_extract_lower_f64",
  "kernel:cuda:navatala_linalg_extract_lower_f64",
  "abi-r1:cuda:navatala_linalg_extract_lower_f64",
  "abi-r1:cuda:navatala_linalg_extract_lower_f64",
  3,
  kAbiArgs_cuda_navatala_linalg_extract_lower_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_set_upper_f32[] = {
  { "val", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_set_upper_f32 = {
  1,
  "navatala_linalg_set_upper_f32",
  "cuda",
  "navatala_linalg_set_upper_f32",
  "kernel:cuda:navatala_linalg_set_upper_f32",
  "abi-r1:cuda:navatala_linalg_set_upper_f32",
  "abi-r1:cuda:navatala_linalg_set_upper_f32",
  3,
  kAbiArgs_cuda_navatala_linalg_set_upper_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_set_upper_f64[] = {
  { "val", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_set_upper_f64 = {
  1,
  "navatala_linalg_set_upper_f64",
  "cuda",
  "navatala_linalg_set_upper_f64",
  "kernel:cuda:navatala_linalg_set_upper_f64",
  "abi-r1:cuda:navatala_linalg_set_upper_f64",
  "abi-r1:cuda:navatala_linalg_set_upper_f64",
  3,
  kAbiArgs_cuda_navatala_linalg_set_upper_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_set_lower_f32[] = {
  { "val", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_set_lower_f32 = {
  1,
  "navatala_linalg_set_lower_f32",
  "cuda",
  "navatala_linalg_set_lower_f32",
  "kernel:cuda:navatala_linalg_set_lower_f32",
  "abi-r1:cuda:navatala_linalg_set_lower_f32",
  "abi-r1:cuda:navatala_linalg_set_lower_f32",
  3,
  kAbiArgs_cuda_navatala_linalg_set_lower_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_set_lower_f64[] = {
  { "val", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_set_lower_f64 = {
  1,
  "navatala_linalg_set_lower_f64",
  "cuda",
  "navatala_linalg_set_lower_f64",
  "kernel:cuda:navatala_linalg_set_lower_f64",
  "abi-r1:cuda:navatala_linalg_set_lower_f64",
  "abi-r1:cuda:navatala_linalg_set_lower_f64",
  3,
  kAbiArgs_cuda_navatala_linalg_set_lower_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_gather_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_gather_rows_f32 = {
  1,
  "navatala_linalg_gather_rows_f32",
  "cuda",
  "navatala_linalg_gather_rows_f32",
  "kernel:cuda:navatala_linalg_gather_rows_f32",
  "abi-r1:cuda:navatala_linalg_gather_rows_f32",
  "abi-r1:cuda:navatala_linalg_gather_rows_f32",
  5,
  kAbiArgs_cuda_navatala_linalg_gather_rows_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_gather_rows_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_gather_rows_f64 = {
  1,
  "navatala_linalg_gather_rows_f64",
  "cuda",
  "navatala_linalg_gather_rows_f64",
  "kernel:cuda:navatala_linalg_gather_rows_f64",
  "abi-r1:cuda:navatala_linalg_gather_rows_f64",
  "abi-r1:cuda:navatala_linalg_gather_rows_f64",
  5,
  kAbiArgs_cuda_navatala_linalg_gather_rows_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_gather_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kCount", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_gather_cols_f32 = {
  1,
  "navatala_linalg_gather_cols_f32",
  "cuda",
  "navatala_linalg_gather_cols_f32",
  "kernel:cuda:navatala_linalg_gather_cols_f32",
  "abi-r1:cuda:navatala_linalg_gather_cols_f32",
  "abi-r1:cuda:navatala_linalg_gather_cols_f32",
  6,
  kAbiArgs_cuda_navatala_linalg_gather_cols_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_gather_cols_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kCount", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_gather_cols_f64 = {
  1,
  "navatala_linalg_gather_cols_f64",
  "cuda",
  "navatala_linalg_gather_cols_f64",
  "kernel:cuda:navatala_linalg_gather_cols_f64",
  "abi-r1:cuda:navatala_linalg_gather_cols_f64",
  "abi-r1:cuda:navatala_linalg_gather_cols_f64",
  6,
  kAbiArgs_cuda_navatala_linalg_gather_cols_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_gather_elements_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_gather_elements_f32 = {
  1,
  "navatala_linalg_gather_elements_f32",
  "cuda",
  "navatala_linalg_gather_elements_f32",
  "kernel:cuda:navatala_linalg_gather_elements_f32",
  "abi-r1:cuda:navatala_linalg_gather_elements_f32",
  "abi-r1:cuda:navatala_linalg_gather_elements_f32",
  6,
  kAbiArgs_cuda_navatala_linalg_gather_elements_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_gather_elements_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_gather_elements_f64 = {
  1,
  "navatala_linalg_gather_elements_f64",
  "cuda",
  "navatala_linalg_gather_elements_f64",
  "kernel:cuda:navatala_linalg_gather_elements_f64",
  "abi-r1:cuda:navatala_linalg_gather_elements_f64",
  "abi-r1:cuda:navatala_linalg_gather_elements_f64",
  6,
  kAbiArgs_cuda_navatala_linalg_gather_elements_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_gather_batched_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kCount", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_gather_batched_f32 = {
  1,
  "navatala_linalg_gather_batched_f32",
  "cuda",
  "navatala_linalg_gather_batched_f32",
  "kernel:cuda:navatala_linalg_gather_batched_f32",
  "abi-r1:cuda:navatala_linalg_gather_batched_f32",
  "abi-r1:cuda:navatala_linalg_gather_batched_f32",
  7,
  kAbiArgs_cuda_navatala_linalg_gather_batched_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_gather_batched_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kCount", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_gather_batched_f64 = {
  1,
  "navatala_linalg_gather_batched_f64",
  "cuda",
  "navatala_linalg_gather_batched_f64",
  "kernel:cuda:navatala_linalg_gather_batched_f64",
  "abi-r1:cuda:navatala_linalg_gather_batched_f64",
  "abi-r1:cuda:navatala_linalg_gather_batched_f64",
  7,
  kAbiArgs_cuda_navatala_linalg_gather_batched_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_scatter_rows_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_scatter_rows_f32 = {
  1,
  "navatala_linalg_scatter_rows_f32",
  "cuda",
  "navatala_linalg_scatter_rows_f32",
  "kernel:cuda:navatala_linalg_scatter_rows_f32",
  "abi-r1:cuda:navatala_linalg_scatter_rows_f32",
  "abi-r1:cuda:navatala_linalg_scatter_rows_f32",
  5,
  kAbiArgs_cuda_navatala_linalg_scatter_rows_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_scatter_rows_f64[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_scatter_rows_f64 = {
  1,
  "navatala_linalg_scatter_rows_f64",
  "cuda",
  "navatala_linalg_scatter_rows_f64",
  "kernel:cuda:navatala_linalg_scatter_rows_f64",
  "abi-r1:cuda:navatala_linalg_scatter_rows_f64",
  "abi-r1:cuda:navatala_linalg_scatter_rows_f64",
  5,
  kAbiArgs_cuda_navatala_linalg_scatter_rows_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_scatter_cols_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_scatter_cols_f32 = {
  1,
  "navatala_linalg_scatter_cols_f32",
  "cuda",
  "navatala_linalg_scatter_cols_f32",
  "kernel:cuda:navatala_linalg_scatter_cols_f32",
  "abi-r1:cuda:navatala_linalg_scatter_cols_f32",
  "abi-r1:cuda:navatala_linalg_scatter_cols_f32",
  6,
  kAbiArgs_cuda_navatala_linalg_scatter_cols_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_scatter_cols_f64[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_scatter_cols_f64 = {
  1,
  "navatala_linalg_scatter_cols_f64",
  "cuda",
  "navatala_linalg_scatter_cols_f64",
  "kernel:cuda:navatala_linalg_scatter_cols_f64",
  "abi-r1:cuda:navatala_linalg_scatter_cols_f64",
  "abi-r1:cuda:navatala_linalg_scatter_cols_f64",
  6,
  kAbiArgs_cuda_navatala_linalg_scatter_cols_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_scatter_add_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_scatter_add_f32 = {
  1,
  "navatala_linalg_scatter_add_f32",
  "cuda",
  "navatala_linalg_scatter_add_f32",
  "kernel:cuda:navatala_linalg_scatter_add_f32",
  "abi-r1:cuda:navatala_linalg_scatter_add_f32",
  "abi-r1:cuda:navatala_linalg_scatter_add_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_scatter_add_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_scatter_add_f64[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_scatter_add_f64 = {
  1,
  "navatala_linalg_scatter_add_f64",
  "cuda",
  "navatala_linalg_scatter_add_f64",
  "kernel:cuda:navatala_linalg_scatter_add_f64",
  "abi-r1:cuda:navatala_linalg_scatter_add_f64",
  "abi-r1:cuda:navatala_linalg_scatter_add_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_scatter_add_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_scatter_max_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_scatter_max_f32 = {
  1,
  "navatala_linalg_scatter_max_f32",
  "cuda",
  "navatala_linalg_scatter_max_f32",
  "kernel:cuda:navatala_linalg_scatter_max_f32",
  "abi-r1:cuda:navatala_linalg_scatter_max_f32",
  "abi-r1:cuda:navatala_linalg_scatter_max_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_scatter_max_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_scatter_max_f64[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_scatter_max_f64 = {
  1,
  "navatala_linalg_scatter_max_f64",
  "cuda",
  "navatala_linalg_scatter_max_f64",
  "kernel:cuda:navatala_linalg_scatter_max_f64",
  "abi-r1:cuda:navatala_linalg_scatter_max_f64",
  "abi-r1:cuda:navatala_linalg_scatter_max_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_scatter_max_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_slice_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowStart", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowEnd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numCols", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_slice_rows_f32 = {
  1,
  "navatala_linalg_slice_rows_f32",
  "cuda",
  "navatala_linalg_slice_rows_f32",
  "kernel:cuda:navatala_linalg_slice_rows_f32",
  "abi-r1:cuda:navatala_linalg_slice_rows_f32",
  "abi-r1:cuda:navatala_linalg_slice_rows_f32",
  5,
  kAbiArgs_cuda_navatala_linalg_slice_rows_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_slice_rows_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowStart", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowEnd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numCols", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_slice_rows_f64 = {
  1,
  "navatala_linalg_slice_rows_f64",
  "cuda",
  "navatala_linalg_slice_rows_f64",
  "kernel:cuda:navatala_linalg_slice_rows_f64",
  "abi-r1:cuda:navatala_linalg_slice_rows_f64",
  "abi-r1:cuda:navatala_linalg_slice_rows_f64",
  5,
  kAbiArgs_cuda_navatala_linalg_slice_rows_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_slice_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colStart", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colEnd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcCols", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_slice_cols_f32 = {
  1,
  "navatala_linalg_slice_cols_f32",
  "cuda",
  "navatala_linalg_slice_cols_f32",
  "kernel:cuda:navatala_linalg_slice_cols_f32",
  "abi-r1:cuda:navatala_linalg_slice_cols_f32",
  "abi-r1:cuda:navatala_linalg_slice_cols_f32",
  6,
  kAbiArgs_cuda_navatala_linalg_slice_cols_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_slice_cols_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colStart", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colEnd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcCols", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_slice_cols_f64 = {
  1,
  "navatala_linalg_slice_cols_f64",
  "cuda",
  "navatala_linalg_slice_cols_f64",
  "kernel:cuda:navatala_linalg_slice_cols_f64",
  "abi-r1:cuda:navatala_linalg_slice_cols_f64",
  "abi-r1:cuda:navatala_linalg_slice_cols_f64",
  6,
  kAbiArgs_cuda_navatala_linalg_slice_cols_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_slice_block_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowStart", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowEnd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colStart", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colEnd", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcCols", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_slice_block_f32 = {
  1,
  "navatala_linalg_slice_block_f32",
  "cuda",
  "navatala_linalg_slice_block_f32",
  "kernel:cuda:navatala_linalg_slice_block_f32",
  "abi-r1:cuda:navatala_linalg_slice_block_f32",
  "abi-r1:cuda:navatala_linalg_slice_block_f32",
  7,
  kAbiArgs_cuda_navatala_linalg_slice_block_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_slice_block_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rowStart", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rowEnd", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colStart", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "colEnd", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "srcCols", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_slice_block_f64 = {
  1,
  "navatala_linalg_slice_block_f64",
  "cuda",
  "navatala_linalg_slice_block_f64",
  "kernel:cuda:navatala_linalg_slice_block_f64",
  "abi-r1:cuda:navatala_linalg_slice_block_f64",
  "abi-r1:cuda:navatala_linalg_slice_block_f64",
  7,
  kAbiArgs_cuda_navatala_linalg_slice_block_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_strided_slice_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "start", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stride", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_strided_slice_f32 = {
  1,
  "navatala_linalg_strided_slice_f32",
  "cuda",
  "navatala_linalg_strided_slice_f32",
  "kernel:cuda:navatala_linalg_strided_slice_f32",
  "abi-r1:cuda:navatala_linalg_strided_slice_f32",
  "abi-r1:cuda:navatala_linalg_strided_slice_f32",
  5,
  kAbiArgs_cuda_navatala_linalg_strided_slice_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_strided_slice_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "start", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stride", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_strided_slice_f64 = {
  1,
  "navatala_linalg_strided_slice_f64",
  "cuda",
  "navatala_linalg_strided_slice_f64",
  "kernel:cuda:navatala_linalg_strided_slice_f64",
  "abi-r1:cuda:navatala_linalg_strided_slice_f64",
  "abi-r1:cuda:navatala_linalg_strided_slice_f64",
  5,
  kAbiArgs_cuda_navatala_linalg_strided_slice_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_extract_diagonal_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_extract_diagonal_f32 = {
  1,
  "navatala_linalg_extract_diagonal_f32",
  "cuda",
  "navatala_linalg_extract_diagonal_f32",
  "kernel:cuda:navatala_linalg_extract_diagonal_f32",
  "abi-r1:cuda:navatala_linalg_extract_diagonal_f32",
  "abi-r1:cuda:navatala_linalg_extract_diagonal_f32",
  3,
  kAbiArgs_cuda_navatala_linalg_extract_diagonal_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_extract_diagonal_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_extract_diagonal_f64 = {
  1,
  "navatala_linalg_extract_diagonal_f64",
  "cuda",
  "navatala_linalg_extract_diagonal_f64",
  "kernel:cuda:navatala_linalg_extract_diagonal_f64",
  "abi-r1:cuda:navatala_linalg_extract_diagonal_f64",
  "abi-r1:cuda:navatala_linalg_extract_diagonal_f64",
  3,
  kAbiArgs_cuda_navatala_linalg_extract_diagonal_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_extract_diagonal_k_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_extract_diagonal_k_f32 = {
  1,
  "navatala_linalg_extract_diagonal_k_f32",
  "cuda",
  "navatala_linalg_extract_diagonal_k_f32",
  "kernel:cuda:navatala_linalg_extract_diagonal_k_f32",
  "abi-r1:cuda:navatala_linalg_extract_diagonal_k_f32",
  "abi-r1:cuda:navatala_linalg_extract_diagonal_k_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_extract_diagonal_k_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_extract_diagonal_k_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_extract_diagonal_k_f64 = {
  1,
  "navatala_linalg_extract_diagonal_k_f64",
  "cuda",
  "navatala_linalg_extract_diagonal_k_f64",
  "kernel:cuda:navatala_linalg_extract_diagonal_k_f64",
  "abi-r1:cuda:navatala_linalg_extract_diagonal_k_f64",
  "abi-r1:cuda:navatala_linalg_extract_diagonal_k_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_extract_diagonal_k_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_set_diagonal_f32[] = {
  { "val", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_set_diagonal_f32 = {
  1,
  "navatala_linalg_set_diagonal_f32",
  "cuda",
  "navatala_linalg_set_diagonal_f32",
  "kernel:cuda:navatala_linalg_set_diagonal_f32",
  "abi-r1:cuda:navatala_linalg_set_diagonal_f32",
  "abi-r1:cuda:navatala_linalg_set_diagonal_f32",
  3,
  kAbiArgs_cuda_navatala_linalg_set_diagonal_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_set_diagonal_f64[] = {
  { "val", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_set_diagonal_f64 = {
  1,
  "navatala_linalg_set_diagonal_f64",
  "cuda",
  "navatala_linalg_set_diagonal_f64",
  "kernel:cuda:navatala_linalg_set_diagonal_f64",
  "abi-r1:cuda:navatala_linalg_set_diagonal_f64",
  "abi-r1:cuda:navatala_linalg_set_diagonal_f64",
  3,
  kAbiArgs_cuda_navatala_linalg_set_diagonal_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_diag_to_matrix_f32[] = {
  { "v", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_diag_to_matrix_f32 = {
  1,
  "navatala_linalg_diag_to_matrix_f32",
  "cuda",
  "navatala_linalg_diag_to_matrix_f32",
  "kernel:cuda:navatala_linalg_diag_to_matrix_f32",
  "abi-r1:cuda:navatala_linalg_diag_to_matrix_f32",
  "abi-r1:cuda:navatala_linalg_diag_to_matrix_f32",
  3,
  kAbiArgs_cuda_navatala_linalg_diag_to_matrix_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_diag_to_matrix_f64[] = {
  { "v", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "A", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_diag_to_matrix_f64 = {
  1,
  "navatala_linalg_diag_to_matrix_f64",
  "cuda",
  "navatala_linalg_diag_to_matrix_f64",
  "kernel:cuda:navatala_linalg_diag_to_matrix_f64",
  "abi-r1:cuda:navatala_linalg_diag_to_matrix_f64",
  "abi-r1:cuda:navatala_linalg_diag_to_matrix_f64",
  3,
  kAbiArgs_cuda_navatala_linalg_diag_to_matrix_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_shift_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_shift_rows_f32 = {
  1,
  "navatala_linalg_shift_rows_f32",
  "cuda",
  "navatala_linalg_shift_rows_f32",
  "kernel:cuda:navatala_linalg_shift_rows_f32",
  "abi-r1:cuda:navatala_linalg_shift_rows_f32",
  "abi-r1:cuda:navatala_linalg_shift_rows_f32",
  5,
  kAbiArgs_cuda_navatala_linalg_shift_rows_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_shift_rows_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_shift_rows_f64 = {
  1,
  "navatala_linalg_shift_rows_f64",
  "cuda",
  "navatala_linalg_shift_rows_f64",
  "kernel:cuda:navatala_linalg_shift_rows_f64",
  "abi-r1:cuda:navatala_linalg_shift_rows_f64",
  "abi-r1:cuda:navatala_linalg_shift_rows_f64",
  5,
  kAbiArgs_cuda_navatala_linalg_shift_rows_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_shift_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_shift_cols_f32 = {
  1,
  "navatala_linalg_shift_cols_f32",
  "cuda",
  "navatala_linalg_shift_cols_f32",
  "kernel:cuda:navatala_linalg_shift_cols_f32",
  "abi-r1:cuda:navatala_linalg_shift_cols_f32",
  "abi-r1:cuda:navatala_linalg_shift_cols_f32",
  5,
  kAbiArgs_cuda_navatala_linalg_shift_cols_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_shift_cols_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_shift_cols_f64 = {
  1,
  "navatala_linalg_shift_cols_f64",
  "cuda",
  "navatala_linalg_shift_cols_f64",
  "kernel:cuda:navatala_linalg_shift_cols_f64",
  "abi-r1:cuda:navatala_linalg_shift_cols_f64",
  "abi-r1:cuda:navatala_linalg_shift_cols_f64",
  5,
  kAbiArgs_cuda_navatala_linalg_shift_cols_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_roll_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_roll_rows_f32 = {
  1,
  "navatala_linalg_roll_rows_f32",
  "cuda",
  "navatala_linalg_roll_rows_f32",
  "kernel:cuda:navatala_linalg_roll_rows_f32",
  "abi-r1:cuda:navatala_linalg_roll_rows_f32",
  "abi-r1:cuda:navatala_linalg_roll_rows_f32",
  5,
  kAbiArgs_cuda_navatala_linalg_roll_rows_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_roll_rows_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_roll_rows_f64 = {
  1,
  "navatala_linalg_roll_rows_f64",
  "cuda",
  "navatala_linalg_roll_rows_f64",
  "kernel:cuda:navatala_linalg_roll_rows_f64",
  "abi-r1:cuda:navatala_linalg_roll_rows_f64",
  "abi-r1:cuda:navatala_linalg_roll_rows_f64",
  5,
  kAbiArgs_cuda_navatala_linalg_roll_rows_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_roll_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_roll_cols_f32 = {
  1,
  "navatala_linalg_roll_cols_f32",
  "cuda",
  "navatala_linalg_roll_cols_f32",
  "kernel:cuda:navatala_linalg_roll_cols_f32",
  "abi-r1:cuda:navatala_linalg_roll_cols_f32",
  "abi-r1:cuda:navatala_linalg_roll_cols_f32",
  5,
  kAbiArgs_cuda_navatala_linalg_roll_cols_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_roll_cols_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_roll_cols_f64 = {
  1,
  "navatala_linalg_roll_cols_f64",
  "cuda",
  "navatala_linalg_roll_cols_f64",
  "kernel:cuda:navatala_linalg_roll_cols_f64",
  "abi-r1:cuda:navatala_linalg_roll_cols_f64",
  "abi-r1:cuda:navatala_linalg_roll_cols_f64",
  5,
  kAbiArgs_cuda_navatala_linalg_roll_cols_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_sort_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_sort_rows_f32 = {
  1,
  "navatala_linalg_sort_rows_f32",
  "cuda",
  "navatala_linalg_sort_rows_f32",
  "kernel:cuda:navatala_linalg_sort_rows_f32",
  "abi-r1:cuda:navatala_linalg_sort_rows_f32",
  "abi-r1:cuda:navatala_linalg_sort_rows_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_sort_rows_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_sort_rows_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_sort_rows_f64 = {
  1,
  "navatala_linalg_sort_rows_f64",
  "cuda",
  "navatala_linalg_sort_rows_f64",
  "kernel:cuda:navatala_linalg_sort_rows_f64",
  "abi-r1:cuda:navatala_linalg_sort_rows_f64",
  "abi-r1:cuda:navatala_linalg_sort_rows_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_sort_rows_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_sort_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_sort_cols_f32 = {
  1,
  "navatala_linalg_sort_cols_f32",
  "cuda",
  "navatala_linalg_sort_cols_f32",
  "kernel:cuda:navatala_linalg_sort_cols_f32",
  "abi-r1:cuda:navatala_linalg_sort_cols_f32",
  "abi-r1:cuda:navatala_linalg_sort_cols_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_sort_cols_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_sort_cols_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_sort_cols_f64 = {
  1,
  "navatala_linalg_sort_cols_f64",
  "cuda",
  "navatala_linalg_sort_cols_f64",
  "kernel:cuda:navatala_linalg_sort_cols_f64",
  "abi-r1:cuda:navatala_linalg_sort_cols_f64",
  "abi-r1:cuda:navatala_linalg_sort_cols_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_sort_cols_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_argsort_rows_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_argsort_rows_f32 = {
  1,
  "navatala_linalg_argsort_rows_f32",
  "cuda",
  "navatala_linalg_argsort_rows_f32",
  "kernel:cuda:navatala_linalg_argsort_rows_f32",
  "abi-r1:cuda:navatala_linalg_argsort_rows_f32",
  "abi-r1:cuda:navatala_linalg_argsort_rows_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_argsort_rows_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_argsort_rows_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_argsort_rows_f64 = {
  1,
  "navatala_linalg_argsort_rows_f64",
  "cuda",
  "navatala_linalg_argsort_rows_f64",
  "kernel:cuda:navatala_linalg_argsort_rows_f64",
  "abi-r1:cuda:navatala_linalg_argsort_rows_f64",
  "abi-r1:cuda:navatala_linalg_argsort_rows_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_argsort_rows_f64
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_argsort_cols_f32[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_argsort_cols_f32 = {
  1,
  "navatala_linalg_argsort_cols_f32",
  "cuda",
  "navatala_linalg_argsort_cols_f32",
  "kernel:cuda:navatala_linalg_argsort_cols_f32",
  "abi-r1:cuda:navatala_linalg_argsort_cols_f32",
  "abi-r1:cuda:navatala_linalg_argsort_cols_f32",
  4,
  kAbiArgs_cuda_navatala_linalg_argsort_cols_f32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_linalg_argsort_cols_f64[] = {
  { "A", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_linalg_argsort_cols_f64 = {
  1,
  "navatala_linalg_argsort_cols_f64",
  "cuda",
  "navatala_linalg_argsort_cols_f64",
  "kernel:cuda:navatala_linalg_argsort_cols_f64",
  "abi-r1:cuda:navatala_linalg_argsort_cols_f64",
  "abi-r1:cuda:navatala_linalg_argsort_cols_f64",
  4,
  kAbiArgs_cuda_navatala_linalg_argsort_cols_f64
};

bool tryGetKernelAbiManifest_cuda_linalg(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "cuda" && kernelName == "navatala_linalg_norm_linf_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_norm_linf_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_norm_linf_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_norm_linf_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_eig_select_top_k_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_eig_select_top_k_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_eig_select_top_k_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_eig_select_top_k_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_cast_f32_to_f16") {
    out = &kAbiManifest_cuda_navatala_linalg_cast_f32_to_f16;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_cast_f64_to_f16") {
    out = &kAbiManifest_cuda_navatala_linalg_cast_f64_to_f16;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_cast_f16_to_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_cast_f16_to_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_cast_f16_to_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_cast_f16_to_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_cast_f32_to_bf16") {
    out = &kAbiManifest_cuda_navatala_linalg_cast_f32_to_bf16;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_cast_f64_to_bf16") {
    out = &kAbiManifest_cuda_navatala_linalg_cast_f64_to_bf16;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_cast_b_f16_to_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_cast_b_f16_to_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_cast_b_f16_to_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_cast_b_f16_to_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_sum_f16") {
    out = &kAbiManifest_cuda_navatala_linalg_reduce_sum_f16;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_sum_bf16") {
    out = &kAbiManifest_cuda_navatala_linalg_reduce_sum_bf16;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_max_f16") {
    out = &kAbiManifest_cuda_navatala_linalg_reduce_max_f16;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_max_bf16") {
    out = &kAbiManifest_cuda_navatala_linalg_reduce_max_bf16;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_add_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_add_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_add_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_add_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_sub_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_sub_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_sub_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_sub_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_mul_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_mul_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_mul_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_mul_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_div_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_div_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_div_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_div_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_pow_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_pow_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_pow_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_pow_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_sqrt_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_sqrt_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_sqrt_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_sqrt_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_exp_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_exp_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_exp_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_exp_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_log_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_log_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_log_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_log_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_abs_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_abs_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_abs_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_abs_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_sign_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_sign_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_sign_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_elt_sign_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_transpose_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_transpose_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_transpose_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_transpose_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_transpose_in_place_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_transpose_in_place_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_transpose_in_place_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_transpose_in_place_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_transpose_batched_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_transpose_batched_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_transpose_batched_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_transpose_batched_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_map_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_map_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_map_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_map_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_row_sum_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_reduce_row_sum_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_row_sum_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_reduce_row_sum_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_col_sum_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_reduce_col_sum_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_col_sum_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_reduce_col_sum_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_row_max_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_reduce_row_max_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_row_max_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_reduce_row_max_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_col_max_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_reduce_col_max_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_col_max_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_reduce_col_max_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_map_reduce_sum_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_map_reduce_sum_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_map_reduce_sum_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_map_reduce_sum_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_outer_product_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_outer_product_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_outer_product_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_outer_product_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_batched_dot_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_batched_dot_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_batched_dot_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_batched_dot_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_trace_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_trace_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_trace_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_trace_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_frobenius_norm_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_frobenius_norm_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_frobenius_norm_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_frobenius_norm_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_extract_upper_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_extract_upper_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_extract_upper_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_extract_upper_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_extract_lower_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_extract_lower_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_extract_lower_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_extract_lower_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_set_upper_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_set_upper_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_set_upper_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_set_upper_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_set_lower_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_set_lower_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_set_lower_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_set_lower_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_gather_rows_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_gather_rows_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_gather_rows_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_gather_rows_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_gather_cols_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_gather_cols_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_gather_cols_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_gather_cols_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_gather_elements_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_gather_elements_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_gather_elements_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_gather_elements_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_gather_batched_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_gather_batched_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_gather_batched_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_gather_batched_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_scatter_rows_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_scatter_rows_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_scatter_rows_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_scatter_rows_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_scatter_cols_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_scatter_cols_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_scatter_cols_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_scatter_cols_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_scatter_add_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_scatter_add_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_scatter_add_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_scatter_add_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_scatter_max_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_scatter_max_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_scatter_max_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_scatter_max_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_slice_rows_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_slice_rows_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_slice_rows_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_slice_rows_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_slice_cols_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_slice_cols_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_slice_cols_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_slice_cols_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_slice_block_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_slice_block_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_slice_block_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_slice_block_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_strided_slice_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_strided_slice_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_strided_slice_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_strided_slice_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_extract_diagonal_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_extract_diagonal_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_extract_diagonal_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_extract_diagonal_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_extract_diagonal_k_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_extract_diagonal_k_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_extract_diagonal_k_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_extract_diagonal_k_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_set_diagonal_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_set_diagonal_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_set_diagonal_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_set_diagonal_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_diag_to_matrix_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_diag_to_matrix_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_diag_to_matrix_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_diag_to_matrix_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_shift_rows_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_shift_rows_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_shift_rows_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_shift_rows_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_shift_cols_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_shift_cols_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_shift_cols_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_shift_cols_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_roll_rows_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_roll_rows_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_roll_rows_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_roll_rows_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_roll_cols_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_roll_cols_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_roll_cols_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_roll_cols_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_sort_rows_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_sort_rows_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_sort_rows_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_sort_rows_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_sort_cols_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_sort_cols_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_sort_cols_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_sort_cols_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_argsort_rows_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_argsort_rows_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_argsort_rows_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_argsort_rows_f64;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_argsort_cols_f32") {
    out = &kAbiManifest_cuda_navatala_linalg_argsort_cols_f32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_argsort_cols_f64") {
    out = &kAbiManifest_cuda_navatala_linalg_argsort_cols_f64;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_cuda_linalg(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "cuda" && kernelName == "navatala_linalg_norm_linf_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_norm_linf_f32";
    std::string_view sv(k_cuda_navatala_linalg_norm_linf_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_norm_linf_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_norm_linf_f64";
    std::string_view sv(k_cuda_navatala_linalg_norm_linf_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_eig_select_top_k_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_eig_select_top_k_f32";
    std::string_view sv(k_cuda_navatala_linalg_eig_select_top_k_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_eig_select_top_k_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_eig_select_top_k_f64";
    std::string_view sv(k_cuda_navatala_linalg_eig_select_top_k_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_cast_f32_to_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_cast_f32_to_f16";
    std::string_view sv(k_cuda_navatala_linalg_cast_f32_to_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_cast_f64_to_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_cast_f64_to_f16";
    std::string_view sv(k_cuda_navatala_linalg_cast_f64_to_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_cast_f16_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_cast_f16_to_f32";
    std::string_view sv(k_cuda_navatala_linalg_cast_f16_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_cast_f16_to_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_cast_f16_to_f64";
    std::string_view sv(k_cuda_navatala_linalg_cast_f16_to_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_cast_f32_to_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_cast_f32_to_bf16";
    std::string_view sv(k_cuda_navatala_linalg_cast_f32_to_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_cast_f64_to_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_cast_f64_to_bf16";
    std::string_view sv(k_cuda_navatala_linalg_cast_f64_to_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_cast_b_f16_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_cast_b_f16_to_f32";
    std::string_view sv(k_cuda_navatala_linalg_cast_b_f16_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_cast_b_f16_to_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_cast_b_f16_to_f64";
    std::string_view sv(k_cuda_navatala_linalg_cast_b_f16_to_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_sum_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_reduce_sum_f16";
    std::string_view sv(k_cuda_navatala_linalg_reduce_sum_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_sum_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_reduce_sum_bf16";
    std::string_view sv(k_cuda_navatala_linalg_reduce_sum_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_max_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_reduce_max_f16";
    std::string_view sv(k_cuda_navatala_linalg_reduce_max_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_max_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_reduce_max_bf16";
    std::string_view sv(k_cuda_navatala_linalg_reduce_max_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_add_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_add_f32";
    std::string_view sv(k_cuda_navatala_linalg_elt_add_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_add_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_add_f64";
    std::string_view sv(k_cuda_navatala_linalg_elt_add_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_sub_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_sub_f32";
    std::string_view sv(k_cuda_navatala_linalg_elt_sub_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_sub_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_sub_f64";
    std::string_view sv(k_cuda_navatala_linalg_elt_sub_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_mul_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_mul_f32";
    std::string_view sv(k_cuda_navatala_linalg_elt_mul_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_mul_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_mul_f64";
    std::string_view sv(k_cuda_navatala_linalg_elt_mul_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_div_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_div_f32";
    std::string_view sv(k_cuda_navatala_linalg_elt_div_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_div_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_div_f64";
    std::string_view sv(k_cuda_navatala_linalg_elt_div_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_pow_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_pow_f32";
    std::string_view sv(k_cuda_navatala_linalg_elt_pow_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_pow_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_pow_f64";
    std::string_view sv(k_cuda_navatala_linalg_elt_pow_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_sqrt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_sqrt_f32";
    std::string_view sv(k_cuda_navatala_linalg_elt_sqrt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_sqrt_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_sqrt_f64";
    std::string_view sv(k_cuda_navatala_linalg_elt_sqrt_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_exp_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_exp_f32";
    std::string_view sv(k_cuda_navatala_linalg_elt_exp_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_exp_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_exp_f64";
    std::string_view sv(k_cuda_navatala_linalg_elt_exp_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_log_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_log_f32";
    std::string_view sv(k_cuda_navatala_linalg_elt_log_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_log_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_log_f64";
    std::string_view sv(k_cuda_navatala_linalg_elt_log_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_abs_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_abs_f32";
    std::string_view sv(k_cuda_navatala_linalg_elt_abs_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_abs_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_abs_f64";
    std::string_view sv(k_cuda_navatala_linalg_elt_abs_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_sign_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_sign_f32";
    std::string_view sv(k_cuda_navatala_linalg_elt_sign_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_elt_sign_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_elt_sign_f64";
    std::string_view sv(k_cuda_navatala_linalg_elt_sign_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_transpose_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_transpose_f32";
    std::string_view sv(k_cuda_navatala_linalg_transpose_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_transpose_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_transpose_f64";
    std::string_view sv(k_cuda_navatala_linalg_transpose_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_transpose_in_place_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_transpose_in_place_f32";
    std::string_view sv(k_cuda_navatala_linalg_transpose_in_place_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_transpose_in_place_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_transpose_in_place_f64";
    std::string_view sv(k_cuda_navatala_linalg_transpose_in_place_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_transpose_batched_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_transpose_batched_f32";
    std::string_view sv(k_cuda_navatala_linalg_transpose_batched_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_transpose_batched_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_transpose_batched_f64";
    std::string_view sv(k_cuda_navatala_linalg_transpose_batched_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_map_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_map_f32";
    std::string_view sv(k_cuda_navatala_linalg_map_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_map_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_map_f64";
    std::string_view sv(k_cuda_navatala_linalg_map_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_row_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_reduce_row_sum_f32";
    std::string_view sv(k_cuda_navatala_linalg_reduce_row_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_row_sum_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_reduce_row_sum_f64";
    std::string_view sv(k_cuda_navatala_linalg_reduce_row_sum_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_col_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_reduce_col_sum_f32";
    std::string_view sv(k_cuda_navatala_linalg_reduce_col_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_col_sum_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_reduce_col_sum_f64";
    std::string_view sv(k_cuda_navatala_linalg_reduce_col_sum_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_row_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_reduce_row_max_f32";
    std::string_view sv(k_cuda_navatala_linalg_reduce_row_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_row_max_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_reduce_row_max_f64";
    std::string_view sv(k_cuda_navatala_linalg_reduce_row_max_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_col_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_reduce_col_max_f32";
    std::string_view sv(k_cuda_navatala_linalg_reduce_col_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_reduce_col_max_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_reduce_col_max_f64";
    std::string_view sv(k_cuda_navatala_linalg_reduce_col_max_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_map_reduce_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_map_reduce_sum_f32";
    std::string_view sv(k_cuda_navatala_linalg_map_reduce_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_map_reduce_sum_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_map_reduce_sum_f64";
    std::string_view sv(k_cuda_navatala_linalg_map_reduce_sum_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_outer_product_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_outer_product_f32";
    std::string_view sv(k_cuda_navatala_linalg_outer_product_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_outer_product_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_outer_product_f64";
    std::string_view sv(k_cuda_navatala_linalg_outer_product_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_batched_dot_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_batched_dot_f32";
    std::string_view sv(k_cuda_navatala_linalg_batched_dot_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_batched_dot_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_batched_dot_f64";
    std::string_view sv(k_cuda_navatala_linalg_batched_dot_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_trace_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_trace_f32";
    std::string_view sv(k_cuda_navatala_linalg_trace_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_trace_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_trace_f64";
    std::string_view sv(k_cuda_navatala_linalg_trace_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_frobenius_norm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_frobenius_norm_f32";
    std::string_view sv(k_cuda_navatala_linalg_frobenius_norm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_frobenius_norm_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_frobenius_norm_f64";
    std::string_view sv(k_cuda_navatala_linalg_frobenius_norm_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_extract_upper_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_extract_upper_f32";
    std::string_view sv(k_cuda_navatala_linalg_extract_upper_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_extract_upper_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_extract_upper_f64";
    std::string_view sv(k_cuda_navatala_linalg_extract_upper_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_extract_lower_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_extract_lower_f32";
    std::string_view sv(k_cuda_navatala_linalg_extract_lower_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_extract_lower_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_extract_lower_f64";
    std::string_view sv(k_cuda_navatala_linalg_extract_lower_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_set_upper_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_set_upper_f32";
    std::string_view sv(k_cuda_navatala_linalg_set_upper_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_set_upper_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_set_upper_f64";
    std::string_view sv(k_cuda_navatala_linalg_set_upper_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_set_lower_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_set_lower_f32";
    std::string_view sv(k_cuda_navatala_linalg_set_lower_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_set_lower_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_set_lower_f64";
    std::string_view sv(k_cuda_navatala_linalg_set_lower_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_gather_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_gather_rows_f32";
    std::string_view sv(k_cuda_navatala_linalg_gather_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_gather_rows_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_gather_rows_f64";
    std::string_view sv(k_cuda_navatala_linalg_gather_rows_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_gather_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_gather_cols_f32";
    std::string_view sv(k_cuda_navatala_linalg_gather_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_gather_cols_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_gather_cols_f64";
    std::string_view sv(k_cuda_navatala_linalg_gather_cols_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_gather_elements_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_gather_elements_f32";
    std::string_view sv(k_cuda_navatala_linalg_gather_elements_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_gather_elements_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_gather_elements_f64";
    std::string_view sv(k_cuda_navatala_linalg_gather_elements_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_gather_batched_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_gather_batched_f32";
    std::string_view sv(k_cuda_navatala_linalg_gather_batched_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_gather_batched_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_gather_batched_f64";
    std::string_view sv(k_cuda_navatala_linalg_gather_batched_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_scatter_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_scatter_rows_f32";
    std::string_view sv(k_cuda_navatala_linalg_scatter_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_scatter_rows_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_scatter_rows_f64";
    std::string_view sv(k_cuda_navatala_linalg_scatter_rows_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_scatter_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_scatter_cols_f32";
    std::string_view sv(k_cuda_navatala_linalg_scatter_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_scatter_cols_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_scatter_cols_f64";
    std::string_view sv(k_cuda_navatala_linalg_scatter_cols_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_scatter_add_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_scatter_add_f32";
    std::string_view sv(k_cuda_navatala_linalg_scatter_add_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_scatter_add_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_scatter_add_f64";
    std::string_view sv(k_cuda_navatala_linalg_scatter_add_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_scatter_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_scatter_max_f32";
    std::string_view sv(k_cuda_navatala_linalg_scatter_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_scatter_max_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_scatter_max_f64";
    std::string_view sv(k_cuda_navatala_linalg_scatter_max_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_slice_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_slice_rows_f32";
    std::string_view sv(k_cuda_navatala_linalg_slice_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_slice_rows_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_slice_rows_f64";
    std::string_view sv(k_cuda_navatala_linalg_slice_rows_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_slice_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_slice_cols_f32";
    std::string_view sv(k_cuda_navatala_linalg_slice_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_slice_cols_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_slice_cols_f64";
    std::string_view sv(k_cuda_navatala_linalg_slice_cols_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_slice_block_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_slice_block_f32";
    std::string_view sv(k_cuda_navatala_linalg_slice_block_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_slice_block_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_slice_block_f64";
    std::string_view sv(k_cuda_navatala_linalg_slice_block_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_strided_slice_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_strided_slice_f32";
    std::string_view sv(k_cuda_navatala_linalg_strided_slice_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_strided_slice_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_strided_slice_f64";
    std::string_view sv(k_cuda_navatala_linalg_strided_slice_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_extract_diagonal_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_extract_diagonal_f32";
    std::string_view sv(k_cuda_navatala_linalg_extract_diagonal_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_extract_diagonal_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_extract_diagonal_f64";
    std::string_view sv(k_cuda_navatala_linalg_extract_diagonal_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_extract_diagonal_k_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_extract_diagonal_k_f32";
    std::string_view sv(k_cuda_navatala_linalg_extract_diagonal_k_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_extract_diagonal_k_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_extract_diagonal_k_f64";
    std::string_view sv(k_cuda_navatala_linalg_extract_diagonal_k_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_set_diagonal_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_set_diagonal_f32";
    std::string_view sv(k_cuda_navatala_linalg_set_diagonal_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_set_diagonal_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_set_diagonal_f64";
    std::string_view sv(k_cuda_navatala_linalg_set_diagonal_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_diag_to_matrix_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_diag_to_matrix_f32";
    std::string_view sv(k_cuda_navatala_linalg_diag_to_matrix_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_diag_to_matrix_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_diag_to_matrix_f64";
    std::string_view sv(k_cuda_navatala_linalg_diag_to_matrix_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_shift_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_shift_rows_f32";
    std::string_view sv(k_cuda_navatala_linalg_shift_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_shift_rows_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_shift_rows_f64";
    std::string_view sv(k_cuda_navatala_linalg_shift_rows_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_shift_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_shift_cols_f32";
    std::string_view sv(k_cuda_navatala_linalg_shift_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_shift_cols_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_shift_cols_f64";
    std::string_view sv(k_cuda_navatala_linalg_shift_cols_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_roll_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_roll_rows_f32";
    std::string_view sv(k_cuda_navatala_linalg_roll_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_roll_rows_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_roll_rows_f64";
    std::string_view sv(k_cuda_navatala_linalg_roll_rows_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_roll_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_roll_cols_f32";
    std::string_view sv(k_cuda_navatala_linalg_roll_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_roll_cols_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_roll_cols_f64";
    std::string_view sv(k_cuda_navatala_linalg_roll_cols_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_sort_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_sort_rows_f32";
    std::string_view sv(k_cuda_navatala_linalg_sort_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_sort_rows_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_sort_rows_f64";
    std::string_view sv(k_cuda_navatala_linalg_sort_rows_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_sort_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_sort_cols_f32";
    std::string_view sv(k_cuda_navatala_linalg_sort_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_sort_cols_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_sort_cols_f64";
    std::string_view sv(k_cuda_navatala_linalg_sort_cols_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_argsort_rows_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_argsort_rows_f32";
    std::string_view sv(k_cuda_navatala_linalg_argsort_rows_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_argsort_rows_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_argsort_rows_f64";
    std::string_view sv(k_cuda_navatala_linalg_argsort_rows_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_argsort_cols_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_argsort_cols_f32";
    std::string_view sv(k_cuda_navatala_linalg_argsort_cols_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_linalg_argsort_cols_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_linalg_argsort_cols_f64";
    std::string_view sv(k_cuda_navatala_linalg_argsort_cols_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

