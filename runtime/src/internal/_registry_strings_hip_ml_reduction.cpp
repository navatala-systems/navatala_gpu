// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `hip_ml_reduction` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_hip_navatala_ml_reduction_sum_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_reduction_sum_f32(const float* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      float raw = _input[idx];
      float v = raw;
      gsAcc = (gsAcc + v);
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
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_prod_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_reduction_prod_f32(const float* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x3f800000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      float raw = _input[idx];
      float v = raw;
      gsAcc = (gsAcc * v);
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
      float acc = (mine * other);
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = redStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_min_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_reduction_min_f32(const float* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x7f7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      float raw = _input[idx];
      float v = raw;
      gsAcc = (((gsAcc < v)) ? (gsAcc) : (v));
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
      float acc = (((mine < other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = redStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_max_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_reduction_max_f32(const float* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0xff7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      float raw = _input[idx];
      float v = raw;
      gsAcc = (((gsAcc > v)) ? (gsAcc) : (v));
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
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = redStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_amax_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_reduction_amax_f32(const float* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0xff7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      float raw = _input[idx];
      float v = abs(raw);
      gsAcc = (((gsAcc > v)) ? (gsAcc) : (v));
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
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = redStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_norm1_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_reduction_norm1_f32(const float* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      float raw = _input[idx];
      float v = abs(raw);
      gsAcc = (gsAcc + v);
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
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_avg_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_reduction_avg_f32(const float* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      float raw = _input[idx];
      float v = raw;
      gsAcc = (gsAcc + v);
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
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = (reduced / nF);
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_norm2_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_reduction_norm2_f32(const float* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      float raw = _input[idx];
      float v = (raw * raw);
      gsAcc = (gsAcc + v);
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
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = sqrt(reduced);
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_sum_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_reduction_sum_f16(const __half* _input, const unsigned int* count, __half* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      __half raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (gsAcc + v);
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
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    __half outV = ((__half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_prod_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_reduction_prod_f16(const __half* _input, const unsigned int* count, __half* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x3f800000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      __half raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (gsAcc * v);
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
      float acc = (mine * other);
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = redStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    __half outV = ((__half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_min_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_reduction_min_f16(const __half* _input, const unsigned int* count, __half* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x7f7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      __half raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (((gsAcc < v)) ? (gsAcc) : (v));
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
      float acc = (((mine < other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = redStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    __half outV = ((__half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_max_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_reduction_max_f16(const __half* _input, const unsigned int* count, __half* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0xff7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      __half raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (((gsAcc > v)) ? (gsAcc) : (v));
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
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = redStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    __half outV = ((__half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_amax_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_reduction_amax_f16(const __half* _input, const unsigned int* count, __half* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0xff7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      __half raw = _input[idx];
      float v = abs(((float)(raw)));
      gsAcc = (((gsAcc > v)) ? (gsAcc) : (v));
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
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = redStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    __half outV = ((__half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_norm1_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_reduction_norm1_f16(const __half* _input, const unsigned int* count, __half* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      __half raw = _input[idx];
      float v = abs(((float)(raw)));
      gsAcc = (gsAcc + v);
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
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    __half outV = ((__half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_avg_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_reduction_avg_f16(const __half* _input, const unsigned int* count, __half* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      __half raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (gsAcc + v);
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
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = (reduced / nF);
    __half outV = ((__half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_norm2_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_reduction_norm2_f16(const __half* _input, const unsigned int* count, __half* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      __half raw = _input[idx];
      float v = (((float)(raw)) * ((float)(raw)));
      gsAcc = (gsAcc + v);
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
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = sqrt(reduced);
    __half outV = ((__half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_sum_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_reduction_sum_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      __hip_bfloat16 raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (gsAcc + v);
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
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    __hip_bfloat16 outV = ((__hip_bfloat16)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_prod_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_reduction_prod_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x3f800000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      __hip_bfloat16 raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (gsAcc * v);
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
      float acc = (mine * other);
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = redStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    __hip_bfloat16 outV = ((__hip_bfloat16)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_min_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_reduction_min_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x7f7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      __hip_bfloat16 raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (((gsAcc < v)) ? (gsAcc) : (v));
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
      float acc = (((mine < other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = redStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    __hip_bfloat16 outV = ((__hip_bfloat16)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_max_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_reduction_max_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0xff7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      __hip_bfloat16 raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (((gsAcc > v)) ? (gsAcc) : (v));
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
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = redStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    __hip_bfloat16 outV = ((__hip_bfloat16)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_amax_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_reduction_amax_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0xff7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      __hip_bfloat16 raw = _input[idx];
      float v = abs(((float)(raw)));
      gsAcc = (((gsAcc > v)) ? (gsAcc) : (v));
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
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = redStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    __hip_bfloat16 outV = ((__hip_bfloat16)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_norm1_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_reduction_norm1_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      __hip_bfloat16 raw = _input[idx];
      float v = abs(((float)(raw)));
      gsAcc = (gsAcc + v);
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
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    __hip_bfloat16 outV = ((__hip_bfloat16)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_avg_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_reduction_avg_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      __hip_bfloat16 raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (gsAcc + v);
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
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = (reduced / nF);
    __hip_bfloat16 outV = ((__hip_bfloat16)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_hip_navatala_ml_reduction_norm2_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_reduction_norm2_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if (idx < countVal) {
      __hip_bfloat16 raw = _input[idx];
      float v = (((float)(raw)) * ((float)(raw)));
      gsAcc = (gsAcc + v);
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
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = sqrt(reduced);
    __hip_bfloat16 outV = ((__hip_bfloat16)(finalF));
    result[0] = outV;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_sum_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_sum_f32 = {
  1,
  "navatala_ml_reduction_sum_f32",
  "hip",
  "navatala_ml_reduction_sum_f32",
  "kernel:hip:navatala_ml_reduction_sum_f32",
  "abi-r1:hip:navatala_ml_reduction_sum_f32",
  "abi-r1:hip:navatala_ml_reduction_sum_f32",
  3,
  kAbiArgs_hip_navatala_ml_reduction_sum_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_prod_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_prod_f32 = {
  1,
  "navatala_ml_reduction_prod_f32",
  "hip",
  "navatala_ml_reduction_prod_f32",
  "kernel:hip:navatala_ml_reduction_prod_f32",
  "abi-r1:hip:navatala_ml_reduction_prod_f32",
  "abi-r1:hip:navatala_ml_reduction_prod_f32",
  3,
  kAbiArgs_hip_navatala_ml_reduction_prod_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_min_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_min_f32 = {
  1,
  "navatala_ml_reduction_min_f32",
  "hip",
  "navatala_ml_reduction_min_f32",
  "kernel:hip:navatala_ml_reduction_min_f32",
  "abi-r1:hip:navatala_ml_reduction_min_f32",
  "abi-r1:hip:navatala_ml_reduction_min_f32",
  3,
  kAbiArgs_hip_navatala_ml_reduction_min_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_max_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_max_f32 = {
  1,
  "navatala_ml_reduction_max_f32",
  "hip",
  "navatala_ml_reduction_max_f32",
  "kernel:hip:navatala_ml_reduction_max_f32",
  "abi-r1:hip:navatala_ml_reduction_max_f32",
  "abi-r1:hip:navatala_ml_reduction_max_f32",
  3,
  kAbiArgs_hip_navatala_ml_reduction_max_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_amax_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_amax_f32 = {
  1,
  "navatala_ml_reduction_amax_f32",
  "hip",
  "navatala_ml_reduction_amax_f32",
  "kernel:hip:navatala_ml_reduction_amax_f32",
  "abi-r1:hip:navatala_ml_reduction_amax_f32",
  "abi-r1:hip:navatala_ml_reduction_amax_f32",
  3,
  kAbiArgs_hip_navatala_ml_reduction_amax_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_norm1_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_norm1_f32 = {
  1,
  "navatala_ml_reduction_norm1_f32",
  "hip",
  "navatala_ml_reduction_norm1_f32",
  "kernel:hip:navatala_ml_reduction_norm1_f32",
  "abi-r1:hip:navatala_ml_reduction_norm1_f32",
  "abi-r1:hip:navatala_ml_reduction_norm1_f32",
  3,
  kAbiArgs_hip_navatala_ml_reduction_norm1_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_avg_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_avg_f32 = {
  1,
  "navatala_ml_reduction_avg_f32",
  "hip",
  "navatala_ml_reduction_avg_f32",
  "kernel:hip:navatala_ml_reduction_avg_f32",
  "abi-r1:hip:navatala_ml_reduction_avg_f32",
  "abi-r1:hip:navatala_ml_reduction_avg_f32",
  3,
  kAbiArgs_hip_navatala_ml_reduction_avg_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_norm2_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_norm2_f32 = {
  1,
  "navatala_ml_reduction_norm2_f32",
  "hip",
  "navatala_ml_reduction_norm2_f32",
  "kernel:hip:navatala_ml_reduction_norm2_f32",
  "abi-r1:hip:navatala_ml_reduction_norm2_f32",
  "abi-r1:hip:navatala_ml_reduction_norm2_f32",
  3,
  kAbiArgs_hip_navatala_ml_reduction_norm2_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_sum_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_sum_f16 = {
  1,
  "navatala_ml_reduction_sum_f16",
  "hip",
  "navatala_ml_reduction_sum_f16",
  "kernel:hip:navatala_ml_reduction_sum_f16",
  "abi-r1:hip:navatala_ml_reduction_sum_f16",
  "abi-r1:hip:navatala_ml_reduction_sum_f16",
  3,
  kAbiArgs_hip_navatala_ml_reduction_sum_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_prod_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_prod_f16 = {
  1,
  "navatala_ml_reduction_prod_f16",
  "hip",
  "navatala_ml_reduction_prod_f16",
  "kernel:hip:navatala_ml_reduction_prod_f16",
  "abi-r1:hip:navatala_ml_reduction_prod_f16",
  "abi-r1:hip:navatala_ml_reduction_prod_f16",
  3,
  kAbiArgs_hip_navatala_ml_reduction_prod_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_min_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_min_f16 = {
  1,
  "navatala_ml_reduction_min_f16",
  "hip",
  "navatala_ml_reduction_min_f16",
  "kernel:hip:navatala_ml_reduction_min_f16",
  "abi-r1:hip:navatala_ml_reduction_min_f16",
  "abi-r1:hip:navatala_ml_reduction_min_f16",
  3,
  kAbiArgs_hip_navatala_ml_reduction_min_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_max_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_max_f16 = {
  1,
  "navatala_ml_reduction_max_f16",
  "hip",
  "navatala_ml_reduction_max_f16",
  "kernel:hip:navatala_ml_reduction_max_f16",
  "abi-r1:hip:navatala_ml_reduction_max_f16",
  "abi-r1:hip:navatala_ml_reduction_max_f16",
  3,
  kAbiArgs_hip_navatala_ml_reduction_max_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_amax_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_amax_f16 = {
  1,
  "navatala_ml_reduction_amax_f16",
  "hip",
  "navatala_ml_reduction_amax_f16",
  "kernel:hip:navatala_ml_reduction_amax_f16",
  "abi-r1:hip:navatala_ml_reduction_amax_f16",
  "abi-r1:hip:navatala_ml_reduction_amax_f16",
  3,
  kAbiArgs_hip_navatala_ml_reduction_amax_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_norm1_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_norm1_f16 = {
  1,
  "navatala_ml_reduction_norm1_f16",
  "hip",
  "navatala_ml_reduction_norm1_f16",
  "kernel:hip:navatala_ml_reduction_norm1_f16",
  "abi-r1:hip:navatala_ml_reduction_norm1_f16",
  "abi-r1:hip:navatala_ml_reduction_norm1_f16",
  3,
  kAbiArgs_hip_navatala_ml_reduction_norm1_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_avg_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_avg_f16 = {
  1,
  "navatala_ml_reduction_avg_f16",
  "hip",
  "navatala_ml_reduction_avg_f16",
  "kernel:hip:navatala_ml_reduction_avg_f16",
  "abi-r1:hip:navatala_ml_reduction_avg_f16",
  "abi-r1:hip:navatala_ml_reduction_avg_f16",
  3,
  kAbiArgs_hip_navatala_ml_reduction_avg_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_norm2_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_norm2_f16 = {
  1,
  "navatala_ml_reduction_norm2_f16",
  "hip",
  "navatala_ml_reduction_norm2_f16",
  "kernel:hip:navatala_ml_reduction_norm2_f16",
  "abi-r1:hip:navatala_ml_reduction_norm2_f16",
  "abi-r1:hip:navatala_ml_reduction_norm2_f16",
  3,
  kAbiArgs_hip_navatala_ml_reduction_norm2_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_sum_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_sum_bf16 = {
  1,
  "navatala_ml_reduction_sum_bf16",
  "hip",
  "navatala_ml_reduction_sum_bf16",
  "kernel:hip:navatala_ml_reduction_sum_bf16",
  "abi-r1:hip:navatala_ml_reduction_sum_bf16",
  "abi-r1:hip:navatala_ml_reduction_sum_bf16",
  3,
  kAbiArgs_hip_navatala_ml_reduction_sum_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_prod_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_prod_bf16 = {
  1,
  "navatala_ml_reduction_prod_bf16",
  "hip",
  "navatala_ml_reduction_prod_bf16",
  "kernel:hip:navatala_ml_reduction_prod_bf16",
  "abi-r1:hip:navatala_ml_reduction_prod_bf16",
  "abi-r1:hip:navatala_ml_reduction_prod_bf16",
  3,
  kAbiArgs_hip_navatala_ml_reduction_prod_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_min_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_min_bf16 = {
  1,
  "navatala_ml_reduction_min_bf16",
  "hip",
  "navatala_ml_reduction_min_bf16",
  "kernel:hip:navatala_ml_reduction_min_bf16",
  "abi-r1:hip:navatala_ml_reduction_min_bf16",
  "abi-r1:hip:navatala_ml_reduction_min_bf16",
  3,
  kAbiArgs_hip_navatala_ml_reduction_min_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_max_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_max_bf16 = {
  1,
  "navatala_ml_reduction_max_bf16",
  "hip",
  "navatala_ml_reduction_max_bf16",
  "kernel:hip:navatala_ml_reduction_max_bf16",
  "abi-r1:hip:navatala_ml_reduction_max_bf16",
  "abi-r1:hip:navatala_ml_reduction_max_bf16",
  3,
  kAbiArgs_hip_navatala_ml_reduction_max_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_amax_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_amax_bf16 = {
  1,
  "navatala_ml_reduction_amax_bf16",
  "hip",
  "navatala_ml_reduction_amax_bf16",
  "kernel:hip:navatala_ml_reduction_amax_bf16",
  "abi-r1:hip:navatala_ml_reduction_amax_bf16",
  "abi-r1:hip:navatala_ml_reduction_amax_bf16",
  3,
  kAbiArgs_hip_navatala_ml_reduction_amax_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_norm1_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_norm1_bf16 = {
  1,
  "navatala_ml_reduction_norm1_bf16",
  "hip",
  "navatala_ml_reduction_norm1_bf16",
  "kernel:hip:navatala_ml_reduction_norm1_bf16",
  "abi-r1:hip:navatala_ml_reduction_norm1_bf16",
  "abi-r1:hip:navatala_ml_reduction_norm1_bf16",
  3,
  kAbiArgs_hip_navatala_ml_reduction_norm1_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_avg_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_avg_bf16 = {
  1,
  "navatala_ml_reduction_avg_bf16",
  "hip",
  "navatala_ml_reduction_avg_bf16",
  "kernel:hip:navatala_ml_reduction_avg_bf16",
  "abi-r1:hip:navatala_ml_reduction_avg_bf16",
  "abi-r1:hip:navatala_ml_reduction_avg_bf16",
  3,
  kAbiArgs_hip_navatala_ml_reduction_avg_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_reduction_norm2_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_reduction_norm2_bf16 = {
  1,
  "navatala_ml_reduction_norm2_bf16",
  "hip",
  "navatala_ml_reduction_norm2_bf16",
  "kernel:hip:navatala_ml_reduction_norm2_bf16",
  "abi-r1:hip:navatala_ml_reduction_norm2_bf16",
  "abi-r1:hip:navatala_ml_reduction_norm2_bf16",
  3,
  kAbiArgs_hip_navatala_ml_reduction_norm2_bf16
};

bool tryGetKernelAbiManifest_hip_ml_reduction(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "hip" && kernelName == "navatala_ml_reduction_sum_f32") {
    out = &kAbiManifest_hip_navatala_ml_reduction_sum_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_prod_f32") {
    out = &kAbiManifest_hip_navatala_ml_reduction_prod_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_min_f32") {
    out = &kAbiManifest_hip_navatala_ml_reduction_min_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_max_f32") {
    out = &kAbiManifest_hip_navatala_ml_reduction_max_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_amax_f32") {
    out = &kAbiManifest_hip_navatala_ml_reduction_amax_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_norm1_f32") {
    out = &kAbiManifest_hip_navatala_ml_reduction_norm1_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_avg_f32") {
    out = &kAbiManifest_hip_navatala_ml_reduction_avg_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_norm2_f32") {
    out = &kAbiManifest_hip_navatala_ml_reduction_norm2_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_sum_f16") {
    out = &kAbiManifest_hip_navatala_ml_reduction_sum_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_prod_f16") {
    out = &kAbiManifest_hip_navatala_ml_reduction_prod_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_min_f16") {
    out = &kAbiManifest_hip_navatala_ml_reduction_min_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_max_f16") {
    out = &kAbiManifest_hip_navatala_ml_reduction_max_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_amax_f16") {
    out = &kAbiManifest_hip_navatala_ml_reduction_amax_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_norm1_f16") {
    out = &kAbiManifest_hip_navatala_ml_reduction_norm1_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_avg_f16") {
    out = &kAbiManifest_hip_navatala_ml_reduction_avg_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_norm2_f16") {
    out = &kAbiManifest_hip_navatala_ml_reduction_norm2_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_sum_bf16") {
    out = &kAbiManifest_hip_navatala_ml_reduction_sum_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_prod_bf16") {
    out = &kAbiManifest_hip_navatala_ml_reduction_prod_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_min_bf16") {
    out = &kAbiManifest_hip_navatala_ml_reduction_min_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_max_bf16") {
    out = &kAbiManifest_hip_navatala_ml_reduction_max_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_amax_bf16") {
    out = &kAbiManifest_hip_navatala_ml_reduction_amax_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_norm1_bf16") {
    out = &kAbiManifest_hip_navatala_ml_reduction_norm1_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_avg_bf16") {
    out = &kAbiManifest_hip_navatala_ml_reduction_avg_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_norm2_bf16") {
    out = &kAbiManifest_hip_navatala_ml_reduction_norm2_bf16;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_hip_ml_reduction(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "hip" && kernelName == "navatala_ml_reduction_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_sum_f32";
    std::string_view sv(k_hip_navatala_ml_reduction_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_prod_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_prod_f32";
    std::string_view sv(k_hip_navatala_ml_reduction_prod_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_min_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_min_f32";
    std::string_view sv(k_hip_navatala_ml_reduction_min_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_max_f32";
    std::string_view sv(k_hip_navatala_ml_reduction_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_amax_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_amax_f32";
    std::string_view sv(k_hip_navatala_ml_reduction_amax_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_norm1_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_norm1_f32";
    std::string_view sv(k_hip_navatala_ml_reduction_norm1_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_avg_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_avg_f32";
    std::string_view sv(k_hip_navatala_ml_reduction_avg_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_norm2_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_norm2_f32";
    std::string_view sv(k_hip_navatala_ml_reduction_norm2_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_sum_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_sum_f16";
    std::string_view sv(k_hip_navatala_ml_reduction_sum_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_prod_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_prod_f16";
    std::string_view sv(k_hip_navatala_ml_reduction_prod_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_min_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_min_f16";
    std::string_view sv(k_hip_navatala_ml_reduction_min_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_max_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_max_f16";
    std::string_view sv(k_hip_navatala_ml_reduction_max_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_amax_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_amax_f16";
    std::string_view sv(k_hip_navatala_ml_reduction_amax_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_norm1_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_norm1_f16";
    std::string_view sv(k_hip_navatala_ml_reduction_norm1_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_avg_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_avg_f16";
    std::string_view sv(k_hip_navatala_ml_reduction_avg_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_norm2_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_norm2_f16";
    std::string_view sv(k_hip_navatala_ml_reduction_norm2_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_sum_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_sum_bf16";
    std::string_view sv(k_hip_navatala_ml_reduction_sum_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_prod_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_prod_bf16";
    std::string_view sv(k_hip_navatala_ml_reduction_prod_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_min_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_min_bf16";
    std::string_view sv(k_hip_navatala_ml_reduction_min_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_max_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_max_bf16";
    std::string_view sv(k_hip_navatala_ml_reduction_max_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_amax_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_amax_bf16";
    std::string_view sv(k_hip_navatala_ml_reduction_amax_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_norm1_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_norm1_bf16";
    std::string_view sv(k_hip_navatala_ml_reduction_norm1_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_avg_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_avg_bf16";
    std::string_view sv(k_hip_navatala_ml_reduction_avg_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_reduction_norm2_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_reduction_norm2_bf16";
    std::string_view sv(k_hip_navatala_ml_reduction_norm2_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

