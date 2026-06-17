// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `hip_ml_pointwise` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_hip_navatala_ml_pointwise_neg_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_neg_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (-x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_abs_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_abs_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = abs(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_sqrt_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_sqrt_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = sqrt(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_rsqrt_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_rsqrt_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (__uint_as_float(0x3f800000u) / sqrt(x));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_reciprocal_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_reciprocal_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (__uint_as_float(0x3f800000u) / x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_exp_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_exp_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = exp(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_log_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_log_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = log(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_sin_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_sin_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = sin(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cos_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_cos_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = cos(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_tan_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_tan_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (sin(x) / cos(x));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_floor_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_floor_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = floor(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_ceil_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_ceil_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = ceil(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_round_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_round_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = round(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_relu_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_relu_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (((x > __uint_as_float(0x00000000u))) ? (x) : (__uint_as_float(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_leaky_relu_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_leaky_relu_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (((x > __uint_as_float(0x00000000u))) ? (x) : ((__uint_as_float(0x3c23d70au) * x)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_elu_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_elu_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (((x > __uint_as_float(0x00000000u))) ? (x) : ((exp(x) - __uint_as_float(0x3f800000u))));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_tanh_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_tanh_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = tanh(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_sigmoid_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_sigmoid_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (__uint_as_float(0x3f800000u) / (__uint_as_float(0x3f800000u) + exp((-x))));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_softplus_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_softplus_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = log((__uint_as_float(0x3f800000u) + exp(x)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_swish_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_swish_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (x * (__uint_as_float(0x3f800000u) / (__uint_as_float(0x3f800000u) + exp((-x)))));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_gelu_tanh_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_gelu_tanh_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = ((__uint_as_float(0x3f000000u) * x) * (__uint_as_float(0x3f800000u) + tanh((__uint_as_float(0x3f4c422au) * (x + (__uint_as_float(0x3d372713u) * (x * (x * x))))))));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_add_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_add_f32(const float* a, const float* b, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (av + bv);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_sub_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_sub_f32(const float* a, const float* b, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (av - bv);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_mul_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_mul_f32(const float* a, const float* b, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (av * bv);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_div_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_div_f32(const float* a, const float* b, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (av / bv);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_max_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_max_f32(const float* a, const float* b, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av > bv)) ? (av) : (bv));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_min_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_min_f32(const float* a, const float* b, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av < bv)) ? (av) : (bv));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_eq_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_eq_f32(const float* a, const float* b, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av == bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_ne_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_ne_f32(const float* a, const float* b, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av != bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_lt_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_lt_f32(const float* a, const float* b, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av < bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_le_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_le_f32(const float* a, const float* b, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av <= bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_gt_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_gt_f32(const float* a, const float* b, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av > bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_ge_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_ge_f32(const float* a, const float* b, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av >= bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_relu_bwd_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_relu_bwd_f32(const float* a, const float* b, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (av * (((bv > __uint_as_float(0x00000000u))) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u))));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_sigmoid_bwd_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_sigmoid_bwd_f32(const float* a, const float* b, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (av * (bv * (__uint_as_float(0x3f800000u) - bv)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_tanh_bwd_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_pointwise_tanh_bwd_f32(const float* a, const float* b, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (av * (__uint_as_float(0x3f800000u) - (bv * bv)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_neg_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_neg_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (-x);
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_abs_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_abs_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = abs(x);
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_sqrt_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_sqrt_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = sqrt(x);
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_rsqrt_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_rsqrt_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (__uint_as_float(0x3f800000u) / sqrt(x));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_reciprocal_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_reciprocal_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (__uint_as_float(0x3f800000u) / x);
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_exp_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_exp_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = exp(x);
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_log_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_log_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = log(x);
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_sin_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_sin_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = sin(x);
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cos_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_cos_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = cos(x);
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_tan_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_tan_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (sin(x) / cos(x));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_floor_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_floor_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = floor(x);
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_ceil_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_ceil_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = ceil(x);
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_round_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_round_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = round(x);
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_relu_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_relu_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (((x > __uint_as_float(0x00000000u))) ? (x) : (__uint_as_float(0x00000000u)));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_leaky_relu_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_leaky_relu_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (((x > __uint_as_float(0x00000000u))) ? (x) : ((__uint_as_float(0x3c23d70au) * x)));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_elu_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_elu_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (((x > __uint_as_float(0x00000000u))) ? (x) : ((exp(x) - __uint_as_float(0x3f800000u))));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_tanh_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_tanh_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = tanh(x);
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_sigmoid_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_sigmoid_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (__uint_as_float(0x3f800000u) / (__uint_as_float(0x3f800000u) + exp((-x))));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_softplus_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_softplus_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = log((__uint_as_float(0x3f800000u) + exp(x)));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_swish_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_swish_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (x * (__uint_as_float(0x3f800000u) / (__uint_as_float(0x3f800000u) + exp((-x)))));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_gelu_tanh_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_gelu_tanh_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = ((__uint_as_float(0x3f000000u) * x) * (__uint_as_float(0x3f800000u) + tanh((__uint_as_float(0x3f4c422au) * (x + (__uint_as_float(0x3d372713u) * (x * (x * x))))))));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_add_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_add_f16(const __half* a, const __half* b, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half aRaw = a[gid];
    __half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av + bv);
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_sub_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_sub_f16(const __half* a, const __half* b, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half aRaw = a[gid];
    __half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av - bv);
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_mul_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_mul_f16(const __half* a, const __half* b, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half aRaw = a[gid];
    __half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av * bv);
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_div_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_div_f16(const __half* a, const __half* b, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half aRaw = a[gid];
    __half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av / bv);
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_max_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_max_f16(const __half* a, const __half* b, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half aRaw = a[gid];
    __half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av > bv)) ? (av) : (bv));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_min_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_min_f16(const __half* a, const __half* b, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half aRaw = a[gid];
    __half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av < bv)) ? (av) : (bv));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_eq_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_eq_f16(const __half* a, const __half* b, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half aRaw = a[gid];
    __half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av == bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_ne_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_ne_f16(const __half* a, const __half* b, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half aRaw = a[gid];
    __half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av != bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_lt_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_lt_f16(const __half* a, const __half* b, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half aRaw = a[gid];
    __half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av < bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_le_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_le_f16(const __half* a, const __half* b, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half aRaw = a[gid];
    __half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av <= bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_gt_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_gt_f16(const __half* a, const __half* b, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half aRaw = a[gid];
    __half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av > bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_ge_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_ge_f16(const __half* a, const __half* b, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half aRaw = a[gid];
    __half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av >= bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_relu_bwd_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_relu_bwd_f16(const __half* a, const __half* b, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half aRaw = a[gid];
    __half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av * (((bv > __uint_as_float(0x00000000u))) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u))));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_sigmoid_bwd_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_sigmoid_bwd_f16(const __half* a, const __half* b, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half aRaw = a[gid];
    __half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av * (bv * (__uint_as_float(0x3f800000u) - bv)));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_tanh_bwd_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_ml_pointwise_tanh_bwd_f16(const __half* a, const __half* b, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __half aRaw = a[gid];
    __half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av * (__uint_as_float(0x3f800000u) - (bv * bv)));
    __half y = ((__half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_neg_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_neg_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (-x);
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_abs_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_abs_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = abs(x);
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_sqrt_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_sqrt_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = sqrt(x);
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_rsqrt_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_rsqrt_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (__uint_as_float(0x3f800000u) / sqrt(x));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_reciprocal_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_reciprocal_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (__uint_as_float(0x3f800000u) / x);
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_exp_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_exp_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = exp(x);
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_log_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_log_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = log(x);
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_sin_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_sin_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = sin(x);
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cos_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_cos_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = cos(x);
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_tan_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_tan_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (sin(x) / cos(x));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_floor_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_floor_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = floor(x);
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_ceil_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_ceil_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = ceil(x);
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_round_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_round_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = round(x);
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_relu_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_relu_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (((x > __uint_as_float(0x00000000u))) ? (x) : (__uint_as_float(0x00000000u)));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_leaky_relu_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_leaky_relu_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (((x > __uint_as_float(0x00000000u))) ? (x) : ((__uint_as_float(0x3c23d70au) * x)));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_elu_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_elu_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (((x > __uint_as_float(0x00000000u))) ? (x) : ((exp(x) - __uint_as_float(0x3f800000u))));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_tanh_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_tanh_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = tanh(x);
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_sigmoid_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_sigmoid_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (__uint_as_float(0x3f800000u) / (__uint_as_float(0x3f800000u) + exp((-x))));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_softplus_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_softplus_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = log((__uint_as_float(0x3f800000u) + exp(x)));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_swish_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_swish_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (x * (__uint_as_float(0x3f800000u) / (__uint_as_float(0x3f800000u) + exp((-x)))));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_gelu_tanh_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_gelu_tanh_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = ((__uint_as_float(0x3f000000u) * x) * (__uint_as_float(0x3f800000u) + tanh((__uint_as_float(0x3f4c422au) * (x + (__uint_as_float(0x3d372713u) * (x * (x * x))))))));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_add_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_add_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 aRaw = a[gid];
    __hip_bfloat16 bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av + bv);
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_sub_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_sub_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 aRaw = a[gid];
    __hip_bfloat16 bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av - bv);
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_mul_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_mul_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 aRaw = a[gid];
    __hip_bfloat16 bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av * bv);
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_div_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_div_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 aRaw = a[gid];
    __hip_bfloat16 bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av / bv);
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_max_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_max_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 aRaw = a[gid];
    __hip_bfloat16 bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av > bv)) ? (av) : (bv));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_min_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_min_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 aRaw = a[gid];
    __hip_bfloat16 bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av < bv)) ? (av) : (bv));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_eq_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_eq_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 aRaw = a[gid];
    __hip_bfloat16 bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av == bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_ne_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_ne_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 aRaw = a[gid];
    __hip_bfloat16 bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av != bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_lt_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_lt_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 aRaw = a[gid];
    __hip_bfloat16 bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av < bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_le_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_le_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 aRaw = a[gid];
    __hip_bfloat16 bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av <= bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_gt_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_gt_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 aRaw = a[gid];
    __hip_bfloat16 bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av > bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_cmp_ge_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_cmp_ge_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 aRaw = a[gid];
    __hip_bfloat16 bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av >= bv)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_relu_bwd_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_relu_bwd_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 aRaw = a[gid];
    __hip_bfloat16 bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av * (((bv > __uint_as_float(0x00000000u))) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u))));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_sigmoid_bwd_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_sigmoid_bwd_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 aRaw = a[gid];
    __hip_bfloat16 bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av * (bv * (__uint_as_float(0x3f800000u) - bv)));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_hip_navatala_ml_pointwise_tanh_bwd_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_pointwise_tanh_bwd_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if ((gid < countVal)) {
    __hip_bfloat16 aRaw = a[gid];
    __hip_bfloat16 bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av * (__uint_as_float(0x3f800000u) - (bv * bv)));
    __hip_bfloat16 y = ((__hip_bfloat16)(yF));
    _output[gid] = y;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_neg_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_neg_f32 = {
  1,
  "navatala_ml_pointwise_neg_f32",
  "hip",
  "navatala_ml_pointwise_neg_f32",
  "kernel:hip:navatala_ml_pointwise_neg_f32",
  "abi-r1:hip:navatala_ml_pointwise_neg_f32",
  "abi-r1:hip:navatala_ml_pointwise_neg_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_neg_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_abs_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_abs_f32 = {
  1,
  "navatala_ml_pointwise_abs_f32",
  "hip",
  "navatala_ml_pointwise_abs_f32",
  "kernel:hip:navatala_ml_pointwise_abs_f32",
  "abi-r1:hip:navatala_ml_pointwise_abs_f32",
  "abi-r1:hip:navatala_ml_pointwise_abs_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_abs_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_sqrt_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_sqrt_f32 = {
  1,
  "navatala_ml_pointwise_sqrt_f32",
  "hip",
  "navatala_ml_pointwise_sqrt_f32",
  "kernel:hip:navatala_ml_pointwise_sqrt_f32",
  "abi-r1:hip:navatala_ml_pointwise_sqrt_f32",
  "abi-r1:hip:navatala_ml_pointwise_sqrt_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_sqrt_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_rsqrt_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_rsqrt_f32 = {
  1,
  "navatala_ml_pointwise_rsqrt_f32",
  "hip",
  "navatala_ml_pointwise_rsqrt_f32",
  "kernel:hip:navatala_ml_pointwise_rsqrt_f32",
  "abi-r1:hip:navatala_ml_pointwise_rsqrt_f32",
  "abi-r1:hip:navatala_ml_pointwise_rsqrt_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_rsqrt_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_reciprocal_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_reciprocal_f32 = {
  1,
  "navatala_ml_pointwise_reciprocal_f32",
  "hip",
  "navatala_ml_pointwise_reciprocal_f32",
  "kernel:hip:navatala_ml_pointwise_reciprocal_f32",
  "abi-r1:hip:navatala_ml_pointwise_reciprocal_f32",
  "abi-r1:hip:navatala_ml_pointwise_reciprocal_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_reciprocal_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_exp_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_exp_f32 = {
  1,
  "navatala_ml_pointwise_exp_f32",
  "hip",
  "navatala_ml_pointwise_exp_f32",
  "kernel:hip:navatala_ml_pointwise_exp_f32",
  "abi-r1:hip:navatala_ml_pointwise_exp_f32",
  "abi-r1:hip:navatala_ml_pointwise_exp_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_exp_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_log_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_log_f32 = {
  1,
  "navatala_ml_pointwise_log_f32",
  "hip",
  "navatala_ml_pointwise_log_f32",
  "kernel:hip:navatala_ml_pointwise_log_f32",
  "abi-r1:hip:navatala_ml_pointwise_log_f32",
  "abi-r1:hip:navatala_ml_pointwise_log_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_log_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_sin_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_sin_f32 = {
  1,
  "navatala_ml_pointwise_sin_f32",
  "hip",
  "navatala_ml_pointwise_sin_f32",
  "kernel:hip:navatala_ml_pointwise_sin_f32",
  "abi-r1:hip:navatala_ml_pointwise_sin_f32",
  "abi-r1:hip:navatala_ml_pointwise_sin_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_sin_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cos_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cos_f32 = {
  1,
  "navatala_ml_pointwise_cos_f32",
  "hip",
  "navatala_ml_pointwise_cos_f32",
  "kernel:hip:navatala_ml_pointwise_cos_f32",
  "abi-r1:hip:navatala_ml_pointwise_cos_f32",
  "abi-r1:hip:navatala_ml_pointwise_cos_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_cos_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_tan_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_tan_f32 = {
  1,
  "navatala_ml_pointwise_tan_f32",
  "hip",
  "navatala_ml_pointwise_tan_f32",
  "kernel:hip:navatala_ml_pointwise_tan_f32",
  "abi-r1:hip:navatala_ml_pointwise_tan_f32",
  "abi-r1:hip:navatala_ml_pointwise_tan_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_tan_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_floor_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_floor_f32 = {
  1,
  "navatala_ml_pointwise_floor_f32",
  "hip",
  "navatala_ml_pointwise_floor_f32",
  "kernel:hip:navatala_ml_pointwise_floor_f32",
  "abi-r1:hip:navatala_ml_pointwise_floor_f32",
  "abi-r1:hip:navatala_ml_pointwise_floor_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_floor_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_ceil_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_ceil_f32 = {
  1,
  "navatala_ml_pointwise_ceil_f32",
  "hip",
  "navatala_ml_pointwise_ceil_f32",
  "kernel:hip:navatala_ml_pointwise_ceil_f32",
  "abi-r1:hip:navatala_ml_pointwise_ceil_f32",
  "abi-r1:hip:navatala_ml_pointwise_ceil_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_ceil_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_round_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_round_f32 = {
  1,
  "navatala_ml_pointwise_round_f32",
  "hip",
  "navatala_ml_pointwise_round_f32",
  "kernel:hip:navatala_ml_pointwise_round_f32",
  "abi-r1:hip:navatala_ml_pointwise_round_f32",
  "abi-r1:hip:navatala_ml_pointwise_round_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_round_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_relu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_relu_f32 = {
  1,
  "navatala_ml_pointwise_relu_f32",
  "hip",
  "navatala_ml_pointwise_relu_f32",
  "kernel:hip:navatala_ml_pointwise_relu_f32",
  "abi-r1:hip:navatala_ml_pointwise_relu_f32",
  "abi-r1:hip:navatala_ml_pointwise_relu_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_relu_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_leaky_relu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_leaky_relu_f32 = {
  1,
  "navatala_ml_pointwise_leaky_relu_f32",
  "hip",
  "navatala_ml_pointwise_leaky_relu_f32",
  "kernel:hip:navatala_ml_pointwise_leaky_relu_f32",
  "abi-r1:hip:navatala_ml_pointwise_leaky_relu_f32",
  "abi-r1:hip:navatala_ml_pointwise_leaky_relu_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_leaky_relu_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_elu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_elu_f32 = {
  1,
  "navatala_ml_pointwise_elu_f32",
  "hip",
  "navatala_ml_pointwise_elu_f32",
  "kernel:hip:navatala_ml_pointwise_elu_f32",
  "abi-r1:hip:navatala_ml_pointwise_elu_f32",
  "abi-r1:hip:navatala_ml_pointwise_elu_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_elu_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_tanh_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_tanh_f32 = {
  1,
  "navatala_ml_pointwise_tanh_f32",
  "hip",
  "navatala_ml_pointwise_tanh_f32",
  "kernel:hip:navatala_ml_pointwise_tanh_f32",
  "abi-r1:hip:navatala_ml_pointwise_tanh_f32",
  "abi-r1:hip:navatala_ml_pointwise_tanh_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_tanh_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_sigmoid_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_sigmoid_f32 = {
  1,
  "navatala_ml_pointwise_sigmoid_f32",
  "hip",
  "navatala_ml_pointwise_sigmoid_f32",
  "kernel:hip:navatala_ml_pointwise_sigmoid_f32",
  "abi-r1:hip:navatala_ml_pointwise_sigmoid_f32",
  "abi-r1:hip:navatala_ml_pointwise_sigmoid_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_sigmoid_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_softplus_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_softplus_f32 = {
  1,
  "navatala_ml_pointwise_softplus_f32",
  "hip",
  "navatala_ml_pointwise_softplus_f32",
  "kernel:hip:navatala_ml_pointwise_softplus_f32",
  "abi-r1:hip:navatala_ml_pointwise_softplus_f32",
  "abi-r1:hip:navatala_ml_pointwise_softplus_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_softplus_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_swish_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_swish_f32 = {
  1,
  "navatala_ml_pointwise_swish_f32",
  "hip",
  "navatala_ml_pointwise_swish_f32",
  "kernel:hip:navatala_ml_pointwise_swish_f32",
  "abi-r1:hip:navatala_ml_pointwise_swish_f32",
  "abi-r1:hip:navatala_ml_pointwise_swish_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_swish_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_gelu_tanh_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_gelu_tanh_f32 = {
  1,
  "navatala_ml_pointwise_gelu_tanh_f32",
  "hip",
  "navatala_ml_pointwise_gelu_tanh_f32",
  "kernel:hip:navatala_ml_pointwise_gelu_tanh_f32",
  "abi-r1:hip:navatala_ml_pointwise_gelu_tanh_f32",
  "abi-r1:hip:navatala_ml_pointwise_gelu_tanh_f32",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_gelu_tanh_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_add_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_add_f32 = {
  1,
  "navatala_ml_pointwise_add_f32",
  "hip",
  "navatala_ml_pointwise_add_f32",
  "kernel:hip:navatala_ml_pointwise_add_f32",
  "abi-r1:hip:navatala_ml_pointwise_add_f32",
  "abi-r1:hip:navatala_ml_pointwise_add_f32",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_add_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_sub_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_sub_f32 = {
  1,
  "navatala_ml_pointwise_sub_f32",
  "hip",
  "navatala_ml_pointwise_sub_f32",
  "kernel:hip:navatala_ml_pointwise_sub_f32",
  "abi-r1:hip:navatala_ml_pointwise_sub_f32",
  "abi-r1:hip:navatala_ml_pointwise_sub_f32",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_sub_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_mul_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_mul_f32 = {
  1,
  "navatala_ml_pointwise_mul_f32",
  "hip",
  "navatala_ml_pointwise_mul_f32",
  "kernel:hip:navatala_ml_pointwise_mul_f32",
  "abi-r1:hip:navatala_ml_pointwise_mul_f32",
  "abi-r1:hip:navatala_ml_pointwise_mul_f32",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_mul_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_div_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_div_f32 = {
  1,
  "navatala_ml_pointwise_div_f32",
  "hip",
  "navatala_ml_pointwise_div_f32",
  "kernel:hip:navatala_ml_pointwise_div_f32",
  "abi-r1:hip:navatala_ml_pointwise_div_f32",
  "abi-r1:hip:navatala_ml_pointwise_div_f32",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_div_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_max_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_max_f32 = {
  1,
  "navatala_ml_pointwise_max_f32",
  "hip",
  "navatala_ml_pointwise_max_f32",
  "kernel:hip:navatala_ml_pointwise_max_f32",
  "abi-r1:hip:navatala_ml_pointwise_max_f32",
  "abi-r1:hip:navatala_ml_pointwise_max_f32",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_max_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_min_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_min_f32 = {
  1,
  "navatala_ml_pointwise_min_f32",
  "hip",
  "navatala_ml_pointwise_min_f32",
  "kernel:hip:navatala_ml_pointwise_min_f32",
  "abi-r1:hip:navatala_ml_pointwise_min_f32",
  "abi-r1:hip:navatala_ml_pointwise_min_f32",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_min_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_eq_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_eq_f32 = {
  1,
  "navatala_ml_pointwise_cmp_eq_f32",
  "hip",
  "navatala_ml_pointwise_cmp_eq_f32",
  "kernel:hip:navatala_ml_pointwise_cmp_eq_f32",
  "abi-r1:hip:navatala_ml_pointwise_cmp_eq_f32",
  "abi-r1:hip:navatala_ml_pointwise_cmp_eq_f32",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_eq_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_ne_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_ne_f32 = {
  1,
  "navatala_ml_pointwise_cmp_ne_f32",
  "hip",
  "navatala_ml_pointwise_cmp_ne_f32",
  "kernel:hip:navatala_ml_pointwise_cmp_ne_f32",
  "abi-r1:hip:navatala_ml_pointwise_cmp_ne_f32",
  "abi-r1:hip:navatala_ml_pointwise_cmp_ne_f32",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_ne_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_lt_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_lt_f32 = {
  1,
  "navatala_ml_pointwise_cmp_lt_f32",
  "hip",
  "navatala_ml_pointwise_cmp_lt_f32",
  "kernel:hip:navatala_ml_pointwise_cmp_lt_f32",
  "abi-r1:hip:navatala_ml_pointwise_cmp_lt_f32",
  "abi-r1:hip:navatala_ml_pointwise_cmp_lt_f32",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_lt_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_le_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_le_f32 = {
  1,
  "navatala_ml_pointwise_cmp_le_f32",
  "hip",
  "navatala_ml_pointwise_cmp_le_f32",
  "kernel:hip:navatala_ml_pointwise_cmp_le_f32",
  "abi-r1:hip:navatala_ml_pointwise_cmp_le_f32",
  "abi-r1:hip:navatala_ml_pointwise_cmp_le_f32",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_le_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_gt_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_gt_f32 = {
  1,
  "navatala_ml_pointwise_cmp_gt_f32",
  "hip",
  "navatala_ml_pointwise_cmp_gt_f32",
  "kernel:hip:navatala_ml_pointwise_cmp_gt_f32",
  "abi-r1:hip:navatala_ml_pointwise_cmp_gt_f32",
  "abi-r1:hip:navatala_ml_pointwise_cmp_gt_f32",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_gt_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_ge_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_ge_f32 = {
  1,
  "navatala_ml_pointwise_cmp_ge_f32",
  "hip",
  "navatala_ml_pointwise_cmp_ge_f32",
  "kernel:hip:navatala_ml_pointwise_cmp_ge_f32",
  "abi-r1:hip:navatala_ml_pointwise_cmp_ge_f32",
  "abi-r1:hip:navatala_ml_pointwise_cmp_ge_f32",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_ge_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_relu_bwd_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_relu_bwd_f32 = {
  1,
  "navatala_ml_pointwise_relu_bwd_f32",
  "hip",
  "navatala_ml_pointwise_relu_bwd_f32",
  "kernel:hip:navatala_ml_pointwise_relu_bwd_f32",
  "abi-r1:hip:navatala_ml_pointwise_relu_bwd_f32",
  "abi-r1:hip:navatala_ml_pointwise_relu_bwd_f32",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_relu_bwd_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_sigmoid_bwd_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_sigmoid_bwd_f32 = {
  1,
  "navatala_ml_pointwise_sigmoid_bwd_f32",
  "hip",
  "navatala_ml_pointwise_sigmoid_bwd_f32",
  "kernel:hip:navatala_ml_pointwise_sigmoid_bwd_f32",
  "abi-r1:hip:navatala_ml_pointwise_sigmoid_bwd_f32",
  "abi-r1:hip:navatala_ml_pointwise_sigmoid_bwd_f32",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_sigmoid_bwd_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_tanh_bwd_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_tanh_bwd_f32 = {
  1,
  "navatala_ml_pointwise_tanh_bwd_f32",
  "hip",
  "navatala_ml_pointwise_tanh_bwd_f32",
  "kernel:hip:navatala_ml_pointwise_tanh_bwd_f32",
  "abi-r1:hip:navatala_ml_pointwise_tanh_bwd_f32",
  "abi-r1:hip:navatala_ml_pointwise_tanh_bwd_f32",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_tanh_bwd_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_neg_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_neg_f16 = {
  1,
  "navatala_ml_pointwise_neg_f16",
  "hip",
  "navatala_ml_pointwise_neg_f16",
  "kernel:hip:navatala_ml_pointwise_neg_f16",
  "abi-r1:hip:navatala_ml_pointwise_neg_f16",
  "abi-r1:hip:navatala_ml_pointwise_neg_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_neg_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_abs_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_abs_f16 = {
  1,
  "navatala_ml_pointwise_abs_f16",
  "hip",
  "navatala_ml_pointwise_abs_f16",
  "kernel:hip:navatala_ml_pointwise_abs_f16",
  "abi-r1:hip:navatala_ml_pointwise_abs_f16",
  "abi-r1:hip:navatala_ml_pointwise_abs_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_abs_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_sqrt_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_sqrt_f16 = {
  1,
  "navatala_ml_pointwise_sqrt_f16",
  "hip",
  "navatala_ml_pointwise_sqrt_f16",
  "kernel:hip:navatala_ml_pointwise_sqrt_f16",
  "abi-r1:hip:navatala_ml_pointwise_sqrt_f16",
  "abi-r1:hip:navatala_ml_pointwise_sqrt_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_sqrt_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_rsqrt_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_rsqrt_f16 = {
  1,
  "navatala_ml_pointwise_rsqrt_f16",
  "hip",
  "navatala_ml_pointwise_rsqrt_f16",
  "kernel:hip:navatala_ml_pointwise_rsqrt_f16",
  "abi-r1:hip:navatala_ml_pointwise_rsqrt_f16",
  "abi-r1:hip:navatala_ml_pointwise_rsqrt_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_rsqrt_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_reciprocal_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_reciprocal_f16 = {
  1,
  "navatala_ml_pointwise_reciprocal_f16",
  "hip",
  "navatala_ml_pointwise_reciprocal_f16",
  "kernel:hip:navatala_ml_pointwise_reciprocal_f16",
  "abi-r1:hip:navatala_ml_pointwise_reciprocal_f16",
  "abi-r1:hip:navatala_ml_pointwise_reciprocal_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_reciprocal_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_exp_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_exp_f16 = {
  1,
  "navatala_ml_pointwise_exp_f16",
  "hip",
  "navatala_ml_pointwise_exp_f16",
  "kernel:hip:navatala_ml_pointwise_exp_f16",
  "abi-r1:hip:navatala_ml_pointwise_exp_f16",
  "abi-r1:hip:navatala_ml_pointwise_exp_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_exp_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_log_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_log_f16 = {
  1,
  "navatala_ml_pointwise_log_f16",
  "hip",
  "navatala_ml_pointwise_log_f16",
  "kernel:hip:navatala_ml_pointwise_log_f16",
  "abi-r1:hip:navatala_ml_pointwise_log_f16",
  "abi-r1:hip:navatala_ml_pointwise_log_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_log_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_sin_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_sin_f16 = {
  1,
  "navatala_ml_pointwise_sin_f16",
  "hip",
  "navatala_ml_pointwise_sin_f16",
  "kernel:hip:navatala_ml_pointwise_sin_f16",
  "abi-r1:hip:navatala_ml_pointwise_sin_f16",
  "abi-r1:hip:navatala_ml_pointwise_sin_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_sin_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cos_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cos_f16 = {
  1,
  "navatala_ml_pointwise_cos_f16",
  "hip",
  "navatala_ml_pointwise_cos_f16",
  "kernel:hip:navatala_ml_pointwise_cos_f16",
  "abi-r1:hip:navatala_ml_pointwise_cos_f16",
  "abi-r1:hip:navatala_ml_pointwise_cos_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_cos_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_tan_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_tan_f16 = {
  1,
  "navatala_ml_pointwise_tan_f16",
  "hip",
  "navatala_ml_pointwise_tan_f16",
  "kernel:hip:navatala_ml_pointwise_tan_f16",
  "abi-r1:hip:navatala_ml_pointwise_tan_f16",
  "abi-r1:hip:navatala_ml_pointwise_tan_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_tan_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_floor_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_floor_f16 = {
  1,
  "navatala_ml_pointwise_floor_f16",
  "hip",
  "navatala_ml_pointwise_floor_f16",
  "kernel:hip:navatala_ml_pointwise_floor_f16",
  "abi-r1:hip:navatala_ml_pointwise_floor_f16",
  "abi-r1:hip:navatala_ml_pointwise_floor_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_floor_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_ceil_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_ceil_f16 = {
  1,
  "navatala_ml_pointwise_ceil_f16",
  "hip",
  "navatala_ml_pointwise_ceil_f16",
  "kernel:hip:navatala_ml_pointwise_ceil_f16",
  "abi-r1:hip:navatala_ml_pointwise_ceil_f16",
  "abi-r1:hip:navatala_ml_pointwise_ceil_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_ceil_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_round_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_round_f16 = {
  1,
  "navatala_ml_pointwise_round_f16",
  "hip",
  "navatala_ml_pointwise_round_f16",
  "kernel:hip:navatala_ml_pointwise_round_f16",
  "abi-r1:hip:navatala_ml_pointwise_round_f16",
  "abi-r1:hip:navatala_ml_pointwise_round_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_round_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_relu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_relu_f16 = {
  1,
  "navatala_ml_pointwise_relu_f16",
  "hip",
  "navatala_ml_pointwise_relu_f16",
  "kernel:hip:navatala_ml_pointwise_relu_f16",
  "abi-r1:hip:navatala_ml_pointwise_relu_f16",
  "abi-r1:hip:navatala_ml_pointwise_relu_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_relu_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_leaky_relu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_leaky_relu_f16 = {
  1,
  "navatala_ml_pointwise_leaky_relu_f16",
  "hip",
  "navatala_ml_pointwise_leaky_relu_f16",
  "kernel:hip:navatala_ml_pointwise_leaky_relu_f16",
  "abi-r1:hip:navatala_ml_pointwise_leaky_relu_f16",
  "abi-r1:hip:navatala_ml_pointwise_leaky_relu_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_leaky_relu_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_elu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_elu_f16 = {
  1,
  "navatala_ml_pointwise_elu_f16",
  "hip",
  "navatala_ml_pointwise_elu_f16",
  "kernel:hip:navatala_ml_pointwise_elu_f16",
  "abi-r1:hip:navatala_ml_pointwise_elu_f16",
  "abi-r1:hip:navatala_ml_pointwise_elu_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_elu_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_tanh_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_tanh_f16 = {
  1,
  "navatala_ml_pointwise_tanh_f16",
  "hip",
  "navatala_ml_pointwise_tanh_f16",
  "kernel:hip:navatala_ml_pointwise_tanh_f16",
  "abi-r1:hip:navatala_ml_pointwise_tanh_f16",
  "abi-r1:hip:navatala_ml_pointwise_tanh_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_tanh_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_sigmoid_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_sigmoid_f16 = {
  1,
  "navatala_ml_pointwise_sigmoid_f16",
  "hip",
  "navatala_ml_pointwise_sigmoid_f16",
  "kernel:hip:navatala_ml_pointwise_sigmoid_f16",
  "abi-r1:hip:navatala_ml_pointwise_sigmoid_f16",
  "abi-r1:hip:navatala_ml_pointwise_sigmoid_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_sigmoid_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_softplus_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_softplus_f16 = {
  1,
  "navatala_ml_pointwise_softplus_f16",
  "hip",
  "navatala_ml_pointwise_softplus_f16",
  "kernel:hip:navatala_ml_pointwise_softplus_f16",
  "abi-r1:hip:navatala_ml_pointwise_softplus_f16",
  "abi-r1:hip:navatala_ml_pointwise_softplus_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_softplus_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_swish_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_swish_f16 = {
  1,
  "navatala_ml_pointwise_swish_f16",
  "hip",
  "navatala_ml_pointwise_swish_f16",
  "kernel:hip:navatala_ml_pointwise_swish_f16",
  "abi-r1:hip:navatala_ml_pointwise_swish_f16",
  "abi-r1:hip:navatala_ml_pointwise_swish_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_swish_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_gelu_tanh_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_gelu_tanh_f16 = {
  1,
  "navatala_ml_pointwise_gelu_tanh_f16",
  "hip",
  "navatala_ml_pointwise_gelu_tanh_f16",
  "kernel:hip:navatala_ml_pointwise_gelu_tanh_f16",
  "abi-r1:hip:navatala_ml_pointwise_gelu_tanh_f16",
  "abi-r1:hip:navatala_ml_pointwise_gelu_tanh_f16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_gelu_tanh_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_add_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_add_f16 = {
  1,
  "navatala_ml_pointwise_add_f16",
  "hip",
  "navatala_ml_pointwise_add_f16",
  "kernel:hip:navatala_ml_pointwise_add_f16",
  "abi-r1:hip:navatala_ml_pointwise_add_f16",
  "abi-r1:hip:navatala_ml_pointwise_add_f16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_add_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_sub_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_sub_f16 = {
  1,
  "navatala_ml_pointwise_sub_f16",
  "hip",
  "navatala_ml_pointwise_sub_f16",
  "kernel:hip:navatala_ml_pointwise_sub_f16",
  "abi-r1:hip:navatala_ml_pointwise_sub_f16",
  "abi-r1:hip:navatala_ml_pointwise_sub_f16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_sub_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_mul_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_mul_f16 = {
  1,
  "navatala_ml_pointwise_mul_f16",
  "hip",
  "navatala_ml_pointwise_mul_f16",
  "kernel:hip:navatala_ml_pointwise_mul_f16",
  "abi-r1:hip:navatala_ml_pointwise_mul_f16",
  "abi-r1:hip:navatala_ml_pointwise_mul_f16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_mul_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_div_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_div_f16 = {
  1,
  "navatala_ml_pointwise_div_f16",
  "hip",
  "navatala_ml_pointwise_div_f16",
  "kernel:hip:navatala_ml_pointwise_div_f16",
  "abi-r1:hip:navatala_ml_pointwise_div_f16",
  "abi-r1:hip:navatala_ml_pointwise_div_f16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_div_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_max_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_max_f16 = {
  1,
  "navatala_ml_pointwise_max_f16",
  "hip",
  "navatala_ml_pointwise_max_f16",
  "kernel:hip:navatala_ml_pointwise_max_f16",
  "abi-r1:hip:navatala_ml_pointwise_max_f16",
  "abi-r1:hip:navatala_ml_pointwise_max_f16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_max_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_min_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_min_f16 = {
  1,
  "navatala_ml_pointwise_min_f16",
  "hip",
  "navatala_ml_pointwise_min_f16",
  "kernel:hip:navatala_ml_pointwise_min_f16",
  "abi-r1:hip:navatala_ml_pointwise_min_f16",
  "abi-r1:hip:navatala_ml_pointwise_min_f16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_min_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_eq_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_eq_f16 = {
  1,
  "navatala_ml_pointwise_cmp_eq_f16",
  "hip",
  "navatala_ml_pointwise_cmp_eq_f16",
  "kernel:hip:navatala_ml_pointwise_cmp_eq_f16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_eq_f16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_eq_f16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_eq_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_ne_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_ne_f16 = {
  1,
  "navatala_ml_pointwise_cmp_ne_f16",
  "hip",
  "navatala_ml_pointwise_cmp_ne_f16",
  "kernel:hip:navatala_ml_pointwise_cmp_ne_f16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_ne_f16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_ne_f16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_ne_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_lt_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_lt_f16 = {
  1,
  "navatala_ml_pointwise_cmp_lt_f16",
  "hip",
  "navatala_ml_pointwise_cmp_lt_f16",
  "kernel:hip:navatala_ml_pointwise_cmp_lt_f16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_lt_f16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_lt_f16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_lt_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_le_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_le_f16 = {
  1,
  "navatala_ml_pointwise_cmp_le_f16",
  "hip",
  "navatala_ml_pointwise_cmp_le_f16",
  "kernel:hip:navatala_ml_pointwise_cmp_le_f16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_le_f16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_le_f16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_le_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_gt_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_gt_f16 = {
  1,
  "navatala_ml_pointwise_cmp_gt_f16",
  "hip",
  "navatala_ml_pointwise_cmp_gt_f16",
  "kernel:hip:navatala_ml_pointwise_cmp_gt_f16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_gt_f16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_gt_f16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_gt_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_ge_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_ge_f16 = {
  1,
  "navatala_ml_pointwise_cmp_ge_f16",
  "hip",
  "navatala_ml_pointwise_cmp_ge_f16",
  "kernel:hip:navatala_ml_pointwise_cmp_ge_f16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_ge_f16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_ge_f16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_ge_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_relu_bwd_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_relu_bwd_f16 = {
  1,
  "navatala_ml_pointwise_relu_bwd_f16",
  "hip",
  "navatala_ml_pointwise_relu_bwd_f16",
  "kernel:hip:navatala_ml_pointwise_relu_bwd_f16",
  "abi-r1:hip:navatala_ml_pointwise_relu_bwd_f16",
  "abi-r1:hip:navatala_ml_pointwise_relu_bwd_f16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_relu_bwd_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_sigmoid_bwd_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_sigmoid_bwd_f16 = {
  1,
  "navatala_ml_pointwise_sigmoid_bwd_f16",
  "hip",
  "navatala_ml_pointwise_sigmoid_bwd_f16",
  "kernel:hip:navatala_ml_pointwise_sigmoid_bwd_f16",
  "abi-r1:hip:navatala_ml_pointwise_sigmoid_bwd_f16",
  "abi-r1:hip:navatala_ml_pointwise_sigmoid_bwd_f16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_sigmoid_bwd_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_tanh_bwd_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_tanh_bwd_f16 = {
  1,
  "navatala_ml_pointwise_tanh_bwd_f16",
  "hip",
  "navatala_ml_pointwise_tanh_bwd_f16",
  "kernel:hip:navatala_ml_pointwise_tanh_bwd_f16",
  "abi-r1:hip:navatala_ml_pointwise_tanh_bwd_f16",
  "abi-r1:hip:navatala_ml_pointwise_tanh_bwd_f16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_tanh_bwd_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_neg_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_neg_bf16 = {
  1,
  "navatala_ml_pointwise_neg_bf16",
  "hip",
  "navatala_ml_pointwise_neg_bf16",
  "kernel:hip:navatala_ml_pointwise_neg_bf16",
  "abi-r1:hip:navatala_ml_pointwise_neg_bf16",
  "abi-r1:hip:navatala_ml_pointwise_neg_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_neg_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_abs_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_abs_bf16 = {
  1,
  "navatala_ml_pointwise_abs_bf16",
  "hip",
  "navatala_ml_pointwise_abs_bf16",
  "kernel:hip:navatala_ml_pointwise_abs_bf16",
  "abi-r1:hip:navatala_ml_pointwise_abs_bf16",
  "abi-r1:hip:navatala_ml_pointwise_abs_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_abs_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_sqrt_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_sqrt_bf16 = {
  1,
  "navatala_ml_pointwise_sqrt_bf16",
  "hip",
  "navatala_ml_pointwise_sqrt_bf16",
  "kernel:hip:navatala_ml_pointwise_sqrt_bf16",
  "abi-r1:hip:navatala_ml_pointwise_sqrt_bf16",
  "abi-r1:hip:navatala_ml_pointwise_sqrt_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_sqrt_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_rsqrt_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_rsqrt_bf16 = {
  1,
  "navatala_ml_pointwise_rsqrt_bf16",
  "hip",
  "navatala_ml_pointwise_rsqrt_bf16",
  "kernel:hip:navatala_ml_pointwise_rsqrt_bf16",
  "abi-r1:hip:navatala_ml_pointwise_rsqrt_bf16",
  "abi-r1:hip:navatala_ml_pointwise_rsqrt_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_rsqrt_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_reciprocal_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_reciprocal_bf16 = {
  1,
  "navatala_ml_pointwise_reciprocal_bf16",
  "hip",
  "navatala_ml_pointwise_reciprocal_bf16",
  "kernel:hip:navatala_ml_pointwise_reciprocal_bf16",
  "abi-r1:hip:navatala_ml_pointwise_reciprocal_bf16",
  "abi-r1:hip:navatala_ml_pointwise_reciprocal_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_reciprocal_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_exp_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_exp_bf16 = {
  1,
  "navatala_ml_pointwise_exp_bf16",
  "hip",
  "navatala_ml_pointwise_exp_bf16",
  "kernel:hip:navatala_ml_pointwise_exp_bf16",
  "abi-r1:hip:navatala_ml_pointwise_exp_bf16",
  "abi-r1:hip:navatala_ml_pointwise_exp_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_exp_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_log_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_log_bf16 = {
  1,
  "navatala_ml_pointwise_log_bf16",
  "hip",
  "navatala_ml_pointwise_log_bf16",
  "kernel:hip:navatala_ml_pointwise_log_bf16",
  "abi-r1:hip:navatala_ml_pointwise_log_bf16",
  "abi-r1:hip:navatala_ml_pointwise_log_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_log_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_sin_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_sin_bf16 = {
  1,
  "navatala_ml_pointwise_sin_bf16",
  "hip",
  "navatala_ml_pointwise_sin_bf16",
  "kernel:hip:navatala_ml_pointwise_sin_bf16",
  "abi-r1:hip:navatala_ml_pointwise_sin_bf16",
  "abi-r1:hip:navatala_ml_pointwise_sin_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_sin_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cos_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cos_bf16 = {
  1,
  "navatala_ml_pointwise_cos_bf16",
  "hip",
  "navatala_ml_pointwise_cos_bf16",
  "kernel:hip:navatala_ml_pointwise_cos_bf16",
  "abi-r1:hip:navatala_ml_pointwise_cos_bf16",
  "abi-r1:hip:navatala_ml_pointwise_cos_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_cos_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_tan_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_tan_bf16 = {
  1,
  "navatala_ml_pointwise_tan_bf16",
  "hip",
  "navatala_ml_pointwise_tan_bf16",
  "kernel:hip:navatala_ml_pointwise_tan_bf16",
  "abi-r1:hip:navatala_ml_pointwise_tan_bf16",
  "abi-r1:hip:navatala_ml_pointwise_tan_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_tan_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_floor_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_floor_bf16 = {
  1,
  "navatala_ml_pointwise_floor_bf16",
  "hip",
  "navatala_ml_pointwise_floor_bf16",
  "kernel:hip:navatala_ml_pointwise_floor_bf16",
  "abi-r1:hip:navatala_ml_pointwise_floor_bf16",
  "abi-r1:hip:navatala_ml_pointwise_floor_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_floor_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_ceil_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_ceil_bf16 = {
  1,
  "navatala_ml_pointwise_ceil_bf16",
  "hip",
  "navatala_ml_pointwise_ceil_bf16",
  "kernel:hip:navatala_ml_pointwise_ceil_bf16",
  "abi-r1:hip:navatala_ml_pointwise_ceil_bf16",
  "abi-r1:hip:navatala_ml_pointwise_ceil_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_ceil_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_round_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_round_bf16 = {
  1,
  "navatala_ml_pointwise_round_bf16",
  "hip",
  "navatala_ml_pointwise_round_bf16",
  "kernel:hip:navatala_ml_pointwise_round_bf16",
  "abi-r1:hip:navatala_ml_pointwise_round_bf16",
  "abi-r1:hip:navatala_ml_pointwise_round_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_round_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_relu_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_relu_bf16 = {
  1,
  "navatala_ml_pointwise_relu_bf16",
  "hip",
  "navatala_ml_pointwise_relu_bf16",
  "kernel:hip:navatala_ml_pointwise_relu_bf16",
  "abi-r1:hip:navatala_ml_pointwise_relu_bf16",
  "abi-r1:hip:navatala_ml_pointwise_relu_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_relu_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_leaky_relu_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_leaky_relu_bf16 = {
  1,
  "navatala_ml_pointwise_leaky_relu_bf16",
  "hip",
  "navatala_ml_pointwise_leaky_relu_bf16",
  "kernel:hip:navatala_ml_pointwise_leaky_relu_bf16",
  "abi-r1:hip:navatala_ml_pointwise_leaky_relu_bf16",
  "abi-r1:hip:navatala_ml_pointwise_leaky_relu_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_leaky_relu_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_elu_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_elu_bf16 = {
  1,
  "navatala_ml_pointwise_elu_bf16",
  "hip",
  "navatala_ml_pointwise_elu_bf16",
  "kernel:hip:navatala_ml_pointwise_elu_bf16",
  "abi-r1:hip:navatala_ml_pointwise_elu_bf16",
  "abi-r1:hip:navatala_ml_pointwise_elu_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_elu_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_tanh_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_tanh_bf16 = {
  1,
  "navatala_ml_pointwise_tanh_bf16",
  "hip",
  "navatala_ml_pointwise_tanh_bf16",
  "kernel:hip:navatala_ml_pointwise_tanh_bf16",
  "abi-r1:hip:navatala_ml_pointwise_tanh_bf16",
  "abi-r1:hip:navatala_ml_pointwise_tanh_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_tanh_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_sigmoid_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_sigmoid_bf16 = {
  1,
  "navatala_ml_pointwise_sigmoid_bf16",
  "hip",
  "navatala_ml_pointwise_sigmoid_bf16",
  "kernel:hip:navatala_ml_pointwise_sigmoid_bf16",
  "abi-r1:hip:navatala_ml_pointwise_sigmoid_bf16",
  "abi-r1:hip:navatala_ml_pointwise_sigmoid_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_sigmoid_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_softplus_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_softplus_bf16 = {
  1,
  "navatala_ml_pointwise_softplus_bf16",
  "hip",
  "navatala_ml_pointwise_softplus_bf16",
  "kernel:hip:navatala_ml_pointwise_softplus_bf16",
  "abi-r1:hip:navatala_ml_pointwise_softplus_bf16",
  "abi-r1:hip:navatala_ml_pointwise_softplus_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_softplus_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_swish_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_swish_bf16 = {
  1,
  "navatala_ml_pointwise_swish_bf16",
  "hip",
  "navatala_ml_pointwise_swish_bf16",
  "kernel:hip:navatala_ml_pointwise_swish_bf16",
  "abi-r1:hip:navatala_ml_pointwise_swish_bf16",
  "abi-r1:hip:navatala_ml_pointwise_swish_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_swish_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_gelu_tanh_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_gelu_tanh_bf16 = {
  1,
  "navatala_ml_pointwise_gelu_tanh_bf16",
  "hip",
  "navatala_ml_pointwise_gelu_tanh_bf16",
  "kernel:hip:navatala_ml_pointwise_gelu_tanh_bf16",
  "abi-r1:hip:navatala_ml_pointwise_gelu_tanh_bf16",
  "abi-r1:hip:navatala_ml_pointwise_gelu_tanh_bf16",
  3,
  kAbiArgs_hip_navatala_ml_pointwise_gelu_tanh_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_add_bf16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_add_bf16 = {
  1,
  "navatala_ml_pointwise_add_bf16",
  "hip",
  "navatala_ml_pointwise_add_bf16",
  "kernel:hip:navatala_ml_pointwise_add_bf16",
  "abi-r1:hip:navatala_ml_pointwise_add_bf16",
  "abi-r1:hip:navatala_ml_pointwise_add_bf16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_add_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_sub_bf16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_sub_bf16 = {
  1,
  "navatala_ml_pointwise_sub_bf16",
  "hip",
  "navatala_ml_pointwise_sub_bf16",
  "kernel:hip:navatala_ml_pointwise_sub_bf16",
  "abi-r1:hip:navatala_ml_pointwise_sub_bf16",
  "abi-r1:hip:navatala_ml_pointwise_sub_bf16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_sub_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_mul_bf16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_mul_bf16 = {
  1,
  "navatala_ml_pointwise_mul_bf16",
  "hip",
  "navatala_ml_pointwise_mul_bf16",
  "kernel:hip:navatala_ml_pointwise_mul_bf16",
  "abi-r1:hip:navatala_ml_pointwise_mul_bf16",
  "abi-r1:hip:navatala_ml_pointwise_mul_bf16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_mul_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_div_bf16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_div_bf16 = {
  1,
  "navatala_ml_pointwise_div_bf16",
  "hip",
  "navatala_ml_pointwise_div_bf16",
  "kernel:hip:navatala_ml_pointwise_div_bf16",
  "abi-r1:hip:navatala_ml_pointwise_div_bf16",
  "abi-r1:hip:navatala_ml_pointwise_div_bf16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_div_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_max_bf16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_max_bf16 = {
  1,
  "navatala_ml_pointwise_max_bf16",
  "hip",
  "navatala_ml_pointwise_max_bf16",
  "kernel:hip:navatala_ml_pointwise_max_bf16",
  "abi-r1:hip:navatala_ml_pointwise_max_bf16",
  "abi-r1:hip:navatala_ml_pointwise_max_bf16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_max_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_min_bf16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_min_bf16 = {
  1,
  "navatala_ml_pointwise_min_bf16",
  "hip",
  "navatala_ml_pointwise_min_bf16",
  "kernel:hip:navatala_ml_pointwise_min_bf16",
  "abi-r1:hip:navatala_ml_pointwise_min_bf16",
  "abi-r1:hip:navatala_ml_pointwise_min_bf16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_min_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_eq_bf16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_eq_bf16 = {
  1,
  "navatala_ml_pointwise_cmp_eq_bf16",
  "hip",
  "navatala_ml_pointwise_cmp_eq_bf16",
  "kernel:hip:navatala_ml_pointwise_cmp_eq_bf16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_eq_bf16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_eq_bf16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_eq_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_ne_bf16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_ne_bf16 = {
  1,
  "navatala_ml_pointwise_cmp_ne_bf16",
  "hip",
  "navatala_ml_pointwise_cmp_ne_bf16",
  "kernel:hip:navatala_ml_pointwise_cmp_ne_bf16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_ne_bf16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_ne_bf16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_ne_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_lt_bf16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_lt_bf16 = {
  1,
  "navatala_ml_pointwise_cmp_lt_bf16",
  "hip",
  "navatala_ml_pointwise_cmp_lt_bf16",
  "kernel:hip:navatala_ml_pointwise_cmp_lt_bf16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_lt_bf16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_lt_bf16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_lt_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_le_bf16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_le_bf16 = {
  1,
  "navatala_ml_pointwise_cmp_le_bf16",
  "hip",
  "navatala_ml_pointwise_cmp_le_bf16",
  "kernel:hip:navatala_ml_pointwise_cmp_le_bf16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_le_bf16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_le_bf16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_le_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_gt_bf16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_gt_bf16 = {
  1,
  "navatala_ml_pointwise_cmp_gt_bf16",
  "hip",
  "navatala_ml_pointwise_cmp_gt_bf16",
  "kernel:hip:navatala_ml_pointwise_cmp_gt_bf16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_gt_bf16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_gt_bf16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_gt_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_cmp_ge_bf16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_cmp_ge_bf16 = {
  1,
  "navatala_ml_pointwise_cmp_ge_bf16",
  "hip",
  "navatala_ml_pointwise_cmp_ge_bf16",
  "kernel:hip:navatala_ml_pointwise_cmp_ge_bf16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_ge_bf16",
  "abi-r1:hip:navatala_ml_pointwise_cmp_ge_bf16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_cmp_ge_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_relu_bwd_bf16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_relu_bwd_bf16 = {
  1,
  "navatala_ml_pointwise_relu_bwd_bf16",
  "hip",
  "navatala_ml_pointwise_relu_bwd_bf16",
  "kernel:hip:navatala_ml_pointwise_relu_bwd_bf16",
  "abi-r1:hip:navatala_ml_pointwise_relu_bwd_bf16",
  "abi-r1:hip:navatala_ml_pointwise_relu_bwd_bf16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_relu_bwd_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_sigmoid_bwd_bf16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_sigmoid_bwd_bf16 = {
  1,
  "navatala_ml_pointwise_sigmoid_bwd_bf16",
  "hip",
  "navatala_ml_pointwise_sigmoid_bwd_bf16",
  "kernel:hip:navatala_ml_pointwise_sigmoid_bwd_bf16",
  "abi-r1:hip:navatala_ml_pointwise_sigmoid_bwd_bf16",
  "abi-r1:hip:navatala_ml_pointwise_sigmoid_bwd_bf16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_sigmoid_bwd_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_ml_pointwise_tanh_bwd_bf16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_ml_pointwise_tanh_bwd_bf16 = {
  1,
  "navatala_ml_pointwise_tanh_bwd_bf16",
  "hip",
  "navatala_ml_pointwise_tanh_bwd_bf16",
  "kernel:hip:navatala_ml_pointwise_tanh_bwd_bf16",
  "abi-r1:hip:navatala_ml_pointwise_tanh_bwd_bf16",
  "abi-r1:hip:navatala_ml_pointwise_tanh_bwd_bf16",
  4,
  kAbiArgs_hip_navatala_ml_pointwise_tanh_bwd_bf16
};

bool tryGetKernelAbiManifest_hip_ml_pointwise(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_neg_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_neg_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_abs_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_abs_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sqrt_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_sqrt_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_rsqrt_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_rsqrt_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_reciprocal_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_reciprocal_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_exp_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_exp_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_log_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_log_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sin_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_sin_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cos_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cos_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tan_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_tan_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_floor_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_floor_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_ceil_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_ceil_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_round_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_round_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_relu_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_relu_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_leaky_relu_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_leaky_relu_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_elu_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_elu_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tanh_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_tanh_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sigmoid_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_sigmoid_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_softplus_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_softplus_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_swish_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_swish_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_gelu_tanh_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_gelu_tanh_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_add_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_add_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sub_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_sub_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_mul_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_mul_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_div_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_div_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_max_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_max_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_min_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_min_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_eq_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_eq_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_ne_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_ne_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_lt_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_lt_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_le_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_le_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_gt_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_gt_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_ge_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_ge_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_relu_bwd_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_relu_bwd_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sigmoid_bwd_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_sigmoid_bwd_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tanh_bwd_f32") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_tanh_bwd_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_neg_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_neg_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_abs_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_abs_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sqrt_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_sqrt_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_rsqrt_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_rsqrt_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_reciprocal_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_reciprocal_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_exp_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_exp_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_log_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_log_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sin_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_sin_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cos_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cos_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tan_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_tan_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_floor_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_floor_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_ceil_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_ceil_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_round_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_round_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_relu_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_relu_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_leaky_relu_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_leaky_relu_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_elu_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_elu_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tanh_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_tanh_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sigmoid_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_sigmoid_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_softplus_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_softplus_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_swish_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_swish_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_gelu_tanh_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_gelu_tanh_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_add_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_add_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sub_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_sub_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_mul_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_mul_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_div_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_div_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_max_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_max_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_min_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_min_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_eq_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_eq_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_ne_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_ne_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_lt_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_lt_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_le_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_le_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_gt_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_gt_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_ge_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_ge_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_relu_bwd_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_relu_bwd_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sigmoid_bwd_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_sigmoid_bwd_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tanh_bwd_f16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_tanh_bwd_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_neg_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_neg_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_abs_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_abs_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sqrt_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_sqrt_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_rsqrt_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_rsqrt_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_reciprocal_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_reciprocal_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_exp_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_exp_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_log_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_log_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sin_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_sin_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cos_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cos_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tan_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_tan_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_floor_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_floor_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_ceil_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_ceil_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_round_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_round_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_relu_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_relu_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_leaky_relu_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_leaky_relu_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_elu_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_elu_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tanh_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_tanh_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sigmoid_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_sigmoid_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_softplus_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_softplus_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_swish_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_swish_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_gelu_tanh_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_gelu_tanh_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_add_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_add_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sub_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_sub_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_mul_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_mul_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_div_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_div_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_max_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_max_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_min_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_min_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_eq_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_eq_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_ne_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_ne_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_lt_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_lt_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_le_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_le_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_gt_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_gt_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_ge_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_cmp_ge_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_relu_bwd_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_relu_bwd_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sigmoid_bwd_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_sigmoid_bwd_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tanh_bwd_bf16") {
    out = &kAbiManifest_hip_navatala_ml_pointwise_tanh_bwd_bf16;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_hip_ml_pointwise(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_neg_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_neg_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_neg_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_abs_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_abs_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_abs_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sqrt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_sqrt_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_sqrt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_rsqrt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_rsqrt_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_rsqrt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_reciprocal_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_reciprocal_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_reciprocal_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_exp_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_exp_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_exp_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_log_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_log_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_log_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sin_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_sin_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_sin_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cos_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cos_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_cos_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tan_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_tan_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_tan_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_floor_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_floor_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_floor_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_ceil_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_ceil_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_ceil_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_round_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_round_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_round_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_relu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_relu_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_relu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_leaky_relu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_leaky_relu_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_leaky_relu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_elu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_elu_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_elu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tanh_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_tanh_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_tanh_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sigmoid_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_sigmoid_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_sigmoid_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_softplus_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_softplus_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_softplus_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_swish_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_swish_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_swish_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_gelu_tanh_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_gelu_tanh_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_gelu_tanh_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_add_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_add_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_add_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sub_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_sub_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_sub_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_mul_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_mul_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_mul_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_div_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_div_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_div_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_max_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_min_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_min_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_min_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_eq_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_eq_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_eq_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_ne_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_ne_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_ne_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_lt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_lt_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_lt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_le_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_le_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_le_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_gt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_gt_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_gt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_ge_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_ge_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_ge_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_relu_bwd_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_relu_bwd_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_relu_bwd_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sigmoid_bwd_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_sigmoid_bwd_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_sigmoid_bwd_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tanh_bwd_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_tanh_bwd_f32";
    std::string_view sv(k_hip_navatala_ml_pointwise_tanh_bwd_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_neg_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_neg_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_neg_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_abs_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_abs_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_abs_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sqrt_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_sqrt_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_sqrt_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_rsqrt_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_rsqrt_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_rsqrt_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_reciprocal_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_reciprocal_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_reciprocal_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_exp_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_exp_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_exp_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_log_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_log_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_log_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sin_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_sin_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_sin_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cos_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cos_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_cos_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tan_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_tan_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_tan_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_floor_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_floor_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_floor_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_ceil_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_ceil_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_ceil_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_round_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_round_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_round_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_relu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_relu_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_relu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_leaky_relu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_leaky_relu_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_leaky_relu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_elu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_elu_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_elu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tanh_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_tanh_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_tanh_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sigmoid_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_sigmoid_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_sigmoid_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_softplus_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_softplus_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_softplus_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_swish_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_swish_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_swish_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_gelu_tanh_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_gelu_tanh_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_gelu_tanh_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_add_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_add_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_add_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sub_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_sub_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_sub_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_mul_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_mul_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_mul_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_div_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_div_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_div_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_max_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_max_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_max_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_min_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_min_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_min_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_eq_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_eq_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_eq_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_ne_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_ne_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_ne_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_lt_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_lt_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_lt_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_le_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_le_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_le_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_gt_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_gt_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_gt_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_ge_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_ge_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_ge_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_relu_bwd_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_relu_bwd_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_relu_bwd_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sigmoid_bwd_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_sigmoid_bwd_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_sigmoid_bwd_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tanh_bwd_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_tanh_bwd_f16";
    std::string_view sv(k_hip_navatala_ml_pointwise_tanh_bwd_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_neg_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_neg_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_neg_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_abs_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_abs_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_abs_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sqrt_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_sqrt_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_sqrt_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_rsqrt_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_rsqrt_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_rsqrt_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_reciprocal_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_reciprocal_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_reciprocal_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_exp_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_exp_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_exp_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_log_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_log_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_log_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sin_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_sin_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_sin_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cos_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cos_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_cos_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tan_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_tan_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_tan_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_floor_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_floor_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_floor_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_ceil_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_ceil_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_ceil_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_round_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_round_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_round_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_relu_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_relu_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_relu_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_leaky_relu_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_leaky_relu_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_leaky_relu_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_elu_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_elu_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_elu_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tanh_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_tanh_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_tanh_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sigmoid_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_sigmoid_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_sigmoid_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_softplus_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_softplus_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_softplus_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_swish_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_swish_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_swish_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_gelu_tanh_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_gelu_tanh_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_gelu_tanh_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_add_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_add_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_add_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sub_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_sub_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_sub_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_mul_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_mul_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_mul_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_div_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_div_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_div_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_max_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_max_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_max_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_min_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_min_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_min_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_eq_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_eq_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_eq_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_ne_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_ne_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_ne_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_lt_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_lt_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_lt_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_le_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_le_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_le_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_gt_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_gt_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_gt_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_cmp_ge_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_cmp_ge_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_cmp_ge_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_relu_bwd_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_relu_bwd_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_relu_bwd_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_sigmoid_bwd_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_sigmoid_bwd_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_sigmoid_bwd_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_ml_pointwise_tanh_bwd_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_ml_pointwise_tanh_bwd_bf16";
    std::string_view sv(k_hip_navatala_ml_pointwise_tanh_bwd_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

