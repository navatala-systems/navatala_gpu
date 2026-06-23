// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `opencl_ml_pointwise` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_opencl_navatala_ml_pointwise_neg_f32 = R"kernel(
__kernel void navatala_ml_pointwise_neg_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (-x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_abs_f32 = R"kernel(
__kernel void navatala_ml_pointwise_abs_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = fabs(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_sqrt_f32 = R"kernel(
__kernel void navatala_ml_pointwise_sqrt_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = sqrt(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_rsqrt_f32 = R"kernel(
__kernel void navatala_ml_pointwise_rsqrt_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (as_float(0x3f800000u) / sqrt(x));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_reciprocal_f32 = R"kernel(
__kernel void navatala_ml_pointwise_reciprocal_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (as_float(0x3f800000u) / x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_exp_f32 = R"kernel(
__kernel void navatala_ml_pointwise_exp_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = exp(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_log_f32 = R"kernel(
__kernel void navatala_ml_pointwise_log_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = log(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_sin_f32 = R"kernel(
__kernel void navatala_ml_pointwise_sin_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = sin(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_cos_f32 = R"kernel(
__kernel void navatala_ml_pointwise_cos_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = cos(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_tan_f32 = R"kernel(
__kernel void navatala_ml_pointwise_tan_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (sin(x) / cos(x));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_floor_f32 = R"kernel(
__kernel void navatala_ml_pointwise_floor_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = floor(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_ceil_f32 = R"kernel(
__kernel void navatala_ml_pointwise_ceil_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = ceil(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_round_f32 = R"kernel(
__kernel void navatala_ml_pointwise_round_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = round(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_relu_f32 = R"kernel(
__kernel void navatala_ml_pointwise_relu_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (((x > as_float(0x00000000u))) ? (x) : (as_float(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_leaky_relu_f32 = R"kernel(
__kernel void navatala_ml_pointwise_leaky_relu_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (((x > as_float(0x00000000u))) ? (x) : ((as_float(0x3c23d70au) * x)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_elu_f32 = R"kernel(
__kernel void navatala_ml_pointwise_elu_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (((x > as_float(0x00000000u))) ? (x) : ((exp(x) - as_float(0x3f800000u))));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_tanh_f32 = R"kernel(
__kernel void navatala_ml_pointwise_tanh_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = tanh(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_sigmoid_f32 = R"kernel(
__kernel void navatala_ml_pointwise_sigmoid_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (as_float(0x3f800000u) / (as_float(0x3f800000u) + exp((-x))));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_softplus_f32 = R"kernel(
__kernel void navatala_ml_pointwise_softplus_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = log((as_float(0x3f800000u) + exp(x)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_swish_f32 = R"kernel(
__kernel void navatala_ml_pointwise_swish_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (x * (as_float(0x3f800000u) / (as_float(0x3f800000u) + exp((-x)))));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_gelu_tanh_f32 = R"kernel(
__kernel void navatala_ml_pointwise_gelu_tanh_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = ((as_float(0x3f000000u) * x) * (as_float(0x3f800000u) + tanh((as_float(0x3f4c422au) * (x + (as_float(0x3d372713u) * (x * (x * x))))))));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_add_f32 = R"kernel(
__kernel void navatala_ml_pointwise_add_f32(__global const float* a, __global const float* b, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
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
const char* k_opencl_navatala_ml_pointwise_sub_f32 = R"kernel(
__kernel void navatala_ml_pointwise_sub_f32(__global const float* a, __global const float* b, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
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
const char* k_opencl_navatala_ml_pointwise_mul_f32 = R"kernel(
__kernel void navatala_ml_pointwise_mul_f32(__global const float* a, __global const float* b, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
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
const char* k_opencl_navatala_ml_pointwise_div_f32 = R"kernel(
__kernel void navatala_ml_pointwise_div_f32(__global const float* a, __global const float* b, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
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
const char* k_opencl_navatala_ml_pointwise_max_f32 = R"kernel(
__kernel void navatala_ml_pointwise_max_f32(__global const float* a, __global const float* b, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
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
const char* k_opencl_navatala_ml_pointwise_min_f32 = R"kernel(
__kernel void navatala_ml_pointwise_min_f32(__global const float* a, __global const float* b, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
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
const char* k_opencl_navatala_ml_pointwise_cmp_eq_f32 = R"kernel(
__kernel void navatala_ml_pointwise_cmp_eq_f32(__global const float* a, __global const float* b, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av == bv)) ? (as_float(0x3f800000u)) : (as_float(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_cmp_ne_f32 = R"kernel(
__kernel void navatala_ml_pointwise_cmp_ne_f32(__global const float* a, __global const float* b, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av != bv)) ? (as_float(0x3f800000u)) : (as_float(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_cmp_lt_f32 = R"kernel(
__kernel void navatala_ml_pointwise_cmp_lt_f32(__global const float* a, __global const float* b, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av < bv)) ? (as_float(0x3f800000u)) : (as_float(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_cmp_le_f32 = R"kernel(
__kernel void navatala_ml_pointwise_cmp_le_f32(__global const float* a, __global const float* b, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av <= bv)) ? (as_float(0x3f800000u)) : (as_float(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_cmp_gt_f32 = R"kernel(
__kernel void navatala_ml_pointwise_cmp_gt_f32(__global const float* a, __global const float* b, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av > bv)) ? (as_float(0x3f800000u)) : (as_float(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_cmp_ge_f32 = R"kernel(
__kernel void navatala_ml_pointwise_cmp_ge_f32(__global const float* a, __global const float* b, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av >= bv)) ? (as_float(0x3f800000u)) : (as_float(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_relu_bwd_f32 = R"kernel(
__kernel void navatala_ml_pointwise_relu_bwd_f32(__global const float* a, __global const float* b, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (av * (((bv > as_float(0x00000000u))) ? (as_float(0x3f800000u)) : (as_float(0x00000000u))));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_sigmoid_bwd_f32 = R"kernel(
__kernel void navatala_ml_pointwise_sigmoid_bwd_f32(__global const float* a, __global const float* b, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (av * (bv * (as_float(0x3f800000u) - bv)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_tanh_bwd_f32 = R"kernel(
__kernel void navatala_ml_pointwise_tanh_bwd_f32(__global const float* a, __global const float* b, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (av * (as_float(0x3f800000u) - (bv * bv)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_neg_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_neg_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (-x);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_abs_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_abs_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = fabs(x);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_sqrt_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_sqrt_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = sqrt(x);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_rsqrt_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_rsqrt_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (as_float(0x3f800000u) / sqrt(x));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_reciprocal_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_reciprocal_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (as_float(0x3f800000u) / x);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_exp_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_exp_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = exp(x);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_log_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_log_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = log(x);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_sin_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_sin_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = sin(x);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_cos_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_cos_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = cos(x);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_tan_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_tan_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (sin(x) / cos(x));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_floor_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_floor_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = floor(x);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_ceil_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_ceil_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = ceil(x);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_round_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_round_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = round(x);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_relu_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_relu_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (((x > as_float(0x00000000u))) ? (x) : (as_float(0x00000000u)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_leaky_relu_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_leaky_relu_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (((x > as_float(0x00000000u))) ? (x) : ((as_float(0x3c23d70au) * x)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_elu_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_elu_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (((x > as_float(0x00000000u))) ? (x) : ((exp(x) - as_float(0x3f800000u))));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_tanh_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_tanh_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = tanh(x);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_sigmoid_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_sigmoid_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (as_float(0x3f800000u) / (as_float(0x3f800000u) + exp((-x))));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_softplus_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_softplus_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = log((as_float(0x3f800000u) + exp(x)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_swish_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_swish_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (x * (as_float(0x3f800000u) / (as_float(0x3f800000u) + exp((-x)))));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_gelu_tanh_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_gelu_tanh_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = ((as_float(0x3f000000u) * x) * (as_float(0x3f800000u) + tanh((as_float(0x3f4c422au) * (x + (as_float(0x3d372713u) * (x * (x * x))))))));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_add_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_add_f16(__global const half* a, __global const half* b, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av + bv);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_sub_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_sub_f16(__global const half* a, __global const half* b, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av - bv);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_mul_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_mul_f16(__global const half* a, __global const half* b, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av * bv);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_div_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_div_f16(__global const half* a, __global const half* b, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av / bv);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_max_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_max_f16(__global const half* a, __global const half* b, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av > bv)) ? (av) : (bv));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_min_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_min_f16(__global const half* a, __global const half* b, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av < bv)) ? (av) : (bv));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_cmp_eq_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_cmp_eq_f16(__global const half* a, __global const half* b, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av == bv)) ? (as_float(0x3f800000u)) : (as_float(0x00000000u)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_cmp_ne_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_cmp_ne_f16(__global const half* a, __global const half* b, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av != bv)) ? (as_float(0x3f800000u)) : (as_float(0x00000000u)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_cmp_lt_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_cmp_lt_f16(__global const half* a, __global const half* b, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av < bv)) ? (as_float(0x3f800000u)) : (as_float(0x00000000u)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_cmp_le_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_cmp_le_f16(__global const half* a, __global const half* b, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av <= bv)) ? (as_float(0x3f800000u)) : (as_float(0x00000000u)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_cmp_gt_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_cmp_gt_f16(__global const half* a, __global const half* b, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av > bv)) ? (as_float(0x3f800000u)) : (as_float(0x00000000u)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_cmp_ge_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_cmp_ge_f16(__global const half* a, __global const half* b, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av >= bv)) ? (as_float(0x3f800000u)) : (as_float(0x00000000u)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_relu_bwd_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_relu_bwd_f16(__global const half* a, __global const half* b, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av * (((bv > as_float(0x00000000u))) ? (as_float(0x3f800000u)) : (as_float(0x00000000u))));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_sigmoid_bwd_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_sigmoid_bwd_f16(__global const half* a, __global const half* b, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av * (bv * (as_float(0x3f800000u) - bv)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_opencl_navatala_ml_pointwise_tanh_bwd_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_pointwise_tanh_bwd_f16(__global const half* a, __global const half* b, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av * (as_float(0x3f800000u) - (bv * bv)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_neg_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_neg_f32 = {
  1,
  "navatala_ml_pointwise_neg_f32",
  "opencl",
  "navatala_ml_pointwise_neg_f32",
  "kernel:opencl:navatala_ml_pointwise_neg_f32",
  "abi-r1:opencl:navatala_ml_pointwise_neg_f32",
  "abi-r1:opencl:navatala_ml_pointwise_neg_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_neg_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_abs_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_abs_f32 = {
  1,
  "navatala_ml_pointwise_abs_f32",
  "opencl",
  "navatala_ml_pointwise_abs_f32",
  "kernel:opencl:navatala_ml_pointwise_abs_f32",
  "abi-r1:opencl:navatala_ml_pointwise_abs_f32",
  "abi-r1:opencl:navatala_ml_pointwise_abs_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_abs_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_sqrt_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_sqrt_f32 = {
  1,
  "navatala_ml_pointwise_sqrt_f32",
  "opencl",
  "navatala_ml_pointwise_sqrt_f32",
  "kernel:opencl:navatala_ml_pointwise_sqrt_f32",
  "abi-r1:opencl:navatala_ml_pointwise_sqrt_f32",
  "abi-r1:opencl:navatala_ml_pointwise_sqrt_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_sqrt_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_rsqrt_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_rsqrt_f32 = {
  1,
  "navatala_ml_pointwise_rsqrt_f32",
  "opencl",
  "navatala_ml_pointwise_rsqrt_f32",
  "kernel:opencl:navatala_ml_pointwise_rsqrt_f32",
  "abi-r1:opencl:navatala_ml_pointwise_rsqrt_f32",
  "abi-r1:opencl:navatala_ml_pointwise_rsqrt_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_rsqrt_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_reciprocal_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_reciprocal_f32 = {
  1,
  "navatala_ml_pointwise_reciprocal_f32",
  "opencl",
  "navatala_ml_pointwise_reciprocal_f32",
  "kernel:opencl:navatala_ml_pointwise_reciprocal_f32",
  "abi-r1:opencl:navatala_ml_pointwise_reciprocal_f32",
  "abi-r1:opencl:navatala_ml_pointwise_reciprocal_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_reciprocal_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_exp_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_exp_f32 = {
  1,
  "navatala_ml_pointwise_exp_f32",
  "opencl",
  "navatala_ml_pointwise_exp_f32",
  "kernel:opencl:navatala_ml_pointwise_exp_f32",
  "abi-r1:opencl:navatala_ml_pointwise_exp_f32",
  "abi-r1:opencl:navatala_ml_pointwise_exp_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_exp_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_log_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_log_f32 = {
  1,
  "navatala_ml_pointwise_log_f32",
  "opencl",
  "navatala_ml_pointwise_log_f32",
  "kernel:opencl:navatala_ml_pointwise_log_f32",
  "abi-r1:opencl:navatala_ml_pointwise_log_f32",
  "abi-r1:opencl:navatala_ml_pointwise_log_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_log_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_sin_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_sin_f32 = {
  1,
  "navatala_ml_pointwise_sin_f32",
  "opencl",
  "navatala_ml_pointwise_sin_f32",
  "kernel:opencl:navatala_ml_pointwise_sin_f32",
  "abi-r1:opencl:navatala_ml_pointwise_sin_f32",
  "abi-r1:opencl:navatala_ml_pointwise_sin_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_sin_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_cos_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_cos_f32 = {
  1,
  "navatala_ml_pointwise_cos_f32",
  "opencl",
  "navatala_ml_pointwise_cos_f32",
  "kernel:opencl:navatala_ml_pointwise_cos_f32",
  "abi-r1:opencl:navatala_ml_pointwise_cos_f32",
  "abi-r1:opencl:navatala_ml_pointwise_cos_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_cos_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_tan_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_tan_f32 = {
  1,
  "navatala_ml_pointwise_tan_f32",
  "opencl",
  "navatala_ml_pointwise_tan_f32",
  "kernel:opencl:navatala_ml_pointwise_tan_f32",
  "abi-r1:opencl:navatala_ml_pointwise_tan_f32",
  "abi-r1:opencl:navatala_ml_pointwise_tan_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_tan_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_floor_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_floor_f32 = {
  1,
  "navatala_ml_pointwise_floor_f32",
  "opencl",
  "navatala_ml_pointwise_floor_f32",
  "kernel:opencl:navatala_ml_pointwise_floor_f32",
  "abi-r1:opencl:navatala_ml_pointwise_floor_f32",
  "abi-r1:opencl:navatala_ml_pointwise_floor_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_floor_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_ceil_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_ceil_f32 = {
  1,
  "navatala_ml_pointwise_ceil_f32",
  "opencl",
  "navatala_ml_pointwise_ceil_f32",
  "kernel:opencl:navatala_ml_pointwise_ceil_f32",
  "abi-r1:opencl:navatala_ml_pointwise_ceil_f32",
  "abi-r1:opencl:navatala_ml_pointwise_ceil_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_ceil_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_round_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_round_f32 = {
  1,
  "navatala_ml_pointwise_round_f32",
  "opencl",
  "navatala_ml_pointwise_round_f32",
  "kernel:opencl:navatala_ml_pointwise_round_f32",
  "abi-r1:opencl:navatala_ml_pointwise_round_f32",
  "abi-r1:opencl:navatala_ml_pointwise_round_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_round_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_relu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_relu_f32 = {
  1,
  "navatala_ml_pointwise_relu_f32",
  "opencl",
  "navatala_ml_pointwise_relu_f32",
  "kernel:opencl:navatala_ml_pointwise_relu_f32",
  "abi-r1:opencl:navatala_ml_pointwise_relu_f32",
  "abi-r1:opencl:navatala_ml_pointwise_relu_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_relu_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_leaky_relu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_leaky_relu_f32 = {
  1,
  "navatala_ml_pointwise_leaky_relu_f32",
  "opencl",
  "navatala_ml_pointwise_leaky_relu_f32",
  "kernel:opencl:navatala_ml_pointwise_leaky_relu_f32",
  "abi-r1:opencl:navatala_ml_pointwise_leaky_relu_f32",
  "abi-r1:opencl:navatala_ml_pointwise_leaky_relu_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_leaky_relu_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_elu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_elu_f32 = {
  1,
  "navatala_ml_pointwise_elu_f32",
  "opencl",
  "navatala_ml_pointwise_elu_f32",
  "kernel:opencl:navatala_ml_pointwise_elu_f32",
  "abi-r1:opencl:navatala_ml_pointwise_elu_f32",
  "abi-r1:opencl:navatala_ml_pointwise_elu_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_elu_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_tanh_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_tanh_f32 = {
  1,
  "navatala_ml_pointwise_tanh_f32",
  "opencl",
  "navatala_ml_pointwise_tanh_f32",
  "kernel:opencl:navatala_ml_pointwise_tanh_f32",
  "abi-r1:opencl:navatala_ml_pointwise_tanh_f32",
  "abi-r1:opencl:navatala_ml_pointwise_tanh_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_tanh_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_sigmoid_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_sigmoid_f32 = {
  1,
  "navatala_ml_pointwise_sigmoid_f32",
  "opencl",
  "navatala_ml_pointwise_sigmoid_f32",
  "kernel:opencl:navatala_ml_pointwise_sigmoid_f32",
  "abi-r1:opencl:navatala_ml_pointwise_sigmoid_f32",
  "abi-r1:opencl:navatala_ml_pointwise_sigmoid_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_sigmoid_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_softplus_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_softplus_f32 = {
  1,
  "navatala_ml_pointwise_softplus_f32",
  "opencl",
  "navatala_ml_pointwise_softplus_f32",
  "kernel:opencl:navatala_ml_pointwise_softplus_f32",
  "abi-r1:opencl:navatala_ml_pointwise_softplus_f32",
  "abi-r1:opencl:navatala_ml_pointwise_softplus_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_softplus_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_swish_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_swish_f32 = {
  1,
  "navatala_ml_pointwise_swish_f32",
  "opencl",
  "navatala_ml_pointwise_swish_f32",
  "kernel:opencl:navatala_ml_pointwise_swish_f32",
  "abi-r1:opencl:navatala_ml_pointwise_swish_f32",
  "abi-r1:opencl:navatala_ml_pointwise_swish_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_swish_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_gelu_tanh_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_gelu_tanh_f32 = {
  1,
  "navatala_ml_pointwise_gelu_tanh_f32",
  "opencl",
  "navatala_ml_pointwise_gelu_tanh_f32",
  "kernel:opencl:navatala_ml_pointwise_gelu_tanh_f32",
  "abi-r1:opencl:navatala_ml_pointwise_gelu_tanh_f32",
  "abi-r1:opencl:navatala_ml_pointwise_gelu_tanh_f32",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_gelu_tanh_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_add_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_add_f32 = {
  1,
  "navatala_ml_pointwise_add_f32",
  "opencl",
  "navatala_ml_pointwise_add_f32",
  "kernel:opencl:navatala_ml_pointwise_add_f32",
  "abi-r1:opencl:navatala_ml_pointwise_add_f32",
  "abi-r1:opencl:navatala_ml_pointwise_add_f32",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_add_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_sub_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_sub_f32 = {
  1,
  "navatala_ml_pointwise_sub_f32",
  "opencl",
  "navatala_ml_pointwise_sub_f32",
  "kernel:opencl:navatala_ml_pointwise_sub_f32",
  "abi-r1:opencl:navatala_ml_pointwise_sub_f32",
  "abi-r1:opencl:navatala_ml_pointwise_sub_f32",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_sub_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_mul_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_mul_f32 = {
  1,
  "navatala_ml_pointwise_mul_f32",
  "opencl",
  "navatala_ml_pointwise_mul_f32",
  "kernel:opencl:navatala_ml_pointwise_mul_f32",
  "abi-r1:opencl:navatala_ml_pointwise_mul_f32",
  "abi-r1:opencl:navatala_ml_pointwise_mul_f32",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_mul_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_div_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_div_f32 = {
  1,
  "navatala_ml_pointwise_div_f32",
  "opencl",
  "navatala_ml_pointwise_div_f32",
  "kernel:opencl:navatala_ml_pointwise_div_f32",
  "abi-r1:opencl:navatala_ml_pointwise_div_f32",
  "abi-r1:opencl:navatala_ml_pointwise_div_f32",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_div_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_max_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_max_f32 = {
  1,
  "navatala_ml_pointwise_max_f32",
  "opencl",
  "navatala_ml_pointwise_max_f32",
  "kernel:opencl:navatala_ml_pointwise_max_f32",
  "abi-r1:opencl:navatala_ml_pointwise_max_f32",
  "abi-r1:opencl:navatala_ml_pointwise_max_f32",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_max_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_min_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_min_f32 = {
  1,
  "navatala_ml_pointwise_min_f32",
  "opencl",
  "navatala_ml_pointwise_min_f32",
  "kernel:opencl:navatala_ml_pointwise_min_f32",
  "abi-r1:opencl:navatala_ml_pointwise_min_f32",
  "abi-r1:opencl:navatala_ml_pointwise_min_f32",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_min_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_cmp_eq_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_cmp_eq_f32 = {
  1,
  "navatala_ml_pointwise_cmp_eq_f32",
  "opencl",
  "navatala_ml_pointwise_cmp_eq_f32",
  "kernel:opencl:navatala_ml_pointwise_cmp_eq_f32",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_eq_f32",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_eq_f32",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_cmp_eq_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_cmp_ne_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_cmp_ne_f32 = {
  1,
  "navatala_ml_pointwise_cmp_ne_f32",
  "opencl",
  "navatala_ml_pointwise_cmp_ne_f32",
  "kernel:opencl:navatala_ml_pointwise_cmp_ne_f32",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_ne_f32",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_ne_f32",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_cmp_ne_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_cmp_lt_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_cmp_lt_f32 = {
  1,
  "navatala_ml_pointwise_cmp_lt_f32",
  "opencl",
  "navatala_ml_pointwise_cmp_lt_f32",
  "kernel:opencl:navatala_ml_pointwise_cmp_lt_f32",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_lt_f32",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_lt_f32",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_cmp_lt_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_cmp_le_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_cmp_le_f32 = {
  1,
  "navatala_ml_pointwise_cmp_le_f32",
  "opencl",
  "navatala_ml_pointwise_cmp_le_f32",
  "kernel:opencl:navatala_ml_pointwise_cmp_le_f32",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_le_f32",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_le_f32",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_cmp_le_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_cmp_gt_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_cmp_gt_f32 = {
  1,
  "navatala_ml_pointwise_cmp_gt_f32",
  "opencl",
  "navatala_ml_pointwise_cmp_gt_f32",
  "kernel:opencl:navatala_ml_pointwise_cmp_gt_f32",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_gt_f32",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_gt_f32",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_cmp_gt_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_cmp_ge_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_cmp_ge_f32 = {
  1,
  "navatala_ml_pointwise_cmp_ge_f32",
  "opencl",
  "navatala_ml_pointwise_cmp_ge_f32",
  "kernel:opencl:navatala_ml_pointwise_cmp_ge_f32",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_ge_f32",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_ge_f32",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_cmp_ge_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_relu_bwd_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_relu_bwd_f32 = {
  1,
  "navatala_ml_pointwise_relu_bwd_f32",
  "opencl",
  "navatala_ml_pointwise_relu_bwd_f32",
  "kernel:opencl:navatala_ml_pointwise_relu_bwd_f32",
  "abi-r1:opencl:navatala_ml_pointwise_relu_bwd_f32",
  "abi-r1:opencl:navatala_ml_pointwise_relu_bwd_f32",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_relu_bwd_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_sigmoid_bwd_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_sigmoid_bwd_f32 = {
  1,
  "navatala_ml_pointwise_sigmoid_bwd_f32",
  "opencl",
  "navatala_ml_pointwise_sigmoid_bwd_f32",
  "kernel:opencl:navatala_ml_pointwise_sigmoid_bwd_f32",
  "abi-r1:opencl:navatala_ml_pointwise_sigmoid_bwd_f32",
  "abi-r1:opencl:navatala_ml_pointwise_sigmoid_bwd_f32",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_sigmoid_bwd_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_tanh_bwd_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_tanh_bwd_f32 = {
  1,
  "navatala_ml_pointwise_tanh_bwd_f32",
  "opencl",
  "navatala_ml_pointwise_tanh_bwd_f32",
  "kernel:opencl:navatala_ml_pointwise_tanh_bwd_f32",
  "abi-r1:opencl:navatala_ml_pointwise_tanh_bwd_f32",
  "abi-r1:opencl:navatala_ml_pointwise_tanh_bwd_f32",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_tanh_bwd_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_neg_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_neg_f16 = {
  1,
  "navatala_ml_pointwise_neg_f16",
  "opencl",
  "navatala_ml_pointwise_neg_f16",
  "kernel:opencl:navatala_ml_pointwise_neg_f16",
  "abi-r1:opencl:navatala_ml_pointwise_neg_f16",
  "abi-r1:opencl:navatala_ml_pointwise_neg_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_neg_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_abs_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_abs_f16 = {
  1,
  "navatala_ml_pointwise_abs_f16",
  "opencl",
  "navatala_ml_pointwise_abs_f16",
  "kernel:opencl:navatala_ml_pointwise_abs_f16",
  "abi-r1:opencl:navatala_ml_pointwise_abs_f16",
  "abi-r1:opencl:navatala_ml_pointwise_abs_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_abs_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_sqrt_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_sqrt_f16 = {
  1,
  "navatala_ml_pointwise_sqrt_f16",
  "opencl",
  "navatala_ml_pointwise_sqrt_f16",
  "kernel:opencl:navatala_ml_pointwise_sqrt_f16",
  "abi-r1:opencl:navatala_ml_pointwise_sqrt_f16",
  "abi-r1:opencl:navatala_ml_pointwise_sqrt_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_sqrt_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_rsqrt_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_rsqrt_f16 = {
  1,
  "navatala_ml_pointwise_rsqrt_f16",
  "opencl",
  "navatala_ml_pointwise_rsqrt_f16",
  "kernel:opencl:navatala_ml_pointwise_rsqrt_f16",
  "abi-r1:opencl:navatala_ml_pointwise_rsqrt_f16",
  "abi-r1:opencl:navatala_ml_pointwise_rsqrt_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_rsqrt_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_reciprocal_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_reciprocal_f16 = {
  1,
  "navatala_ml_pointwise_reciprocal_f16",
  "opencl",
  "navatala_ml_pointwise_reciprocal_f16",
  "kernel:opencl:navatala_ml_pointwise_reciprocal_f16",
  "abi-r1:opencl:navatala_ml_pointwise_reciprocal_f16",
  "abi-r1:opencl:navatala_ml_pointwise_reciprocal_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_reciprocal_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_exp_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_exp_f16 = {
  1,
  "navatala_ml_pointwise_exp_f16",
  "opencl",
  "navatala_ml_pointwise_exp_f16",
  "kernel:opencl:navatala_ml_pointwise_exp_f16",
  "abi-r1:opencl:navatala_ml_pointwise_exp_f16",
  "abi-r1:opencl:navatala_ml_pointwise_exp_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_exp_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_log_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_log_f16 = {
  1,
  "navatala_ml_pointwise_log_f16",
  "opencl",
  "navatala_ml_pointwise_log_f16",
  "kernel:opencl:navatala_ml_pointwise_log_f16",
  "abi-r1:opencl:navatala_ml_pointwise_log_f16",
  "abi-r1:opencl:navatala_ml_pointwise_log_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_log_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_sin_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_sin_f16 = {
  1,
  "navatala_ml_pointwise_sin_f16",
  "opencl",
  "navatala_ml_pointwise_sin_f16",
  "kernel:opencl:navatala_ml_pointwise_sin_f16",
  "abi-r1:opencl:navatala_ml_pointwise_sin_f16",
  "abi-r1:opencl:navatala_ml_pointwise_sin_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_sin_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_cos_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_cos_f16 = {
  1,
  "navatala_ml_pointwise_cos_f16",
  "opencl",
  "navatala_ml_pointwise_cos_f16",
  "kernel:opencl:navatala_ml_pointwise_cos_f16",
  "abi-r1:opencl:navatala_ml_pointwise_cos_f16",
  "abi-r1:opencl:navatala_ml_pointwise_cos_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_cos_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_tan_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_tan_f16 = {
  1,
  "navatala_ml_pointwise_tan_f16",
  "opencl",
  "navatala_ml_pointwise_tan_f16",
  "kernel:opencl:navatala_ml_pointwise_tan_f16",
  "abi-r1:opencl:navatala_ml_pointwise_tan_f16",
  "abi-r1:opencl:navatala_ml_pointwise_tan_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_tan_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_floor_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_floor_f16 = {
  1,
  "navatala_ml_pointwise_floor_f16",
  "opencl",
  "navatala_ml_pointwise_floor_f16",
  "kernel:opencl:navatala_ml_pointwise_floor_f16",
  "abi-r1:opencl:navatala_ml_pointwise_floor_f16",
  "abi-r1:opencl:navatala_ml_pointwise_floor_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_floor_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_ceil_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_ceil_f16 = {
  1,
  "navatala_ml_pointwise_ceil_f16",
  "opencl",
  "navatala_ml_pointwise_ceil_f16",
  "kernel:opencl:navatala_ml_pointwise_ceil_f16",
  "abi-r1:opencl:navatala_ml_pointwise_ceil_f16",
  "abi-r1:opencl:navatala_ml_pointwise_ceil_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_ceil_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_round_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_round_f16 = {
  1,
  "navatala_ml_pointwise_round_f16",
  "opencl",
  "navatala_ml_pointwise_round_f16",
  "kernel:opencl:navatala_ml_pointwise_round_f16",
  "abi-r1:opencl:navatala_ml_pointwise_round_f16",
  "abi-r1:opencl:navatala_ml_pointwise_round_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_round_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_relu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_relu_f16 = {
  1,
  "navatala_ml_pointwise_relu_f16",
  "opencl",
  "navatala_ml_pointwise_relu_f16",
  "kernel:opencl:navatala_ml_pointwise_relu_f16",
  "abi-r1:opencl:navatala_ml_pointwise_relu_f16",
  "abi-r1:opencl:navatala_ml_pointwise_relu_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_relu_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_leaky_relu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_leaky_relu_f16 = {
  1,
  "navatala_ml_pointwise_leaky_relu_f16",
  "opencl",
  "navatala_ml_pointwise_leaky_relu_f16",
  "kernel:opencl:navatala_ml_pointwise_leaky_relu_f16",
  "abi-r1:opencl:navatala_ml_pointwise_leaky_relu_f16",
  "abi-r1:opencl:navatala_ml_pointwise_leaky_relu_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_leaky_relu_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_elu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_elu_f16 = {
  1,
  "navatala_ml_pointwise_elu_f16",
  "opencl",
  "navatala_ml_pointwise_elu_f16",
  "kernel:opencl:navatala_ml_pointwise_elu_f16",
  "abi-r1:opencl:navatala_ml_pointwise_elu_f16",
  "abi-r1:opencl:navatala_ml_pointwise_elu_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_elu_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_tanh_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_tanh_f16 = {
  1,
  "navatala_ml_pointwise_tanh_f16",
  "opencl",
  "navatala_ml_pointwise_tanh_f16",
  "kernel:opencl:navatala_ml_pointwise_tanh_f16",
  "abi-r1:opencl:navatala_ml_pointwise_tanh_f16",
  "abi-r1:opencl:navatala_ml_pointwise_tanh_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_tanh_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_sigmoid_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_sigmoid_f16 = {
  1,
  "navatala_ml_pointwise_sigmoid_f16",
  "opencl",
  "navatala_ml_pointwise_sigmoid_f16",
  "kernel:opencl:navatala_ml_pointwise_sigmoid_f16",
  "abi-r1:opencl:navatala_ml_pointwise_sigmoid_f16",
  "abi-r1:opencl:navatala_ml_pointwise_sigmoid_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_sigmoid_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_softplus_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_softplus_f16 = {
  1,
  "navatala_ml_pointwise_softplus_f16",
  "opencl",
  "navatala_ml_pointwise_softplus_f16",
  "kernel:opencl:navatala_ml_pointwise_softplus_f16",
  "abi-r1:opencl:navatala_ml_pointwise_softplus_f16",
  "abi-r1:opencl:navatala_ml_pointwise_softplus_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_softplus_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_swish_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_swish_f16 = {
  1,
  "navatala_ml_pointwise_swish_f16",
  "opencl",
  "navatala_ml_pointwise_swish_f16",
  "kernel:opencl:navatala_ml_pointwise_swish_f16",
  "abi-r1:opencl:navatala_ml_pointwise_swish_f16",
  "abi-r1:opencl:navatala_ml_pointwise_swish_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_swish_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_gelu_tanh_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_gelu_tanh_f16 = {
  1,
  "navatala_ml_pointwise_gelu_tanh_f16",
  "opencl",
  "navatala_ml_pointwise_gelu_tanh_f16",
  "kernel:opencl:navatala_ml_pointwise_gelu_tanh_f16",
  "abi-r1:opencl:navatala_ml_pointwise_gelu_tanh_f16",
  "abi-r1:opencl:navatala_ml_pointwise_gelu_tanh_f16",
  3,
  kAbiArgs_opencl_navatala_ml_pointwise_gelu_tanh_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_add_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_add_f16 = {
  1,
  "navatala_ml_pointwise_add_f16",
  "opencl",
  "navatala_ml_pointwise_add_f16",
  "kernel:opencl:navatala_ml_pointwise_add_f16",
  "abi-r1:opencl:navatala_ml_pointwise_add_f16",
  "abi-r1:opencl:navatala_ml_pointwise_add_f16",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_add_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_sub_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_sub_f16 = {
  1,
  "navatala_ml_pointwise_sub_f16",
  "opencl",
  "navatala_ml_pointwise_sub_f16",
  "kernel:opencl:navatala_ml_pointwise_sub_f16",
  "abi-r1:opencl:navatala_ml_pointwise_sub_f16",
  "abi-r1:opencl:navatala_ml_pointwise_sub_f16",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_sub_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_mul_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_mul_f16 = {
  1,
  "navatala_ml_pointwise_mul_f16",
  "opencl",
  "navatala_ml_pointwise_mul_f16",
  "kernel:opencl:navatala_ml_pointwise_mul_f16",
  "abi-r1:opencl:navatala_ml_pointwise_mul_f16",
  "abi-r1:opencl:navatala_ml_pointwise_mul_f16",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_mul_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_div_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_div_f16 = {
  1,
  "navatala_ml_pointwise_div_f16",
  "opencl",
  "navatala_ml_pointwise_div_f16",
  "kernel:opencl:navatala_ml_pointwise_div_f16",
  "abi-r1:opencl:navatala_ml_pointwise_div_f16",
  "abi-r1:opencl:navatala_ml_pointwise_div_f16",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_div_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_max_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_max_f16 = {
  1,
  "navatala_ml_pointwise_max_f16",
  "opencl",
  "navatala_ml_pointwise_max_f16",
  "kernel:opencl:navatala_ml_pointwise_max_f16",
  "abi-r1:opencl:navatala_ml_pointwise_max_f16",
  "abi-r1:opencl:navatala_ml_pointwise_max_f16",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_max_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_min_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_min_f16 = {
  1,
  "navatala_ml_pointwise_min_f16",
  "opencl",
  "navatala_ml_pointwise_min_f16",
  "kernel:opencl:navatala_ml_pointwise_min_f16",
  "abi-r1:opencl:navatala_ml_pointwise_min_f16",
  "abi-r1:opencl:navatala_ml_pointwise_min_f16",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_min_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_cmp_eq_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_cmp_eq_f16 = {
  1,
  "navatala_ml_pointwise_cmp_eq_f16",
  "opencl",
  "navatala_ml_pointwise_cmp_eq_f16",
  "kernel:opencl:navatala_ml_pointwise_cmp_eq_f16",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_eq_f16",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_eq_f16",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_cmp_eq_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_cmp_ne_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_cmp_ne_f16 = {
  1,
  "navatala_ml_pointwise_cmp_ne_f16",
  "opencl",
  "navatala_ml_pointwise_cmp_ne_f16",
  "kernel:opencl:navatala_ml_pointwise_cmp_ne_f16",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_ne_f16",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_ne_f16",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_cmp_ne_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_cmp_lt_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_cmp_lt_f16 = {
  1,
  "navatala_ml_pointwise_cmp_lt_f16",
  "opencl",
  "navatala_ml_pointwise_cmp_lt_f16",
  "kernel:opencl:navatala_ml_pointwise_cmp_lt_f16",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_lt_f16",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_lt_f16",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_cmp_lt_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_cmp_le_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_cmp_le_f16 = {
  1,
  "navatala_ml_pointwise_cmp_le_f16",
  "opencl",
  "navatala_ml_pointwise_cmp_le_f16",
  "kernel:opencl:navatala_ml_pointwise_cmp_le_f16",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_le_f16",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_le_f16",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_cmp_le_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_cmp_gt_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_cmp_gt_f16 = {
  1,
  "navatala_ml_pointwise_cmp_gt_f16",
  "opencl",
  "navatala_ml_pointwise_cmp_gt_f16",
  "kernel:opencl:navatala_ml_pointwise_cmp_gt_f16",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_gt_f16",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_gt_f16",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_cmp_gt_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_cmp_ge_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_cmp_ge_f16 = {
  1,
  "navatala_ml_pointwise_cmp_ge_f16",
  "opencl",
  "navatala_ml_pointwise_cmp_ge_f16",
  "kernel:opencl:navatala_ml_pointwise_cmp_ge_f16",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_ge_f16",
  "abi-r1:opencl:navatala_ml_pointwise_cmp_ge_f16",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_cmp_ge_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_relu_bwd_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_relu_bwd_f16 = {
  1,
  "navatala_ml_pointwise_relu_bwd_f16",
  "opencl",
  "navatala_ml_pointwise_relu_bwd_f16",
  "kernel:opencl:navatala_ml_pointwise_relu_bwd_f16",
  "abi-r1:opencl:navatala_ml_pointwise_relu_bwd_f16",
  "abi-r1:opencl:navatala_ml_pointwise_relu_bwd_f16",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_relu_bwd_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_sigmoid_bwd_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_sigmoid_bwd_f16 = {
  1,
  "navatala_ml_pointwise_sigmoid_bwd_f16",
  "opencl",
  "navatala_ml_pointwise_sigmoid_bwd_f16",
  "kernel:opencl:navatala_ml_pointwise_sigmoid_bwd_f16",
  "abi-r1:opencl:navatala_ml_pointwise_sigmoid_bwd_f16",
  "abi-r1:opencl:navatala_ml_pointwise_sigmoid_bwd_f16",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_sigmoid_bwd_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_pointwise_tanh_bwd_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_pointwise_tanh_bwd_f16 = {
  1,
  "navatala_ml_pointwise_tanh_bwd_f16",
  "opencl",
  "navatala_ml_pointwise_tanh_bwd_f16",
  "kernel:opencl:navatala_ml_pointwise_tanh_bwd_f16",
  "abi-r1:opencl:navatala_ml_pointwise_tanh_bwd_f16",
  "abi-r1:opencl:navatala_ml_pointwise_tanh_bwd_f16",
  4,
  kAbiArgs_opencl_navatala_ml_pointwise_tanh_bwd_f16
};

bool tryGetKernelAbiManifest_opencl_ml_pointwise(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_neg_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_neg_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_abs_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_abs_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sqrt_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_sqrt_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_rsqrt_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_rsqrt_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_reciprocal_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_reciprocal_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_exp_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_exp_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_log_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_log_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sin_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_sin_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cos_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_cos_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_tan_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_tan_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_floor_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_floor_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_ceil_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_ceil_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_round_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_round_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_relu_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_relu_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_leaky_relu_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_leaky_relu_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_elu_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_elu_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_tanh_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_tanh_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sigmoid_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_sigmoid_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_softplus_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_softplus_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_swish_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_swish_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_gelu_tanh_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_gelu_tanh_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_add_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_add_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sub_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_sub_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_mul_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_mul_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_div_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_div_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_max_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_max_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_min_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_min_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_eq_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_cmp_eq_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_ne_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_cmp_ne_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_lt_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_cmp_lt_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_le_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_cmp_le_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_gt_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_cmp_gt_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_ge_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_cmp_ge_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_relu_bwd_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_relu_bwd_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sigmoid_bwd_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_sigmoid_bwd_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_tanh_bwd_f32") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_tanh_bwd_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_neg_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_neg_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_abs_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_abs_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sqrt_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_sqrt_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_rsqrt_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_rsqrt_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_reciprocal_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_reciprocal_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_exp_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_exp_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_log_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_log_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sin_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_sin_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cos_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_cos_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_tan_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_tan_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_floor_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_floor_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_ceil_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_ceil_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_round_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_round_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_relu_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_relu_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_leaky_relu_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_leaky_relu_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_elu_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_elu_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_tanh_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_tanh_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sigmoid_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_sigmoid_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_softplus_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_softplus_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_swish_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_swish_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_gelu_tanh_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_gelu_tanh_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_add_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_add_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sub_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_sub_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_mul_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_mul_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_div_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_div_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_max_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_max_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_min_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_min_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_eq_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_cmp_eq_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_ne_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_cmp_ne_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_lt_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_cmp_lt_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_le_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_cmp_le_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_gt_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_cmp_gt_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_ge_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_cmp_ge_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_relu_bwd_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_relu_bwd_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sigmoid_bwd_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_sigmoid_bwd_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_tanh_bwd_f16") {
    out = &kAbiManifest_opencl_navatala_ml_pointwise_tanh_bwd_f16;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_opencl_ml_pointwise(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_neg_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_neg_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_neg_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_abs_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_abs_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_abs_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sqrt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_sqrt_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_sqrt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_rsqrt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_rsqrt_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_rsqrt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_reciprocal_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_reciprocal_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_reciprocal_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_exp_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_exp_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_exp_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_log_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_log_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_log_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sin_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_sin_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_sin_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cos_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_cos_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_cos_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_tan_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_tan_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_tan_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_floor_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_floor_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_floor_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_ceil_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_ceil_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_ceil_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_round_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_round_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_round_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_relu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_relu_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_relu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_leaky_relu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_leaky_relu_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_leaky_relu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_elu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_elu_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_elu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_tanh_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_tanh_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_tanh_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sigmoid_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_sigmoid_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_sigmoid_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_softplus_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_softplus_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_softplus_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_swish_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_swish_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_swish_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_gelu_tanh_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_gelu_tanh_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_gelu_tanh_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_add_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_add_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_add_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sub_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_sub_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_sub_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_mul_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_mul_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_mul_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_div_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_div_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_div_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_max_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_min_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_min_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_min_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_eq_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_cmp_eq_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_cmp_eq_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_ne_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_cmp_ne_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_cmp_ne_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_lt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_cmp_lt_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_cmp_lt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_le_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_cmp_le_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_cmp_le_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_gt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_cmp_gt_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_cmp_gt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_ge_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_cmp_ge_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_cmp_ge_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_relu_bwd_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_relu_bwd_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_relu_bwd_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sigmoid_bwd_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_sigmoid_bwd_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_sigmoid_bwd_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_tanh_bwd_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_tanh_bwd_f32";
    std::string_view sv(k_opencl_navatala_ml_pointwise_tanh_bwd_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_neg_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_neg_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_neg_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_abs_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_abs_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_abs_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sqrt_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_sqrt_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_sqrt_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_rsqrt_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_rsqrt_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_rsqrt_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_reciprocal_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_reciprocal_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_reciprocal_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_exp_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_exp_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_exp_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_log_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_log_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_log_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sin_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_sin_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_sin_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cos_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_cos_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_cos_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_tan_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_tan_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_tan_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_floor_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_floor_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_floor_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_ceil_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_ceil_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_ceil_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_round_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_round_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_round_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_relu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_relu_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_relu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_leaky_relu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_leaky_relu_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_leaky_relu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_elu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_elu_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_elu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_tanh_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_tanh_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_tanh_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sigmoid_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_sigmoid_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_sigmoid_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_softplus_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_softplus_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_softplus_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_swish_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_swish_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_swish_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_gelu_tanh_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_gelu_tanh_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_gelu_tanh_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_add_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_add_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_add_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sub_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_sub_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_sub_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_mul_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_mul_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_mul_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_div_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_div_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_div_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_max_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_max_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_max_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_min_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_min_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_min_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_eq_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_cmp_eq_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_cmp_eq_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_ne_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_cmp_ne_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_cmp_ne_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_lt_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_cmp_lt_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_cmp_lt_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_le_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_cmp_le_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_cmp_le_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_gt_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_cmp_gt_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_cmp_gt_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_cmp_ge_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_cmp_ge_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_cmp_ge_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_relu_bwd_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_relu_bwd_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_relu_bwd_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_sigmoid_bwd_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_sigmoid_bwd_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_sigmoid_bwd_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_pointwise_tanh_bwd_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_pointwise_tanh_bwd_f16";
    std::string_view sv(k_opencl_navatala_ml_pointwise_tanh_bwd_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

