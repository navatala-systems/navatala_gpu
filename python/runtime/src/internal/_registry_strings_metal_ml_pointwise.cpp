// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_ml_pointwise` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_ml_pointwise_neg_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_neg_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_abs_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_abs_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = abs(x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_sqrt_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_sqrt_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_rsqrt_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_rsqrt_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (as_type<float>(0x3f800000u) / sqrt(x));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_reciprocal_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_reciprocal_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (as_type<float>(0x3f800000u) / x);
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_exp_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_exp_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_log_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_log_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_sin_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_sin_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_cos_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_cos_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_tan_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_tan_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_floor_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_floor_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_ceil_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_ceil_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_round_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_round_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_relu_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_relu_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (((x > as_type<float>(0x00000000u))) ? (x) : (as_type<float>(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_leaky_relu_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_leaky_relu_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (((x > as_type<float>(0x00000000u))) ? (x) : ((as_type<float>(0x3c23d70au) * x)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_elu_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_elu_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (((x > as_type<float>(0x00000000u))) ? (x) : ((exp(x) - as_type<float>(0x3f800000u))));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_tanh_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_tanh_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_sigmoid_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_sigmoid_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (as_type<float>(0x3f800000u) / (as_type<float>(0x3f800000u) + exp((-x))));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_softplus_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_softplus_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = log((as_type<float>(0x3f800000u) + exp(x)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_swish_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_swish_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = (x * (as_type<float>(0x3f800000u) / (as_type<float>(0x3f800000u) + exp((-x)))));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_gelu_tanh_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_gelu_tanh_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float xRaw = _input[gid];
    float x = xRaw;
    float yF = ((as_type<float>(0x3f000000u) * x) * (as_type<float>(0x3f800000u) + tanh((as_type<float>(0x3f4c422au) * (x + (as_type<float>(0x3d372713u) * (x * (x * x))))))));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_add_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_add_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_sub_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_sub_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_mul_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_mul_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_div_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_div_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_max_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_max_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_min_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_min_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_cmp_eq_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_cmp_eq_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av == bv)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_cmp_ne_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_cmp_ne_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av != bv)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_cmp_lt_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_cmp_lt_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av < bv)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_cmp_le_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_cmp_le_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av <= bv)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_cmp_gt_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_cmp_gt_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av > bv)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_cmp_ge_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_cmp_ge_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (((av >= bv)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_relu_bwd_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_relu_bwd_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (av * (((bv > as_type<float>(0x00000000u))) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u))));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_sigmoid_bwd_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_sigmoid_bwd_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (av * (bv * (as_type<float>(0x3f800000u) - bv)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_tanh_bwd_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_tanh_bwd_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float aRaw = a[gid];
    float bRaw = b[gid];
    float av = aRaw;
    float bv = bRaw;
    float yF = (av * (as_type<float>(0x3f800000u) - (bv * bv)));
    float y = yF;
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_neg_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_neg_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_abs_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_abs_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = abs(x);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_sqrt_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_sqrt_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_rsqrt_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_rsqrt_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (as_type<float>(0x3f800000u) / sqrt(x));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_reciprocal_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_reciprocal_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (as_type<float>(0x3f800000u) / x);
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_exp_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_exp_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_log_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_log_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_sin_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_sin_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_cos_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_cos_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_tan_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_tan_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_floor_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_floor_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_ceil_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_ceil_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_round_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_round_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_relu_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_relu_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (((x > as_type<float>(0x00000000u))) ? (x) : (as_type<float>(0x00000000u)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_leaky_relu_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_leaky_relu_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (((x > as_type<float>(0x00000000u))) ? (x) : ((as_type<float>(0x3c23d70au) * x)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_elu_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_elu_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (((x > as_type<float>(0x00000000u))) ? (x) : ((exp(x) - as_type<float>(0x3f800000u))));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_tanh_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_tanh_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_sigmoid_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_sigmoid_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (as_type<float>(0x3f800000u) / (as_type<float>(0x3f800000u) + exp((-x))));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_softplus_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_softplus_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = log((as_type<float>(0x3f800000u) + exp(x)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_swish_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_swish_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = (x * (as_type<float>(0x3f800000u) / (as_type<float>(0x3f800000u) + exp((-x)))));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_gelu_tanh_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_gelu_tanh_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half xRaw = _input[gid];
    float x = ((float)(xRaw));
    float yF = ((as_type<float>(0x3f000000u) * x) * (as_type<float>(0x3f800000u) + tanh((as_type<float>(0x3f4c422au) * (x + (as_type<float>(0x3d372713u) * (x * (x * x))))))));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_add_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_add_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_sub_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_sub_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_mul_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_mul_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_div_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_div_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_max_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_max_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_min_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_min_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
const char* k_metal_navatala_ml_pointwise_cmp_eq_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_cmp_eq_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av == bv)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_cmp_ne_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_cmp_ne_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av != bv)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_cmp_lt_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_cmp_lt_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av < bv)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_cmp_le_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_cmp_le_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av <= bv)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_cmp_gt_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_cmp_gt_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av > bv)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_cmp_ge_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_cmp_ge_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (((av >= bv)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_relu_bwd_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_relu_bwd_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av * (((bv > as_type<float>(0x00000000u))) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u))));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_sigmoid_bwd_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_sigmoid_bwd_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av * (bv * (as_type<float>(0x3f800000u) - bv)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";
const char* k_metal_navatala_ml_pointwise_tanh_bwd_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pointwise_tanh_bwd_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half aRaw = a[gid];
    half bRaw = b[gid];
    float av = ((float)(aRaw));
    float bv = ((float)(bRaw));
    float yF = (av * (as_type<float>(0x3f800000u) - (bv * bv)));
    half y = ((half)(yF));
    _output[gid] = y;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_neg_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_neg_f32 = {
  1,
  "navatala_ml_pointwise_neg_f32",
  "metal",
  "navatala_ml_pointwise_neg_f32",
  "kernel:metal:navatala_ml_pointwise_neg_f32",
  "abi-r1:metal:navatala_ml_pointwise_neg_f32",
  "abi-r1:metal:navatala_ml_pointwise_neg_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_neg_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_abs_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_abs_f32 = {
  1,
  "navatala_ml_pointwise_abs_f32",
  "metal",
  "navatala_ml_pointwise_abs_f32",
  "kernel:metal:navatala_ml_pointwise_abs_f32",
  "abi-r1:metal:navatala_ml_pointwise_abs_f32",
  "abi-r1:metal:navatala_ml_pointwise_abs_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_abs_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_sqrt_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_sqrt_f32 = {
  1,
  "navatala_ml_pointwise_sqrt_f32",
  "metal",
  "navatala_ml_pointwise_sqrt_f32",
  "kernel:metal:navatala_ml_pointwise_sqrt_f32",
  "abi-r1:metal:navatala_ml_pointwise_sqrt_f32",
  "abi-r1:metal:navatala_ml_pointwise_sqrt_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_sqrt_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_rsqrt_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_rsqrt_f32 = {
  1,
  "navatala_ml_pointwise_rsqrt_f32",
  "metal",
  "navatala_ml_pointwise_rsqrt_f32",
  "kernel:metal:navatala_ml_pointwise_rsqrt_f32",
  "abi-r1:metal:navatala_ml_pointwise_rsqrt_f32",
  "abi-r1:metal:navatala_ml_pointwise_rsqrt_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_rsqrt_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_reciprocal_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_reciprocal_f32 = {
  1,
  "navatala_ml_pointwise_reciprocal_f32",
  "metal",
  "navatala_ml_pointwise_reciprocal_f32",
  "kernel:metal:navatala_ml_pointwise_reciprocal_f32",
  "abi-r1:metal:navatala_ml_pointwise_reciprocal_f32",
  "abi-r1:metal:navatala_ml_pointwise_reciprocal_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_reciprocal_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_exp_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_exp_f32 = {
  1,
  "navatala_ml_pointwise_exp_f32",
  "metal",
  "navatala_ml_pointwise_exp_f32",
  "kernel:metal:navatala_ml_pointwise_exp_f32",
  "abi-r1:metal:navatala_ml_pointwise_exp_f32",
  "abi-r1:metal:navatala_ml_pointwise_exp_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_exp_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_log_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_log_f32 = {
  1,
  "navatala_ml_pointwise_log_f32",
  "metal",
  "navatala_ml_pointwise_log_f32",
  "kernel:metal:navatala_ml_pointwise_log_f32",
  "abi-r1:metal:navatala_ml_pointwise_log_f32",
  "abi-r1:metal:navatala_ml_pointwise_log_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_log_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_sin_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_sin_f32 = {
  1,
  "navatala_ml_pointwise_sin_f32",
  "metal",
  "navatala_ml_pointwise_sin_f32",
  "kernel:metal:navatala_ml_pointwise_sin_f32",
  "abi-r1:metal:navatala_ml_pointwise_sin_f32",
  "abi-r1:metal:navatala_ml_pointwise_sin_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_sin_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_cos_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_cos_f32 = {
  1,
  "navatala_ml_pointwise_cos_f32",
  "metal",
  "navatala_ml_pointwise_cos_f32",
  "kernel:metal:navatala_ml_pointwise_cos_f32",
  "abi-r1:metal:navatala_ml_pointwise_cos_f32",
  "abi-r1:metal:navatala_ml_pointwise_cos_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_cos_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_tan_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_tan_f32 = {
  1,
  "navatala_ml_pointwise_tan_f32",
  "metal",
  "navatala_ml_pointwise_tan_f32",
  "kernel:metal:navatala_ml_pointwise_tan_f32",
  "abi-r1:metal:navatala_ml_pointwise_tan_f32",
  "abi-r1:metal:navatala_ml_pointwise_tan_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_tan_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_floor_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_floor_f32 = {
  1,
  "navatala_ml_pointwise_floor_f32",
  "metal",
  "navatala_ml_pointwise_floor_f32",
  "kernel:metal:navatala_ml_pointwise_floor_f32",
  "abi-r1:metal:navatala_ml_pointwise_floor_f32",
  "abi-r1:metal:navatala_ml_pointwise_floor_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_floor_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_ceil_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_ceil_f32 = {
  1,
  "navatala_ml_pointwise_ceil_f32",
  "metal",
  "navatala_ml_pointwise_ceil_f32",
  "kernel:metal:navatala_ml_pointwise_ceil_f32",
  "abi-r1:metal:navatala_ml_pointwise_ceil_f32",
  "abi-r1:metal:navatala_ml_pointwise_ceil_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_ceil_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_round_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_round_f32 = {
  1,
  "navatala_ml_pointwise_round_f32",
  "metal",
  "navatala_ml_pointwise_round_f32",
  "kernel:metal:navatala_ml_pointwise_round_f32",
  "abi-r1:metal:navatala_ml_pointwise_round_f32",
  "abi-r1:metal:navatala_ml_pointwise_round_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_round_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_relu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_relu_f32 = {
  1,
  "navatala_ml_pointwise_relu_f32",
  "metal",
  "navatala_ml_pointwise_relu_f32",
  "kernel:metal:navatala_ml_pointwise_relu_f32",
  "abi-r1:metal:navatala_ml_pointwise_relu_f32",
  "abi-r1:metal:navatala_ml_pointwise_relu_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_relu_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_leaky_relu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_leaky_relu_f32 = {
  1,
  "navatala_ml_pointwise_leaky_relu_f32",
  "metal",
  "navatala_ml_pointwise_leaky_relu_f32",
  "kernel:metal:navatala_ml_pointwise_leaky_relu_f32",
  "abi-r1:metal:navatala_ml_pointwise_leaky_relu_f32",
  "abi-r1:metal:navatala_ml_pointwise_leaky_relu_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_leaky_relu_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_elu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_elu_f32 = {
  1,
  "navatala_ml_pointwise_elu_f32",
  "metal",
  "navatala_ml_pointwise_elu_f32",
  "kernel:metal:navatala_ml_pointwise_elu_f32",
  "abi-r1:metal:navatala_ml_pointwise_elu_f32",
  "abi-r1:metal:navatala_ml_pointwise_elu_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_elu_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_tanh_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_tanh_f32 = {
  1,
  "navatala_ml_pointwise_tanh_f32",
  "metal",
  "navatala_ml_pointwise_tanh_f32",
  "kernel:metal:navatala_ml_pointwise_tanh_f32",
  "abi-r1:metal:navatala_ml_pointwise_tanh_f32",
  "abi-r1:metal:navatala_ml_pointwise_tanh_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_tanh_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_sigmoid_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_sigmoid_f32 = {
  1,
  "navatala_ml_pointwise_sigmoid_f32",
  "metal",
  "navatala_ml_pointwise_sigmoid_f32",
  "kernel:metal:navatala_ml_pointwise_sigmoid_f32",
  "abi-r1:metal:navatala_ml_pointwise_sigmoid_f32",
  "abi-r1:metal:navatala_ml_pointwise_sigmoid_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_sigmoid_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_softplus_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_softplus_f32 = {
  1,
  "navatala_ml_pointwise_softplus_f32",
  "metal",
  "navatala_ml_pointwise_softplus_f32",
  "kernel:metal:navatala_ml_pointwise_softplus_f32",
  "abi-r1:metal:navatala_ml_pointwise_softplus_f32",
  "abi-r1:metal:navatala_ml_pointwise_softplus_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_softplus_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_swish_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_swish_f32 = {
  1,
  "navatala_ml_pointwise_swish_f32",
  "metal",
  "navatala_ml_pointwise_swish_f32",
  "kernel:metal:navatala_ml_pointwise_swish_f32",
  "abi-r1:metal:navatala_ml_pointwise_swish_f32",
  "abi-r1:metal:navatala_ml_pointwise_swish_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_swish_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_gelu_tanh_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_gelu_tanh_f32 = {
  1,
  "navatala_ml_pointwise_gelu_tanh_f32",
  "metal",
  "navatala_ml_pointwise_gelu_tanh_f32",
  "kernel:metal:navatala_ml_pointwise_gelu_tanh_f32",
  "abi-r1:metal:navatala_ml_pointwise_gelu_tanh_f32",
  "abi-r1:metal:navatala_ml_pointwise_gelu_tanh_f32",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_gelu_tanh_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_add_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_add_f32 = {
  1,
  "navatala_ml_pointwise_add_f32",
  "metal",
  "navatala_ml_pointwise_add_f32",
  "kernel:metal:navatala_ml_pointwise_add_f32",
  "abi-r1:metal:navatala_ml_pointwise_add_f32",
  "abi-r1:metal:navatala_ml_pointwise_add_f32",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_add_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_sub_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_sub_f32 = {
  1,
  "navatala_ml_pointwise_sub_f32",
  "metal",
  "navatala_ml_pointwise_sub_f32",
  "kernel:metal:navatala_ml_pointwise_sub_f32",
  "abi-r1:metal:navatala_ml_pointwise_sub_f32",
  "abi-r1:metal:navatala_ml_pointwise_sub_f32",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_sub_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_mul_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_mul_f32 = {
  1,
  "navatala_ml_pointwise_mul_f32",
  "metal",
  "navatala_ml_pointwise_mul_f32",
  "kernel:metal:navatala_ml_pointwise_mul_f32",
  "abi-r1:metal:navatala_ml_pointwise_mul_f32",
  "abi-r1:metal:navatala_ml_pointwise_mul_f32",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_mul_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_div_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_div_f32 = {
  1,
  "navatala_ml_pointwise_div_f32",
  "metal",
  "navatala_ml_pointwise_div_f32",
  "kernel:metal:navatala_ml_pointwise_div_f32",
  "abi-r1:metal:navatala_ml_pointwise_div_f32",
  "abi-r1:metal:navatala_ml_pointwise_div_f32",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_div_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_max_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_max_f32 = {
  1,
  "navatala_ml_pointwise_max_f32",
  "metal",
  "navatala_ml_pointwise_max_f32",
  "kernel:metal:navatala_ml_pointwise_max_f32",
  "abi-r1:metal:navatala_ml_pointwise_max_f32",
  "abi-r1:metal:navatala_ml_pointwise_max_f32",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_max_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_min_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_min_f32 = {
  1,
  "navatala_ml_pointwise_min_f32",
  "metal",
  "navatala_ml_pointwise_min_f32",
  "kernel:metal:navatala_ml_pointwise_min_f32",
  "abi-r1:metal:navatala_ml_pointwise_min_f32",
  "abi-r1:metal:navatala_ml_pointwise_min_f32",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_min_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_cmp_eq_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_cmp_eq_f32 = {
  1,
  "navatala_ml_pointwise_cmp_eq_f32",
  "metal",
  "navatala_ml_pointwise_cmp_eq_f32",
  "kernel:metal:navatala_ml_pointwise_cmp_eq_f32",
  "abi-r1:metal:navatala_ml_pointwise_cmp_eq_f32",
  "abi-r1:metal:navatala_ml_pointwise_cmp_eq_f32",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_cmp_eq_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_cmp_ne_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_cmp_ne_f32 = {
  1,
  "navatala_ml_pointwise_cmp_ne_f32",
  "metal",
  "navatala_ml_pointwise_cmp_ne_f32",
  "kernel:metal:navatala_ml_pointwise_cmp_ne_f32",
  "abi-r1:metal:navatala_ml_pointwise_cmp_ne_f32",
  "abi-r1:metal:navatala_ml_pointwise_cmp_ne_f32",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_cmp_ne_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_cmp_lt_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_cmp_lt_f32 = {
  1,
  "navatala_ml_pointwise_cmp_lt_f32",
  "metal",
  "navatala_ml_pointwise_cmp_lt_f32",
  "kernel:metal:navatala_ml_pointwise_cmp_lt_f32",
  "abi-r1:metal:navatala_ml_pointwise_cmp_lt_f32",
  "abi-r1:metal:navatala_ml_pointwise_cmp_lt_f32",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_cmp_lt_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_cmp_le_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_cmp_le_f32 = {
  1,
  "navatala_ml_pointwise_cmp_le_f32",
  "metal",
  "navatala_ml_pointwise_cmp_le_f32",
  "kernel:metal:navatala_ml_pointwise_cmp_le_f32",
  "abi-r1:metal:navatala_ml_pointwise_cmp_le_f32",
  "abi-r1:metal:navatala_ml_pointwise_cmp_le_f32",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_cmp_le_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_cmp_gt_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_cmp_gt_f32 = {
  1,
  "navatala_ml_pointwise_cmp_gt_f32",
  "metal",
  "navatala_ml_pointwise_cmp_gt_f32",
  "kernel:metal:navatala_ml_pointwise_cmp_gt_f32",
  "abi-r1:metal:navatala_ml_pointwise_cmp_gt_f32",
  "abi-r1:metal:navatala_ml_pointwise_cmp_gt_f32",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_cmp_gt_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_cmp_ge_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_cmp_ge_f32 = {
  1,
  "navatala_ml_pointwise_cmp_ge_f32",
  "metal",
  "navatala_ml_pointwise_cmp_ge_f32",
  "kernel:metal:navatala_ml_pointwise_cmp_ge_f32",
  "abi-r1:metal:navatala_ml_pointwise_cmp_ge_f32",
  "abi-r1:metal:navatala_ml_pointwise_cmp_ge_f32",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_cmp_ge_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_relu_bwd_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_relu_bwd_f32 = {
  1,
  "navatala_ml_pointwise_relu_bwd_f32",
  "metal",
  "navatala_ml_pointwise_relu_bwd_f32",
  "kernel:metal:navatala_ml_pointwise_relu_bwd_f32",
  "abi-r1:metal:navatala_ml_pointwise_relu_bwd_f32",
  "abi-r1:metal:navatala_ml_pointwise_relu_bwd_f32",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_relu_bwd_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_sigmoid_bwd_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_sigmoid_bwd_f32 = {
  1,
  "navatala_ml_pointwise_sigmoid_bwd_f32",
  "metal",
  "navatala_ml_pointwise_sigmoid_bwd_f32",
  "kernel:metal:navatala_ml_pointwise_sigmoid_bwd_f32",
  "abi-r1:metal:navatala_ml_pointwise_sigmoid_bwd_f32",
  "abi-r1:metal:navatala_ml_pointwise_sigmoid_bwd_f32",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_sigmoid_bwd_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_tanh_bwd_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_tanh_bwd_f32 = {
  1,
  "navatala_ml_pointwise_tanh_bwd_f32",
  "metal",
  "navatala_ml_pointwise_tanh_bwd_f32",
  "kernel:metal:navatala_ml_pointwise_tanh_bwd_f32",
  "abi-r1:metal:navatala_ml_pointwise_tanh_bwd_f32",
  "abi-r1:metal:navatala_ml_pointwise_tanh_bwd_f32",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_tanh_bwd_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_neg_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_neg_f16 = {
  1,
  "navatala_ml_pointwise_neg_f16",
  "metal",
  "navatala_ml_pointwise_neg_f16",
  "kernel:metal:navatala_ml_pointwise_neg_f16",
  "abi-r1:metal:navatala_ml_pointwise_neg_f16",
  "abi-r1:metal:navatala_ml_pointwise_neg_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_neg_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_abs_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_abs_f16 = {
  1,
  "navatala_ml_pointwise_abs_f16",
  "metal",
  "navatala_ml_pointwise_abs_f16",
  "kernel:metal:navatala_ml_pointwise_abs_f16",
  "abi-r1:metal:navatala_ml_pointwise_abs_f16",
  "abi-r1:metal:navatala_ml_pointwise_abs_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_abs_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_sqrt_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_sqrt_f16 = {
  1,
  "navatala_ml_pointwise_sqrt_f16",
  "metal",
  "navatala_ml_pointwise_sqrt_f16",
  "kernel:metal:navatala_ml_pointwise_sqrt_f16",
  "abi-r1:metal:navatala_ml_pointwise_sqrt_f16",
  "abi-r1:metal:navatala_ml_pointwise_sqrt_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_sqrt_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_rsqrt_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_rsqrt_f16 = {
  1,
  "navatala_ml_pointwise_rsqrt_f16",
  "metal",
  "navatala_ml_pointwise_rsqrt_f16",
  "kernel:metal:navatala_ml_pointwise_rsqrt_f16",
  "abi-r1:metal:navatala_ml_pointwise_rsqrt_f16",
  "abi-r1:metal:navatala_ml_pointwise_rsqrt_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_rsqrt_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_reciprocal_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_reciprocal_f16 = {
  1,
  "navatala_ml_pointwise_reciprocal_f16",
  "metal",
  "navatala_ml_pointwise_reciprocal_f16",
  "kernel:metal:navatala_ml_pointwise_reciprocal_f16",
  "abi-r1:metal:navatala_ml_pointwise_reciprocal_f16",
  "abi-r1:metal:navatala_ml_pointwise_reciprocal_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_reciprocal_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_exp_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_exp_f16 = {
  1,
  "navatala_ml_pointwise_exp_f16",
  "metal",
  "navatala_ml_pointwise_exp_f16",
  "kernel:metal:navatala_ml_pointwise_exp_f16",
  "abi-r1:metal:navatala_ml_pointwise_exp_f16",
  "abi-r1:metal:navatala_ml_pointwise_exp_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_exp_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_log_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_log_f16 = {
  1,
  "navatala_ml_pointwise_log_f16",
  "metal",
  "navatala_ml_pointwise_log_f16",
  "kernel:metal:navatala_ml_pointwise_log_f16",
  "abi-r1:metal:navatala_ml_pointwise_log_f16",
  "abi-r1:metal:navatala_ml_pointwise_log_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_log_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_sin_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_sin_f16 = {
  1,
  "navatala_ml_pointwise_sin_f16",
  "metal",
  "navatala_ml_pointwise_sin_f16",
  "kernel:metal:navatala_ml_pointwise_sin_f16",
  "abi-r1:metal:navatala_ml_pointwise_sin_f16",
  "abi-r1:metal:navatala_ml_pointwise_sin_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_sin_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_cos_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_cos_f16 = {
  1,
  "navatala_ml_pointwise_cos_f16",
  "metal",
  "navatala_ml_pointwise_cos_f16",
  "kernel:metal:navatala_ml_pointwise_cos_f16",
  "abi-r1:metal:navatala_ml_pointwise_cos_f16",
  "abi-r1:metal:navatala_ml_pointwise_cos_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_cos_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_tan_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_tan_f16 = {
  1,
  "navatala_ml_pointwise_tan_f16",
  "metal",
  "navatala_ml_pointwise_tan_f16",
  "kernel:metal:navatala_ml_pointwise_tan_f16",
  "abi-r1:metal:navatala_ml_pointwise_tan_f16",
  "abi-r1:metal:navatala_ml_pointwise_tan_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_tan_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_floor_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_floor_f16 = {
  1,
  "navatala_ml_pointwise_floor_f16",
  "metal",
  "navatala_ml_pointwise_floor_f16",
  "kernel:metal:navatala_ml_pointwise_floor_f16",
  "abi-r1:metal:navatala_ml_pointwise_floor_f16",
  "abi-r1:metal:navatala_ml_pointwise_floor_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_floor_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_ceil_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_ceil_f16 = {
  1,
  "navatala_ml_pointwise_ceil_f16",
  "metal",
  "navatala_ml_pointwise_ceil_f16",
  "kernel:metal:navatala_ml_pointwise_ceil_f16",
  "abi-r1:metal:navatala_ml_pointwise_ceil_f16",
  "abi-r1:metal:navatala_ml_pointwise_ceil_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_ceil_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_round_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_round_f16 = {
  1,
  "navatala_ml_pointwise_round_f16",
  "metal",
  "navatala_ml_pointwise_round_f16",
  "kernel:metal:navatala_ml_pointwise_round_f16",
  "abi-r1:metal:navatala_ml_pointwise_round_f16",
  "abi-r1:metal:navatala_ml_pointwise_round_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_round_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_relu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_relu_f16 = {
  1,
  "navatala_ml_pointwise_relu_f16",
  "metal",
  "navatala_ml_pointwise_relu_f16",
  "kernel:metal:navatala_ml_pointwise_relu_f16",
  "abi-r1:metal:navatala_ml_pointwise_relu_f16",
  "abi-r1:metal:navatala_ml_pointwise_relu_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_relu_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_leaky_relu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_leaky_relu_f16 = {
  1,
  "navatala_ml_pointwise_leaky_relu_f16",
  "metal",
  "navatala_ml_pointwise_leaky_relu_f16",
  "kernel:metal:navatala_ml_pointwise_leaky_relu_f16",
  "abi-r1:metal:navatala_ml_pointwise_leaky_relu_f16",
  "abi-r1:metal:navatala_ml_pointwise_leaky_relu_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_leaky_relu_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_elu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_elu_f16 = {
  1,
  "navatala_ml_pointwise_elu_f16",
  "metal",
  "navatala_ml_pointwise_elu_f16",
  "kernel:metal:navatala_ml_pointwise_elu_f16",
  "abi-r1:metal:navatala_ml_pointwise_elu_f16",
  "abi-r1:metal:navatala_ml_pointwise_elu_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_elu_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_tanh_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_tanh_f16 = {
  1,
  "navatala_ml_pointwise_tanh_f16",
  "metal",
  "navatala_ml_pointwise_tanh_f16",
  "kernel:metal:navatala_ml_pointwise_tanh_f16",
  "abi-r1:metal:navatala_ml_pointwise_tanh_f16",
  "abi-r1:metal:navatala_ml_pointwise_tanh_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_tanh_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_sigmoid_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_sigmoid_f16 = {
  1,
  "navatala_ml_pointwise_sigmoid_f16",
  "metal",
  "navatala_ml_pointwise_sigmoid_f16",
  "kernel:metal:navatala_ml_pointwise_sigmoid_f16",
  "abi-r1:metal:navatala_ml_pointwise_sigmoid_f16",
  "abi-r1:metal:navatala_ml_pointwise_sigmoid_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_sigmoid_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_softplus_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_softplus_f16 = {
  1,
  "navatala_ml_pointwise_softplus_f16",
  "metal",
  "navatala_ml_pointwise_softplus_f16",
  "kernel:metal:navatala_ml_pointwise_softplus_f16",
  "abi-r1:metal:navatala_ml_pointwise_softplus_f16",
  "abi-r1:metal:navatala_ml_pointwise_softplus_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_softplus_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_swish_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_swish_f16 = {
  1,
  "navatala_ml_pointwise_swish_f16",
  "metal",
  "navatala_ml_pointwise_swish_f16",
  "kernel:metal:navatala_ml_pointwise_swish_f16",
  "abi-r1:metal:navatala_ml_pointwise_swish_f16",
  "abi-r1:metal:navatala_ml_pointwise_swish_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_swish_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_gelu_tanh_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_gelu_tanh_f16 = {
  1,
  "navatala_ml_pointwise_gelu_tanh_f16",
  "metal",
  "navatala_ml_pointwise_gelu_tanh_f16",
  "kernel:metal:navatala_ml_pointwise_gelu_tanh_f16",
  "abi-r1:metal:navatala_ml_pointwise_gelu_tanh_f16",
  "abi-r1:metal:navatala_ml_pointwise_gelu_tanh_f16",
  3,
  kAbiArgs_metal_navatala_ml_pointwise_gelu_tanh_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_add_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_add_f16 = {
  1,
  "navatala_ml_pointwise_add_f16",
  "metal",
  "navatala_ml_pointwise_add_f16",
  "kernel:metal:navatala_ml_pointwise_add_f16",
  "abi-r1:metal:navatala_ml_pointwise_add_f16",
  "abi-r1:metal:navatala_ml_pointwise_add_f16",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_add_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_sub_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_sub_f16 = {
  1,
  "navatala_ml_pointwise_sub_f16",
  "metal",
  "navatala_ml_pointwise_sub_f16",
  "kernel:metal:navatala_ml_pointwise_sub_f16",
  "abi-r1:metal:navatala_ml_pointwise_sub_f16",
  "abi-r1:metal:navatala_ml_pointwise_sub_f16",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_sub_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_mul_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_mul_f16 = {
  1,
  "navatala_ml_pointwise_mul_f16",
  "metal",
  "navatala_ml_pointwise_mul_f16",
  "kernel:metal:navatala_ml_pointwise_mul_f16",
  "abi-r1:metal:navatala_ml_pointwise_mul_f16",
  "abi-r1:metal:navatala_ml_pointwise_mul_f16",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_mul_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_div_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_div_f16 = {
  1,
  "navatala_ml_pointwise_div_f16",
  "metal",
  "navatala_ml_pointwise_div_f16",
  "kernel:metal:navatala_ml_pointwise_div_f16",
  "abi-r1:metal:navatala_ml_pointwise_div_f16",
  "abi-r1:metal:navatala_ml_pointwise_div_f16",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_div_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_max_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_max_f16 = {
  1,
  "navatala_ml_pointwise_max_f16",
  "metal",
  "navatala_ml_pointwise_max_f16",
  "kernel:metal:navatala_ml_pointwise_max_f16",
  "abi-r1:metal:navatala_ml_pointwise_max_f16",
  "abi-r1:metal:navatala_ml_pointwise_max_f16",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_max_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_min_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_min_f16 = {
  1,
  "navatala_ml_pointwise_min_f16",
  "metal",
  "navatala_ml_pointwise_min_f16",
  "kernel:metal:navatala_ml_pointwise_min_f16",
  "abi-r1:metal:navatala_ml_pointwise_min_f16",
  "abi-r1:metal:navatala_ml_pointwise_min_f16",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_min_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_cmp_eq_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_cmp_eq_f16 = {
  1,
  "navatala_ml_pointwise_cmp_eq_f16",
  "metal",
  "navatala_ml_pointwise_cmp_eq_f16",
  "kernel:metal:navatala_ml_pointwise_cmp_eq_f16",
  "abi-r1:metal:navatala_ml_pointwise_cmp_eq_f16",
  "abi-r1:metal:navatala_ml_pointwise_cmp_eq_f16",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_cmp_eq_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_cmp_ne_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_cmp_ne_f16 = {
  1,
  "navatala_ml_pointwise_cmp_ne_f16",
  "metal",
  "navatala_ml_pointwise_cmp_ne_f16",
  "kernel:metal:navatala_ml_pointwise_cmp_ne_f16",
  "abi-r1:metal:navatala_ml_pointwise_cmp_ne_f16",
  "abi-r1:metal:navatala_ml_pointwise_cmp_ne_f16",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_cmp_ne_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_cmp_lt_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_cmp_lt_f16 = {
  1,
  "navatala_ml_pointwise_cmp_lt_f16",
  "metal",
  "navatala_ml_pointwise_cmp_lt_f16",
  "kernel:metal:navatala_ml_pointwise_cmp_lt_f16",
  "abi-r1:metal:navatala_ml_pointwise_cmp_lt_f16",
  "abi-r1:metal:navatala_ml_pointwise_cmp_lt_f16",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_cmp_lt_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_cmp_le_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_cmp_le_f16 = {
  1,
  "navatala_ml_pointwise_cmp_le_f16",
  "metal",
  "navatala_ml_pointwise_cmp_le_f16",
  "kernel:metal:navatala_ml_pointwise_cmp_le_f16",
  "abi-r1:metal:navatala_ml_pointwise_cmp_le_f16",
  "abi-r1:metal:navatala_ml_pointwise_cmp_le_f16",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_cmp_le_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_cmp_gt_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_cmp_gt_f16 = {
  1,
  "navatala_ml_pointwise_cmp_gt_f16",
  "metal",
  "navatala_ml_pointwise_cmp_gt_f16",
  "kernel:metal:navatala_ml_pointwise_cmp_gt_f16",
  "abi-r1:metal:navatala_ml_pointwise_cmp_gt_f16",
  "abi-r1:metal:navatala_ml_pointwise_cmp_gt_f16",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_cmp_gt_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_cmp_ge_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_cmp_ge_f16 = {
  1,
  "navatala_ml_pointwise_cmp_ge_f16",
  "metal",
  "navatala_ml_pointwise_cmp_ge_f16",
  "kernel:metal:navatala_ml_pointwise_cmp_ge_f16",
  "abi-r1:metal:navatala_ml_pointwise_cmp_ge_f16",
  "abi-r1:metal:navatala_ml_pointwise_cmp_ge_f16",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_cmp_ge_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_relu_bwd_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_relu_bwd_f16 = {
  1,
  "navatala_ml_pointwise_relu_bwd_f16",
  "metal",
  "navatala_ml_pointwise_relu_bwd_f16",
  "kernel:metal:navatala_ml_pointwise_relu_bwd_f16",
  "abi-r1:metal:navatala_ml_pointwise_relu_bwd_f16",
  "abi-r1:metal:navatala_ml_pointwise_relu_bwd_f16",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_relu_bwd_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_sigmoid_bwd_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_sigmoid_bwd_f16 = {
  1,
  "navatala_ml_pointwise_sigmoid_bwd_f16",
  "metal",
  "navatala_ml_pointwise_sigmoid_bwd_f16",
  "kernel:metal:navatala_ml_pointwise_sigmoid_bwd_f16",
  "abi-r1:metal:navatala_ml_pointwise_sigmoid_bwd_f16",
  "abi-r1:metal:navatala_ml_pointwise_sigmoid_bwd_f16",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_sigmoid_bwd_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pointwise_tanh_bwd_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pointwise_tanh_bwd_f16 = {
  1,
  "navatala_ml_pointwise_tanh_bwd_f16",
  "metal",
  "navatala_ml_pointwise_tanh_bwd_f16",
  "kernel:metal:navatala_ml_pointwise_tanh_bwd_f16",
  "abi-r1:metal:navatala_ml_pointwise_tanh_bwd_f16",
  "abi-r1:metal:navatala_ml_pointwise_tanh_bwd_f16",
  4,
  kAbiArgs_metal_navatala_ml_pointwise_tanh_bwd_f16
};

bool tryGetKernelAbiManifest_metal_ml_pointwise(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_neg_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_neg_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_abs_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_abs_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sqrt_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_sqrt_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_rsqrt_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_rsqrt_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_reciprocal_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_reciprocal_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_exp_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_exp_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_log_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_log_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sin_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_sin_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cos_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_cos_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_tan_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_tan_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_floor_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_floor_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_ceil_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_ceil_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_round_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_round_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_relu_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_relu_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_leaky_relu_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_leaky_relu_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_elu_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_elu_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_tanh_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_tanh_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sigmoid_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_sigmoid_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_softplus_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_softplus_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_swish_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_swish_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_gelu_tanh_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_gelu_tanh_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_add_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_add_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sub_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_sub_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_mul_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_mul_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_div_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_div_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_max_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_max_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_min_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_min_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_eq_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_cmp_eq_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_ne_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_cmp_ne_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_lt_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_cmp_lt_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_le_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_cmp_le_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_gt_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_cmp_gt_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_ge_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_cmp_ge_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_relu_bwd_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_relu_bwd_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sigmoid_bwd_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_sigmoid_bwd_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_tanh_bwd_f32") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_tanh_bwd_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_neg_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_neg_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_abs_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_abs_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sqrt_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_sqrt_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_rsqrt_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_rsqrt_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_reciprocal_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_reciprocal_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_exp_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_exp_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_log_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_log_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sin_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_sin_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cos_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_cos_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_tan_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_tan_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_floor_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_floor_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_ceil_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_ceil_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_round_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_round_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_relu_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_relu_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_leaky_relu_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_leaky_relu_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_elu_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_elu_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_tanh_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_tanh_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sigmoid_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_sigmoid_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_softplus_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_softplus_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_swish_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_swish_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_gelu_tanh_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_gelu_tanh_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_add_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_add_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sub_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_sub_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_mul_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_mul_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_div_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_div_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_max_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_max_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_min_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_min_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_eq_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_cmp_eq_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_ne_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_cmp_ne_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_lt_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_cmp_lt_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_le_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_cmp_le_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_gt_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_cmp_gt_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_ge_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_cmp_ge_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_relu_bwd_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_relu_bwd_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sigmoid_bwd_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_sigmoid_bwd_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_tanh_bwd_f16") {
    out = &kAbiManifest_metal_navatala_ml_pointwise_tanh_bwd_f16;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_ml_pointwise(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_neg_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_neg_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_neg_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_abs_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_abs_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_abs_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sqrt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_sqrt_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_sqrt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_rsqrt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_rsqrt_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_rsqrt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_reciprocal_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_reciprocal_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_reciprocal_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_exp_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_exp_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_exp_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_log_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_log_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_log_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sin_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_sin_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_sin_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cos_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_cos_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_cos_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_tan_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_tan_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_tan_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_floor_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_floor_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_floor_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_ceil_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_ceil_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_ceil_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_round_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_round_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_round_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_relu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_relu_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_relu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_leaky_relu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_leaky_relu_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_leaky_relu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_elu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_elu_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_elu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_tanh_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_tanh_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_tanh_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sigmoid_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_sigmoid_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_sigmoid_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_softplus_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_softplus_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_softplus_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_swish_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_swish_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_swish_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_gelu_tanh_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_gelu_tanh_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_gelu_tanh_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_add_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_add_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_add_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sub_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_sub_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_sub_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_mul_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_mul_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_mul_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_div_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_div_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_div_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_max_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_min_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_min_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_min_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_eq_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_cmp_eq_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_cmp_eq_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_ne_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_cmp_ne_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_cmp_ne_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_lt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_cmp_lt_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_cmp_lt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_le_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_cmp_le_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_cmp_le_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_gt_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_cmp_gt_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_cmp_gt_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_ge_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_cmp_ge_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_cmp_ge_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_relu_bwd_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_relu_bwd_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_relu_bwd_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sigmoid_bwd_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_sigmoid_bwd_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_sigmoid_bwd_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_tanh_bwd_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_tanh_bwd_f32";
    std::string_view sv(k_metal_navatala_ml_pointwise_tanh_bwd_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_neg_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_neg_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_neg_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_abs_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_abs_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_abs_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sqrt_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_sqrt_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_sqrt_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_rsqrt_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_rsqrt_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_rsqrt_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_reciprocal_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_reciprocal_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_reciprocal_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_exp_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_exp_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_exp_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_log_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_log_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_log_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sin_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_sin_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_sin_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cos_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_cos_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_cos_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_tan_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_tan_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_tan_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_floor_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_floor_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_floor_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_ceil_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_ceil_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_ceil_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_round_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_round_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_round_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_relu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_relu_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_relu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_leaky_relu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_leaky_relu_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_leaky_relu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_elu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_elu_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_elu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_tanh_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_tanh_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_tanh_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sigmoid_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_sigmoid_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_sigmoid_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_softplus_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_softplus_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_softplus_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_swish_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_swish_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_swish_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_gelu_tanh_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_gelu_tanh_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_gelu_tanh_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_add_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_add_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_add_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sub_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_sub_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_sub_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_mul_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_mul_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_mul_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_div_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_div_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_div_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_max_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_max_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_max_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_min_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_min_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_min_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_eq_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_cmp_eq_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_cmp_eq_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_ne_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_cmp_ne_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_cmp_ne_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_lt_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_cmp_lt_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_cmp_lt_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_le_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_cmp_le_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_cmp_le_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_gt_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_cmp_gt_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_cmp_gt_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_cmp_ge_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_cmp_ge_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_cmp_ge_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_relu_bwd_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_relu_bwd_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_relu_bwd_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_sigmoid_bwd_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_sigmoid_bwd_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_sigmoid_bwd_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pointwise_tanh_bwd_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pointwise_tanh_bwd_f16";
    std::string_view sv(k_metal_navatala_ml_pointwise_tanh_bwd_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

