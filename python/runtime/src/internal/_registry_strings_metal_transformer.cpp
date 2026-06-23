// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_transformer` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_transformer_gelu_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_gelu_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float x2 = (x * x);
    float x3 = (x2 * x);
    float cubeTerm = (as_type<float>(0x3d372713u) * x3);
    float inner = (x + cubeTerm);
    float tanhArg = (as_type<float>(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (as_type<float>(0x3f800000u) + tanhVal);
    float halfX = (as_type<float>(0x3f000000u) * x);
    float result = (halfX * onePlusTanh);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_gelu_tanh_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_gelu_tanh_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float x2 = (x * x);
    float x3 = (x2 * x);
    float cubeTerm = (as_type<float>(0x3d372713u) * x3);
    float inner = (x + cubeTerm);
    float tanhArg = (as_type<float>(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (as_type<float>(0x3f800000u) + tanhVal);
    float halfX = (as_type<float>(0x3f000000u) * x);
    float result = (halfX * onePlusTanh);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_silu_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_silu_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float negX = (-x);
    float expNegX = exp(negX);
    float denom = (as_type<float>(0x3f800000u) + expNegX);
    float result = (x / denom);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_relu_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_relu_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    bool isPositive = (x > as_type<float>(0x00000000u));
    float result = ((isPositive) ? (x) : (as_type<float>(0x00000000u)));
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_quick_gelu_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_quick_gelu_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float scaled = (as_type<float>(0x3fd9db23u) * x);
    float negScaled = (-scaled);
    float expNegScaled = exp(negScaled);
    float denom = (as_type<float>(0x3f800000u) + expNegScaled);
    float sigmoid = (as_type<float>(0x3f800000u) / denom);
    float result = (x * sigmoid);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_squared_relu_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_squared_relu_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    bool isPositive = (x > as_type<float>(0x00000000u));
    float relu = ((isPositive) ? (x) : (as_type<float>(0x00000000u)));
    float result = (relu * relu);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_bias_gelu_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_bias_gelu_f32(device const float* _input [[buffer(0)]], device const float* bias [[buffer(1)]], device const uint* count [[buffer(2)]], device const uint* biasSize [[buffer(3)]], device float* _output [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint bSize = biasSize[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    uint biasIdx = (gid % bSize);
    float b = bias[biasIdx];
    float xBias = (x + b);
    float x2 = (xBias * xBias);
    float x3 = (x2 * xBias);
    float cubeTerm = (as_type<float>(0x3d372713u) * x3);
    float inner = (xBias + cubeTerm);
    float tanhArg = (as_type<float>(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (as_type<float>(0x3f800000u) + tanhVal);
    float halfXBias = (as_type<float>(0x3f000000u) * xBias);
    float result = (halfXBias * onePlusTanh);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_gelu_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_gelu_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half x2 = (x * x);
    half x3 = (x2 * x);
    half cubeTerm = (half(0.044715) * x3);
    half inner = (x + cubeTerm);
    half tanhArg = (half(0.797885) * inner);
    half tanhVal = tanh(tanhArg);
    half onePlusTanh = (half(1.000000) + tanhVal);
    half halfX = (half(0.500000) * x);
    half result = (halfX * onePlusTanh);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_silu_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_silu_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half negX = (-x);
    half expNegX = exp(negX);
    half denom = (half(1.000000) + expNegX);
    half result = (x / denom);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_relu_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_relu_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    bool isPositive = (x > half(0.000000));
    half result = ((isPositive) ? (x) : (half(0.000000)));
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_quick_gelu_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_quick_gelu_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half scaled = (half(1.702000) * x);
    half negScaled = (-scaled);
    half expNegScaled = exp(negScaled);
    half denom = (half(1.000000) + expNegScaled);
    half sigmoid = (half(1.000000) / denom);
    half result = (x * sigmoid);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_squared_relu_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_squared_relu_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    bool isPositive = (x > half(0.000000));
    half relu = ((isPositive) ? (x) : (half(0.000000)));
    half result = (relu * relu);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_geglu_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_geglu_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint hidden = (n / 2u);
  bool inBounds = (gid < hidden);
  if (inBounds) {
    float gate = _input[gid];
    uint valueIdx = (gid + hidden);
    float value = _input[valueIdx];
    float gate2 = (gate * gate);
    float gate3 = (gate2 * gate);
    float cubeTerm = (as_type<float>(0x3d372713u) * gate3);
    float inner = (gate + cubeTerm);
    float tanhArg = (as_type<float>(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (as_type<float>(0x3f800000u) + tanhVal);
    float halfGate = (as_type<float>(0x3f000000u) * gate);
    float geluGate = (halfGate * onePlusTanh);
    float result = (geluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_swiglu_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_swiglu_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint hidden = (n / 2u);
  bool inBounds = (gid < hidden);
  if (inBounds) {
    float gate = _input[gid];
    uint valueIdx = (gid + hidden);
    float value = _input[valueIdx];
    float negGate = (-gate);
    float expNegGate = exp(negGate);
    float denom = (as_type<float>(0x3f800000u) + expNegGate);
    float siluGate = (gate / denom);
    float result = (siluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_reglu_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_reglu_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint hidden = (n / 2u);
  bool inBounds = (gid < hidden);
  if (inBounds) {
    float gate = _input[gid];
    uint valueIdx = (gid + hidden);
    float value = _input[valueIdx];
    bool isPositive = (gate > as_type<float>(0x00000000u));
    float reluGate = ((isPositive) ? (gate) : (as_type<float>(0x00000000u)));
    float result = (reluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_geglu_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_geglu_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint hidden = (n / 2u);
  bool inBounds = (gid < hidden);
  if (inBounds) {
    half gate = _input[gid];
    uint valueIdx = (gid + hidden);
    half value = _input[valueIdx];
    half gate2 = (gate * gate);
    half gate3 = (gate2 * gate);
    half cubeTerm = (half(0.044715) * gate3);
    half inner = (gate + cubeTerm);
    half tanhArg = (half(0.797885) * inner);
    half tanhVal = tanh(tanhArg);
    half onePlusTanh = (half(1.000000) + tanhVal);
    half halfGate = (half(0.500000) * gate);
    half geluGate = (halfGate * onePlusTanh);
    half result = (geluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_swiglu_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_swiglu_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint hidden = (n / 2u);
  bool inBounds = (gid < hidden);
  if (inBounds) {
    half gate = _input[gid];
    uint valueIdx = (gid + hidden);
    half value = _input[valueIdx];
    half negGate = (-gate);
    half expNegGate = exp(negGate);
    half denom = (half(1.000000) + expNegGate);
    half siluGate = (gate / denom);
    half result = (siluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_reglu_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_reglu_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint hidden = (n / 2u);
  bool inBounds = (gid < hidden);
  if (inBounds) {
    half gate = _input[gid];
    uint valueIdx = (gid + hidden);
    half value = _input[valueIdx];
    bool isPositive = (gate > half(0.000000));
    half reluGate = ((isPositive) ? (gate) : (half(0.000000)));
    half result = (reluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_gelu_backward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_gelu_backward_f32(device const float* _input [[buffer(0)]], device const float* gradOutput [[buffer(1)]], device const uint* count [[buffer(2)]], device float* gradInput [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float dy = gradOutput[gid];
    float x2 = (x * x);
    float x3 = (x2 * x);
    float cubeTerm = (as_type<float>(0x3d372713u) * x3);
    float inner = (x + cubeTerm);
    float u = (as_type<float>(0x3f4c422au) * inner);
    float tanhU = tanh(u);
    float tanh2 = (tanhU * tanhU);
    float sech2 = (as_type<float>(0x3f800000u) - tanh2);
    float threeCoef = as_type<float>(0x3e095d4fu);
    float derivInner = (as_type<float>(0x3f800000u) + (threeCoef * x2));
    float term2a = (as_type<float>(0x3f000000u) * x);
    float term2b = (term2a * sech2);
    float term2c = (term2b * as_type<float>(0x3f4c422au));
    float term2 = (term2c * derivInner);
    float onePlusTanh = (as_type<float>(0x3f800000u) + tanhU);
    float term1 = (as_type<float>(0x3f000000u) * onePlusTanh);
    float geluPrime = (term1 + term2);
    float dx = (dy * geluPrime);
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_metal_navatala_transformer_silu_backward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_silu_backward_f32(device const float* _input [[buffer(0)]], device const float* gradOutput [[buffer(1)]], device const uint* count [[buffer(2)]], device float* gradInput [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float dy = gradOutput[gid];
    float negX = (-x);
    float expNegX = exp(negX);
    float denom = (as_type<float>(0x3f800000u) + expNegX);
    float sigmoid = (as_type<float>(0x3f800000u) / denom);
    float oneMinusSigmoid = (as_type<float>(0x3f800000u) - sigmoid);
    float xTimesOneMinusSig = (x * oneMinusSigmoid);
    float bracketTerm = (as_type<float>(0x3f800000u) + xTimesOneMinusSig);
    float siluPrime = (sigmoid * bracketTerm);
    float dx = (dy * siluPrime);
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_metal_navatala_transformer_relu_backward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_relu_backward_f32(device const float* _input [[buffer(0)]], device const float* gradOutput [[buffer(1)]], device const uint* count [[buffer(2)]], device float* gradInput [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float dy = gradOutput[gid];
    bool isPositive = (x > as_type<float>(0x00000000u));
    float dx = ((isPositive) ? (dy) : (as_type<float>(0x00000000u)));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_metal_navatala_transformer_quick_gelu_backward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_quick_gelu_backward_f32(device const float* _input [[buffer(0)]], device const float* gradOutput [[buffer(1)]], device const uint* count [[buffer(2)]], device float* gradInput [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float dy = gradOutput[gid];
    float scaledX = (as_type<float>(0x3fd9db23u) * x);
    float negScaledX = (-scaledX);
    float expNegScaledX = exp(negScaledX);
    float denom = (as_type<float>(0x3f800000u) + expNegScaledX);
    float sigmoid = (as_type<float>(0x3f800000u) / denom);
    float oneMinusSigmoid = (as_type<float>(0x3f800000u) - sigmoid);
    float term2a = (scaledX * sigmoid);
    float term2 = (term2a * oneMinusSigmoid);
    float quickGeluPrime = (sigmoid + term2);
    float dx = (dy * quickGeluPrime);
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_metal_navatala_transformer_gelu_backward_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_gelu_backward_f16(device const half* _input [[buffer(0)]], device const half* gradOutput [[buffer(1)]], device const uint* count [[buffer(2)]], device half* gradInput [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half dy = gradOutput[gid];
    float xF32 = ((float)(x));
    float dyF32 = ((float)(dy));
    float x2 = (xF32 * xF32);
    float x3 = (x2 * xF32);
    float cubeTerm = (as_type<float>(0x3d372713u) * x3);
    float inner = (xF32 + cubeTerm);
    float u = (as_type<float>(0x3f4c422au) * inner);
    float tanhU = tanh(u);
    float tanh2 = (tanhU * tanhU);
    float sech2 = (as_type<float>(0x3f800000u) - tanh2);
    float threeCoef = as_type<float>(0x3e095d4fu);
    float derivInner = (as_type<float>(0x3f800000u) + (threeCoef * x2));
    float term2a = (as_type<float>(0x3f000000u) * xF32);
    float term2b = (term2a * sech2);
    float term2c = (term2b * as_type<float>(0x3f4c422au));
    float term2 = (term2c * derivInner);
    float onePlusTanh = (as_type<float>(0x3f800000u) + tanhU);
    float term1 = (as_type<float>(0x3f000000u) * onePlusTanh);
    float geluPrime = (term1 + term2);
    float dxF32 = (dyF32 * geluPrime);
    half dx = ((half)(dxF32));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_metal_navatala_transformer_silu_backward_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_silu_backward_f16(device const half* _input [[buffer(0)]], device const half* gradOutput [[buffer(1)]], device const uint* count [[buffer(2)]], device half* gradInput [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half dy = gradOutput[gid];
    float xF32 = ((float)(x));
    float dyF32 = ((float)(dy));
    float negX = (-xF32);
    float expNegX = exp(negX);
    float denom = (as_type<float>(0x3f800000u) + expNegX);
    float sigmoid = (as_type<float>(0x3f800000u) / denom);
    float oneMinusSigmoid = (as_type<float>(0x3f800000u) - sigmoid);
    float xTimesOneMinusSig = (xF32 * oneMinusSigmoid);
    float bracketTerm = (as_type<float>(0x3f800000u) + xTimesOneMinusSig);
    float siluPrime = (sigmoid * bracketTerm);
    float dxF32 = (dyF32 * siluPrime);
    half dx = ((half)(dxF32));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_metal_navatala_transformer_relu_backward_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_relu_backward_f16(device const half* _input [[buffer(0)]], device const half* gradOutput [[buffer(1)]], device const uint* count [[buffer(2)]], device half* gradInput [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half dy = gradOutput[gid];
    bool isPositive = (x > half(0.000000));
    half dx = ((isPositive) ? (dy) : (half(0.000000)));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_metal_navatala_transformer_quick_gelu_backward_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_quick_gelu_backward_f16(device const half* _input [[buffer(0)]], device const half* gradOutput [[buffer(1)]], device const uint* count [[buffer(2)]], device half* gradInput [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half dy = gradOutput[gid];
    float xF32 = ((float)(x));
    float dyF32 = ((float)(dy));
    float scaledX = (as_type<float>(0x3fd9db23u) * xF32);
    float negScaledX = (-scaledX);
    float expNegScaledX = exp(negScaledX);
    float denom = (as_type<float>(0x3f800000u) + expNegScaledX);
    float sigmoid = (as_type<float>(0x3f800000u) / denom);
    float oneMinusSigmoid = (as_type<float>(0x3f800000u) - sigmoid);
    float term2a = (scaledX * sigmoid);
    float term2 = (term2a * oneMinusSigmoid);
    float quickGeluPrime = (sigmoid + term2);
    float dxF32 = (dyF32 * quickGeluPrime);
    half dx = ((half)(dxF32));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_metal_navatala_transformer_rms_norm_forward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_rms_norm_forward_f32(device const float* _input [[buffer(0)]], device const float* gamma [[buffer(1)]], device const float* epsilon [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* hiddenSize [[buffer(4)]], device float* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint hs = hiddenSize[0u];
  float eps = epsilon[0u];
  threadgroup float sumSq[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  float x = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (as_type<float>(0x00000000u)));
  float xSq = (x * x);
  sumSq[lid] = ((hiddenValid) ? (xSq) : (as_type<float>(0x00000000u)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_128 = (lid < 128u);
  if (shouldReduce_sumSq_128) {
    uint neighborIdx_sumSq_128 = (lid + 128u);
    float myVal_sumSq_128 = sumSq[lid];
    float neighborVal_sumSq_128 = sumSq[neighborIdx_sumSq_128];
    float sumVal_sumSq_128 = (myVal_sumSq_128 + neighborVal_sumSq_128);
    sumSq[lid] = sumVal_sumSq_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_64 = (lid < 64u);
  if (shouldReduce_sumSq_64) {
    uint neighborIdx_sumSq_64 = (lid + 64u);
    float myVal_sumSq_64 = sumSq[lid];
    float neighborVal_sumSq_64 = sumSq[neighborIdx_sumSq_64];
    float sumVal_sumSq_64 = (myVal_sumSq_64 + neighborVal_sumSq_64);
    sumSq[lid] = sumVal_sumSq_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_32 = (lid < 32u);
  if (shouldReduce_sumSq_32) {
    uint neighborIdx_sumSq_32 = (lid + 32u);
    float myVal_sumSq_32 = sumSq[lid];
    float neighborVal_sumSq_32 = sumSq[neighborIdx_sumSq_32];
    float sumVal_sumSq_32 = (myVal_sumSq_32 + neighborVal_sumSq_32);
    sumSq[lid] = sumVal_sumSq_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_16 = (lid < 16u);
  if (shouldReduce_sumSq_16) {
    uint neighborIdx_sumSq_16 = (lid + 16u);
    float myVal_sumSq_16 = sumSq[lid];
    float neighborVal_sumSq_16 = sumSq[neighborIdx_sumSq_16];
    float sumVal_sumSq_16 = (myVal_sumSq_16 + neighborVal_sumSq_16);
    sumSq[lid] = sumVal_sumSq_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_8 = (lid < 8u);
  if (shouldReduce_sumSq_8) {
    uint neighborIdx_sumSq_8 = (lid + 8u);
    float myVal_sumSq_8 = sumSq[lid];
    float neighborVal_sumSq_8 = sumSq[neighborIdx_sumSq_8];
    float sumVal_sumSq_8 = (myVal_sumSq_8 + neighborVal_sumSq_8);
    sumSq[lid] = sumVal_sumSq_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_4 = (lid < 4u);
  if (shouldReduce_sumSq_4) {
    uint neighborIdx_sumSq_4 = (lid + 4u);
    float myVal_sumSq_4 = sumSq[lid];
    float neighborVal_sumSq_4 = sumSq[neighborIdx_sumSq_4];
    float sumVal_sumSq_4 = (myVal_sumSq_4 + neighborVal_sumSq_4);
    sumSq[lid] = sumVal_sumSq_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_2 = (lid < 2u);
  if (shouldReduce_sumSq_2) {
    uint neighborIdx_sumSq_2 = (lid + 2u);
    float myVal_sumSq_2 = sumSq[lid];
    float neighborVal_sumSq_2 = sumSq[neighborIdx_sumSq_2];
    float sumVal_sumSq_2 = (myVal_sumSq_2 + neighborVal_sumSq_2);
    sumSq[lid] = sumVal_sumSq_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_1 = (lid < 1u);
  if (shouldReduce_sumSq_1) {
    uint neighborIdx_sumSq_1 = (lid + 1u);
    float myVal_sumSq_1 = sumSq[lid];
    float neighborVal_sumSq_1 = sumSq[neighborIdx_sumSq_1];
    float sumVal_sumSq_1 = (myVal_sumSq_1 + neighborVal_sumSq_1);
    sumSq[lid] = sumVal_sumSq_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalSumSq = sumSq[0u];
  float hsF32 = ((float)(hs));
  float meanSq = (totalSumSq / hsF32);
  float meanSqEps = (meanSq + eps);
  float rms = sqrt(meanSqEps);
  if ((batchValid && hiddenValid)) {
    float g = gamma[lid];
    float xNorm = (x / rms);
    float result = (g * xNorm);
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_rms_norm_forward_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_rms_norm_forward_f16(device const half* _input [[buffer(0)]], device const half* gamma [[buffer(1)]], device const float* epsilon [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* hiddenSize [[buffer(4)]], device half* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint hs = hiddenSize[0u];
  float eps = epsilon[0u];
  threadgroup float sumSq[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  half xF16 = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (half(0.000000)));
  float x = ((float)(xF16));
  float xSq = (x * x);
  sumSq[lid] = ((hiddenValid) ? (xSq) : (as_type<float>(0x00000000u)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_128 = (lid < 128u);
  if (shouldReduce_sumSq_128) {
    uint neighborIdx_sumSq_128 = (lid + 128u);
    float myVal_sumSq_128 = sumSq[lid];
    float neighborVal_sumSq_128 = sumSq[neighborIdx_sumSq_128];
    float sumVal_sumSq_128 = (myVal_sumSq_128 + neighborVal_sumSq_128);
    sumSq[lid] = sumVal_sumSq_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_64 = (lid < 64u);
  if (shouldReduce_sumSq_64) {
    uint neighborIdx_sumSq_64 = (lid + 64u);
    float myVal_sumSq_64 = sumSq[lid];
    float neighborVal_sumSq_64 = sumSq[neighborIdx_sumSq_64];
    float sumVal_sumSq_64 = (myVal_sumSq_64 + neighborVal_sumSq_64);
    sumSq[lid] = sumVal_sumSq_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_32 = (lid < 32u);
  if (shouldReduce_sumSq_32) {
    uint neighborIdx_sumSq_32 = (lid + 32u);
    float myVal_sumSq_32 = sumSq[lid];
    float neighborVal_sumSq_32 = sumSq[neighborIdx_sumSq_32];
    float sumVal_sumSq_32 = (myVal_sumSq_32 + neighborVal_sumSq_32);
    sumSq[lid] = sumVal_sumSq_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_16 = (lid < 16u);
  if (shouldReduce_sumSq_16) {
    uint neighborIdx_sumSq_16 = (lid + 16u);
    float myVal_sumSq_16 = sumSq[lid];
    float neighborVal_sumSq_16 = sumSq[neighborIdx_sumSq_16];
    float sumVal_sumSq_16 = (myVal_sumSq_16 + neighborVal_sumSq_16);
    sumSq[lid] = sumVal_sumSq_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_8 = (lid < 8u);
  if (shouldReduce_sumSq_8) {
    uint neighborIdx_sumSq_8 = (lid + 8u);
    float myVal_sumSq_8 = sumSq[lid];
    float neighborVal_sumSq_8 = sumSq[neighborIdx_sumSq_8];
    float sumVal_sumSq_8 = (myVal_sumSq_8 + neighborVal_sumSq_8);
    sumSq[lid] = sumVal_sumSq_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_4 = (lid < 4u);
  if (shouldReduce_sumSq_4) {
    uint neighborIdx_sumSq_4 = (lid + 4u);
    float myVal_sumSq_4 = sumSq[lid];
    float neighborVal_sumSq_4 = sumSq[neighborIdx_sumSq_4];
    float sumVal_sumSq_4 = (myVal_sumSq_4 + neighborVal_sumSq_4);
    sumSq[lid] = sumVal_sumSq_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_2 = (lid < 2u);
  if (shouldReduce_sumSq_2) {
    uint neighborIdx_sumSq_2 = (lid + 2u);
    float myVal_sumSq_2 = sumSq[lid];
    float neighborVal_sumSq_2 = sumSq[neighborIdx_sumSq_2];
    float sumVal_sumSq_2 = (myVal_sumSq_2 + neighborVal_sumSq_2);
    sumSq[lid] = sumVal_sumSq_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_1 = (lid < 1u);
  if (shouldReduce_sumSq_1) {
    uint neighborIdx_sumSq_1 = (lid + 1u);
    float myVal_sumSq_1 = sumSq[lid];
    float neighborVal_sumSq_1 = sumSq[neighborIdx_sumSq_1];
    float sumVal_sumSq_1 = (myVal_sumSq_1 + neighborVal_sumSq_1);
    sumSq[lid] = sumVal_sumSq_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalSumSq = sumSq[0u];
  float hsF32 = ((float)(hs));
  float meanSq = (totalSumSq / hsF32);
  float meanSqEps = (meanSq + eps);
  float rms = sqrt(meanSqEps);
  if ((batchValid && hiddenValid)) {
    half gF16 = gamma[lid];
    float g = ((float)(gF16));
    float xNorm = (x / rms);
    float resultF32 = (g * xNorm);
    half result = ((half)(resultF32));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_layer_norm_forward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_layer_norm_forward_f32(device const float* _input [[buffer(0)]], device const float* gamma [[buffer(1)]], device const float* beta [[buffer(2)]], device const float* epsilon [[buffer(3)]], device const uint* batchSize [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device float* _output [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint hs = hiddenSize[0u];
  float eps = epsilon[0u];
  threadgroup float sumBuf[256];
  threadgroup float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  float x = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (as_type<float>(0x00000000u)));
  sumBuf[lid] = ((hiddenValid) ? (x) : (as_type<float>(0x00000000u)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sumBuf_128) {
    uint neighborIdx_sumBuf_128 = (lid + 128u);
    float myVal_sumBuf_128 = sumBuf[lid];
    float neighborVal_sumBuf_128 = sumBuf[neighborIdx_sumBuf_128];
    float sumVal_sumBuf_128 = (myVal_sumBuf_128 + neighborVal_sumBuf_128);
    sumBuf[lid] = sumVal_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sumBuf_64) {
    uint neighborIdx_sumBuf_64 = (lid + 64u);
    float myVal_sumBuf_64 = sumBuf[lid];
    float neighborVal_sumBuf_64 = sumBuf[neighborIdx_sumBuf_64];
    float sumVal_sumBuf_64 = (myVal_sumBuf_64 + neighborVal_sumBuf_64);
    sumBuf[lid] = sumVal_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sumBuf_32) {
    uint neighborIdx_sumBuf_32 = (lid + 32u);
    float myVal_sumBuf_32 = sumBuf[lid];
    float neighborVal_sumBuf_32 = sumBuf[neighborIdx_sumBuf_32];
    float sumVal_sumBuf_32 = (myVal_sumBuf_32 + neighborVal_sumBuf_32);
    sumBuf[lid] = sumVal_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sumBuf_16) {
    uint neighborIdx_sumBuf_16 = (lid + 16u);
    float myVal_sumBuf_16 = sumBuf[lid];
    float neighborVal_sumBuf_16 = sumBuf[neighborIdx_sumBuf_16];
    float sumVal_sumBuf_16 = (myVal_sumBuf_16 + neighborVal_sumBuf_16);
    sumBuf[lid] = sumVal_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sumBuf_8) {
    uint neighborIdx_sumBuf_8 = (lid + 8u);
    float myVal_sumBuf_8 = sumBuf[lid];
    float neighborVal_sumBuf_8 = sumBuf[neighborIdx_sumBuf_8];
    float sumVal_sumBuf_8 = (myVal_sumBuf_8 + neighborVal_sumBuf_8);
    sumBuf[lid] = sumVal_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sumBuf_4) {
    uint neighborIdx_sumBuf_4 = (lid + 4u);
    float myVal_sumBuf_4 = sumBuf[lid];
    float neighborVal_sumBuf_4 = sumBuf[neighborIdx_sumBuf_4];
    float sumVal_sumBuf_4 = (myVal_sumBuf_4 + neighborVal_sumBuf_4);
    sumBuf[lid] = sumVal_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sumBuf_2) {
    uint neighborIdx_sumBuf_2 = (lid + 2u);
    float myVal_sumBuf_2 = sumBuf[lid];
    float neighborVal_sumBuf_2 = sumBuf[neighborIdx_sumBuf_2];
    float sumVal_sumBuf_2 = (myVal_sumBuf_2 + neighborVal_sumBuf_2);
    sumBuf[lid] = sumVal_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sumBuf_1) {
    uint neighborIdx_sumBuf_1 = (lid + 1u);
    float myVal_sumBuf_1 = sumBuf[lid];
    float neighborVal_sumBuf_1 = sumBuf[neighborIdx_sumBuf_1];
    float sumVal_sumBuf_1 = (myVal_sumBuf_1 + neighborVal_sumBuf_1);
    sumBuf[lid] = sumVal_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalSum = sumBuf[0u];
  float hsF32 = ((float)(hs));
  float mean = (totalSum / hsF32);
  float xMinusMean = (x - mean);
  float xMinusMeanSq = (xMinusMean * xMinusMean);
  sumSqBuf[lid] = ((hiddenValid) ? (xMinusMeanSq) : (as_type<float>(0x00000000u)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_128 = (lid < 128u);
  if (shouldReduce_sumSqBuf_128) {
    uint neighborIdx_sumSqBuf_128 = (lid + 128u);
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_64 = (lid < 64u);
  if (shouldReduce_sumSqBuf_64) {
    uint neighborIdx_sumSqBuf_64 = (lid + 64u);
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_32 = (lid < 32u);
  if (shouldReduce_sumSqBuf_32) {
    uint neighborIdx_sumSqBuf_32 = (lid + 32u);
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_16 = (lid < 16u);
  if (shouldReduce_sumSqBuf_16) {
    uint neighborIdx_sumSqBuf_16 = (lid + 16u);
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_8 = (lid < 8u);
  if (shouldReduce_sumSqBuf_8) {
    uint neighborIdx_sumSqBuf_8 = (lid + 8u);
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_4 = (lid < 4u);
  if (shouldReduce_sumSqBuf_4) {
    uint neighborIdx_sumSqBuf_4 = (lid + 4u);
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_2 = (lid < 2u);
  if (shouldReduce_sumSqBuf_2) {
    uint neighborIdx_sumSqBuf_2 = (lid + 2u);
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_1 = (lid < 1u);
  if (shouldReduce_sumSqBuf_1) {
    uint neighborIdx_sumSqBuf_1 = (lid + 1u);
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalSumSq = sumSqBuf[0u];
  float variance = (totalSumSq / hsF32);
  float varEps = (variance + eps);
  float std = sqrt(varEps);
  if ((batchValid && hiddenValid)) {
    float g = gamma[lid];
    float b = beta[lid];
    float xNorm = (xMinusMean / std);
    float scaled = (g * xNorm);
    float result = (scaled + b);
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_layer_norm_forward_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_layer_norm_forward_f16(device const half* _input [[buffer(0)]], device const half* gamma [[buffer(1)]], device const half* beta [[buffer(2)]], device const float* epsilon [[buffer(3)]], device const uint* batchSize [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device half* _output [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint hs = hiddenSize[0u];
  float eps = epsilon[0u];
  threadgroup float sumBuf[256];
  threadgroup float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  half xF16 = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (half(0.000000)));
  float x = ((float)(xF16));
  sumBuf[lid] = ((hiddenValid) ? (x) : (as_type<float>(0x00000000u)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sumBuf_128) {
    uint neighborIdx_sumBuf_128 = (lid + 128u);
    float myVal_sumBuf_128 = sumBuf[lid];
    float neighborVal_sumBuf_128 = sumBuf[neighborIdx_sumBuf_128];
    float sumVal_sumBuf_128 = (myVal_sumBuf_128 + neighborVal_sumBuf_128);
    sumBuf[lid] = sumVal_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sumBuf_64) {
    uint neighborIdx_sumBuf_64 = (lid + 64u);
    float myVal_sumBuf_64 = sumBuf[lid];
    float neighborVal_sumBuf_64 = sumBuf[neighborIdx_sumBuf_64];
    float sumVal_sumBuf_64 = (myVal_sumBuf_64 + neighborVal_sumBuf_64);
    sumBuf[lid] = sumVal_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sumBuf_32) {
    uint neighborIdx_sumBuf_32 = (lid + 32u);
    float myVal_sumBuf_32 = sumBuf[lid];
    float neighborVal_sumBuf_32 = sumBuf[neighborIdx_sumBuf_32];
    float sumVal_sumBuf_32 = (myVal_sumBuf_32 + neighborVal_sumBuf_32);
    sumBuf[lid] = sumVal_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sumBuf_16) {
    uint neighborIdx_sumBuf_16 = (lid + 16u);
    float myVal_sumBuf_16 = sumBuf[lid];
    float neighborVal_sumBuf_16 = sumBuf[neighborIdx_sumBuf_16];
    float sumVal_sumBuf_16 = (myVal_sumBuf_16 + neighborVal_sumBuf_16);
    sumBuf[lid] = sumVal_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sumBuf_8) {
    uint neighborIdx_sumBuf_8 = (lid + 8u);
    float myVal_sumBuf_8 = sumBuf[lid];
    float neighborVal_sumBuf_8 = sumBuf[neighborIdx_sumBuf_8];
    float sumVal_sumBuf_8 = (myVal_sumBuf_8 + neighborVal_sumBuf_8);
    sumBuf[lid] = sumVal_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sumBuf_4) {
    uint neighborIdx_sumBuf_4 = (lid + 4u);
    float myVal_sumBuf_4 = sumBuf[lid];
    float neighborVal_sumBuf_4 = sumBuf[neighborIdx_sumBuf_4];
    float sumVal_sumBuf_4 = (myVal_sumBuf_4 + neighborVal_sumBuf_4);
    sumBuf[lid] = sumVal_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sumBuf_2) {
    uint neighborIdx_sumBuf_2 = (lid + 2u);
    float myVal_sumBuf_2 = sumBuf[lid];
    float neighborVal_sumBuf_2 = sumBuf[neighborIdx_sumBuf_2];
    float sumVal_sumBuf_2 = (myVal_sumBuf_2 + neighborVal_sumBuf_2);
    sumBuf[lid] = sumVal_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sumBuf_1) {
    uint neighborIdx_sumBuf_1 = (lid + 1u);
    float myVal_sumBuf_1 = sumBuf[lid];
    float neighborVal_sumBuf_1 = sumBuf[neighborIdx_sumBuf_1];
    float sumVal_sumBuf_1 = (myVal_sumBuf_1 + neighborVal_sumBuf_1);
    sumBuf[lid] = sumVal_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalSum = sumBuf[0u];
  float hsF32 = ((float)(hs));
  float mean = (totalSum / hsF32);
  float xMinusMean = (x - mean);
  float xMinusMeanSq = (xMinusMean * xMinusMean);
  sumSqBuf[lid] = ((hiddenValid) ? (xMinusMeanSq) : (as_type<float>(0x00000000u)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_128 = (lid < 128u);
  if (shouldReduce_sumSqBuf_128) {
    uint neighborIdx_sumSqBuf_128 = (lid + 128u);
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_64 = (lid < 64u);
  if (shouldReduce_sumSqBuf_64) {
    uint neighborIdx_sumSqBuf_64 = (lid + 64u);
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_32 = (lid < 32u);
  if (shouldReduce_sumSqBuf_32) {
    uint neighborIdx_sumSqBuf_32 = (lid + 32u);
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_16 = (lid < 16u);
  if (shouldReduce_sumSqBuf_16) {
    uint neighborIdx_sumSqBuf_16 = (lid + 16u);
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_8 = (lid < 8u);
  if (shouldReduce_sumSqBuf_8) {
    uint neighborIdx_sumSqBuf_8 = (lid + 8u);
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_4 = (lid < 4u);
  if (shouldReduce_sumSqBuf_4) {
    uint neighborIdx_sumSqBuf_4 = (lid + 4u);
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_2 = (lid < 2u);
  if (shouldReduce_sumSqBuf_2) {
    uint neighborIdx_sumSqBuf_2 = (lid + 2u);
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_1 = (lid < 1u);
  if (shouldReduce_sumSqBuf_1) {
    uint neighborIdx_sumSqBuf_1 = (lid + 1u);
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalSumSq = sumSqBuf[0u];
  float variance = (totalSumSq / hsF32);
  float varEps = (variance + eps);
  float std = sqrt(varEps);
  if ((batchValid && hiddenValid)) {
    half gF16 = gamma[lid];
    half bF16 = beta[lid];
    float g = ((float)(gF16));
    float b = ((float)(bF16));
    float xNorm = (xMinusMean / std);
    float scaled = (g * xNorm);
    float resultF32 = (scaled + b);
    half result = ((half)(resultF32));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_layer_norm_backward_f32 = R"kernel(
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

kernel void navatala_transformer_layer_norm_backward_f32(device const float* _input [[buffer(0)]], device const float* gradOutput [[buffer(1)]], device const float* gamma [[buffer(2)]], device const float* mean [[buffer(3)]], device const float* rsigma [[buffer(4)]], device const uint* batchSize [[buffer(5)]], device const uint* hiddenSize [[buffer(6)]], device float* gradInput [[buffer(7)]], device float* gradGamma [[buffer(8)]], device float* gradBeta [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint hs = hiddenSize[0u];
  threadgroup float sumDyGamma[256];
  threadgroup float sumDyGammaXhat[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  float x = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (as_type<float>(0x00000000u)));
  float dy = (((batchValid && hiddenValid)) ? (gradOutput[globalIdx]) : (as_type<float>(0x00000000u)));
  float g = ((hiddenValid) ? (gamma[lid]) : (as_type<float>(0x00000000u)));
  float mu = ((batchValid) ? (mean[batchIdx]) : (as_type<float>(0x00000000u)));
  float rs = ((batchValid) ? (rsigma[batchIdx]) : (as_type<float>(0x3f800000u)));
  float xMinusMean = (x - mu);
  float xHat = (xMinusMean * rs);
  float dyGamma = (dy * g);
  float dyGammaXhat = (dyGamma * xHat);
  sumDyGamma[lid] = ((hiddenValid) ? (dyGamma) : (as_type<float>(0x00000000u)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGamma_128 = (lid < 128u);
  if (shouldReduce_sumDyGamma_128) {
    uint neighborIdx_sumDyGamma_128 = (lid + 128u);
    float myVal_sumDyGamma_128 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_128 = sumDyGamma[neighborIdx_sumDyGamma_128];
    float sumVal_sumDyGamma_128 = (myVal_sumDyGamma_128 + neighborVal_sumDyGamma_128);
    sumDyGamma[lid] = sumVal_sumDyGamma_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGamma_64 = (lid < 64u);
  if (shouldReduce_sumDyGamma_64) {
    uint neighborIdx_sumDyGamma_64 = (lid + 64u);
    float myVal_sumDyGamma_64 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_64 = sumDyGamma[neighborIdx_sumDyGamma_64];
    float sumVal_sumDyGamma_64 = (myVal_sumDyGamma_64 + neighborVal_sumDyGamma_64);
    sumDyGamma[lid] = sumVal_sumDyGamma_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGamma_32 = (lid < 32u);
  if (shouldReduce_sumDyGamma_32) {
    uint neighborIdx_sumDyGamma_32 = (lid + 32u);
    float myVal_sumDyGamma_32 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_32 = sumDyGamma[neighborIdx_sumDyGamma_32];
    float sumVal_sumDyGamma_32 = (myVal_sumDyGamma_32 + neighborVal_sumDyGamma_32);
    sumDyGamma[lid] = sumVal_sumDyGamma_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGamma_16 = (lid < 16u);
  if (shouldReduce_sumDyGamma_16) {
    uint neighborIdx_sumDyGamma_16 = (lid + 16u);
    float myVal_sumDyGamma_16 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_16 = sumDyGamma[neighborIdx_sumDyGamma_16];
    float sumVal_sumDyGamma_16 = (myVal_sumDyGamma_16 + neighborVal_sumDyGamma_16);
    sumDyGamma[lid] = sumVal_sumDyGamma_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGamma_8 = (lid < 8u);
  if (shouldReduce_sumDyGamma_8) {
    uint neighborIdx_sumDyGamma_8 = (lid + 8u);
    float myVal_sumDyGamma_8 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_8 = sumDyGamma[neighborIdx_sumDyGamma_8];
    float sumVal_sumDyGamma_8 = (myVal_sumDyGamma_8 + neighborVal_sumDyGamma_8);
    sumDyGamma[lid] = sumVal_sumDyGamma_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGamma_4 = (lid < 4u);
  if (shouldReduce_sumDyGamma_4) {
    uint neighborIdx_sumDyGamma_4 = (lid + 4u);
    float myVal_sumDyGamma_4 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_4 = sumDyGamma[neighborIdx_sumDyGamma_4];
    float sumVal_sumDyGamma_4 = (myVal_sumDyGamma_4 + neighborVal_sumDyGamma_4);
    sumDyGamma[lid] = sumVal_sumDyGamma_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGamma_2 = (lid < 2u);
  if (shouldReduce_sumDyGamma_2) {
    uint neighborIdx_sumDyGamma_2 = (lid + 2u);
    float myVal_sumDyGamma_2 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_2 = sumDyGamma[neighborIdx_sumDyGamma_2];
    float sumVal_sumDyGamma_2 = (myVal_sumDyGamma_2 + neighborVal_sumDyGamma_2);
    sumDyGamma[lid] = sumVal_sumDyGamma_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGamma_1 = (lid < 1u);
  if (shouldReduce_sumDyGamma_1) {
    uint neighborIdx_sumDyGamma_1 = (lid + 1u);
    float myVal_sumDyGamma_1 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_1 = sumDyGamma[neighborIdx_sumDyGamma_1];
    float sumVal_sumDyGamma_1 = (myVal_sumDyGamma_1 + neighborVal_sumDyGamma_1);
    sumDyGamma[lid] = sumVal_sumDyGamma_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalDyGamma = sumDyGamma[0u];
  sumDyGammaXhat[lid] = ((hiddenValid) ? (dyGammaXhat) : (as_type<float>(0x00000000u)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_128 = (lid < 128u);
  if (shouldReduce_sumDyGammaXhat_128) {
    uint neighborIdx_sumDyGammaXhat_128 = (lid + 128u);
    float myVal_sumDyGammaXhat_128 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_128 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_128];
    float sumVal_sumDyGammaXhat_128 = (myVal_sumDyGammaXhat_128 + neighborVal_sumDyGammaXhat_128);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_64 = (lid < 64u);
  if (shouldReduce_sumDyGammaXhat_64) {
    uint neighborIdx_sumDyGammaXhat_64 = (lid + 64u);
    float myVal_sumDyGammaXhat_64 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_64 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_64];
    float sumVal_sumDyGammaXhat_64 = (myVal_sumDyGammaXhat_64 + neighborVal_sumDyGammaXhat_64);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_32 = (lid < 32u);
  if (shouldReduce_sumDyGammaXhat_32) {
    uint neighborIdx_sumDyGammaXhat_32 = (lid + 32u);
    float myVal_sumDyGammaXhat_32 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_32 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_32];
    float sumVal_sumDyGammaXhat_32 = (myVal_sumDyGammaXhat_32 + neighborVal_sumDyGammaXhat_32);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_16 = (lid < 16u);
  if (shouldReduce_sumDyGammaXhat_16) {
    uint neighborIdx_sumDyGammaXhat_16 = (lid + 16u);
    float myVal_sumDyGammaXhat_16 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_16 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_16];
    float sumVal_sumDyGammaXhat_16 = (myVal_sumDyGammaXhat_16 + neighborVal_sumDyGammaXhat_16);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_8 = (lid < 8u);
  if (shouldReduce_sumDyGammaXhat_8) {
    uint neighborIdx_sumDyGammaXhat_8 = (lid + 8u);
    float myVal_sumDyGammaXhat_8 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_8 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_8];
    float sumVal_sumDyGammaXhat_8 = (myVal_sumDyGammaXhat_8 + neighborVal_sumDyGammaXhat_8);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_4 = (lid < 4u);
  if (shouldReduce_sumDyGammaXhat_4) {
    uint neighborIdx_sumDyGammaXhat_4 = (lid + 4u);
    float myVal_sumDyGammaXhat_4 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_4 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_4];
    float sumVal_sumDyGammaXhat_4 = (myVal_sumDyGammaXhat_4 + neighborVal_sumDyGammaXhat_4);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_2 = (lid < 2u);
  if (shouldReduce_sumDyGammaXhat_2) {
    uint neighborIdx_sumDyGammaXhat_2 = (lid + 2u);
    float myVal_sumDyGammaXhat_2 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_2 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_2];
    float sumVal_sumDyGammaXhat_2 = (myVal_sumDyGammaXhat_2 + neighborVal_sumDyGammaXhat_2);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_1 = (lid < 1u);
  if (shouldReduce_sumDyGammaXhat_1) {
    uint neighborIdx_sumDyGammaXhat_1 = (lid + 1u);
    float myVal_sumDyGammaXhat_1 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_1 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_1];
    float sumVal_sumDyGammaXhat_1 = (myVal_sumDyGammaXhat_1 + neighborVal_sumDyGammaXhat_1);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalDyGammaXhat = sumDyGammaXhat[0u];
  float hsF32 = ((float)(hs));
  float meanDyGamma = (totalDyGamma / hsF32);
  float meanDyGammaXhat = (totalDyGammaXhat / hsF32);
  if ((batchValid && hiddenValid)) {
    float term1 = dyGamma;
    float term2 = meanDyGamma;
    float term3 = (xHat * meanDyGammaXhat);
    float inner = ((term1 - term2) - term3);
    float dx = (rs * inner);
    gradInput[globalIdx] = dx;
    float dyXhat = (dy * xHat);
    atomic_fetch_add_explicit((device atomic_float*)(&gradGamma[lid]), dyXhat, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_float*)(&gradBeta[lid]), dy, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_transformer_layer_norm_backward_f16 = R"kernel(
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

kernel void navatala_transformer_layer_norm_backward_f16(device const half* _input [[buffer(0)]], device const half* gradOutput [[buffer(1)]], device const half* gamma [[buffer(2)]], device const float* mean [[buffer(3)]], device const float* rsigma [[buffer(4)]], device const uint* batchSize [[buffer(5)]], device const uint* hiddenSize [[buffer(6)]], device half* gradInput [[buffer(7)]], device float* gradGamma [[buffer(8)]], device float* gradBeta [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint hs = hiddenSize[0u];
  threadgroup float sumDyGamma[256];
  threadgroup float sumDyGammaXhat[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  half xF16 = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (half(0.000000)));
  float x = ((float)(xF16));
  half dyF16 = (((batchValid && hiddenValid)) ? (gradOutput[globalIdx]) : (half(0.000000)));
  float dy = ((float)(dyF16));
  half gF16 = ((hiddenValid) ? (gamma[lid]) : (half(0.000000)));
  float g = ((float)(gF16));
  float mu = ((batchValid) ? (mean[batchIdx]) : (as_type<float>(0x00000000u)));
  float rs = ((batchValid) ? (rsigma[batchIdx]) : (as_type<float>(0x3f800000u)));
  float xMinusMean = (x - mu);
  float xHat = (xMinusMean * rs);
  float dyGamma = (dy * g);
  float dyGammaXhat = (dyGamma * xHat);
  sumDyGamma[lid] = ((hiddenValid) ? (dyGamma) : (as_type<float>(0x00000000u)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGamma_128 = (lid < 128u);
  if (shouldReduce_sumDyGamma_128) {
    uint neighborIdx_sumDyGamma_128 = (lid + 128u);
    float myVal_sumDyGamma_128 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_128 = sumDyGamma[neighborIdx_sumDyGamma_128];
    float sumVal_sumDyGamma_128 = (myVal_sumDyGamma_128 + neighborVal_sumDyGamma_128);
    sumDyGamma[lid] = sumVal_sumDyGamma_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGamma_64 = (lid < 64u);
  if (shouldReduce_sumDyGamma_64) {
    uint neighborIdx_sumDyGamma_64 = (lid + 64u);
    float myVal_sumDyGamma_64 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_64 = sumDyGamma[neighborIdx_sumDyGamma_64];
    float sumVal_sumDyGamma_64 = (myVal_sumDyGamma_64 + neighborVal_sumDyGamma_64);
    sumDyGamma[lid] = sumVal_sumDyGamma_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGamma_32 = (lid < 32u);
  if (shouldReduce_sumDyGamma_32) {
    uint neighborIdx_sumDyGamma_32 = (lid + 32u);
    float myVal_sumDyGamma_32 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_32 = sumDyGamma[neighborIdx_sumDyGamma_32];
    float sumVal_sumDyGamma_32 = (myVal_sumDyGamma_32 + neighborVal_sumDyGamma_32);
    sumDyGamma[lid] = sumVal_sumDyGamma_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGamma_16 = (lid < 16u);
  if (shouldReduce_sumDyGamma_16) {
    uint neighborIdx_sumDyGamma_16 = (lid + 16u);
    float myVal_sumDyGamma_16 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_16 = sumDyGamma[neighborIdx_sumDyGamma_16];
    float sumVal_sumDyGamma_16 = (myVal_sumDyGamma_16 + neighborVal_sumDyGamma_16);
    sumDyGamma[lid] = sumVal_sumDyGamma_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGamma_8 = (lid < 8u);
  if (shouldReduce_sumDyGamma_8) {
    uint neighborIdx_sumDyGamma_8 = (lid + 8u);
    float myVal_sumDyGamma_8 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_8 = sumDyGamma[neighborIdx_sumDyGamma_8];
    float sumVal_sumDyGamma_8 = (myVal_sumDyGamma_8 + neighborVal_sumDyGamma_8);
    sumDyGamma[lid] = sumVal_sumDyGamma_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGamma_4 = (lid < 4u);
  if (shouldReduce_sumDyGamma_4) {
    uint neighborIdx_sumDyGamma_4 = (lid + 4u);
    float myVal_sumDyGamma_4 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_4 = sumDyGamma[neighborIdx_sumDyGamma_4];
    float sumVal_sumDyGamma_4 = (myVal_sumDyGamma_4 + neighborVal_sumDyGamma_4);
    sumDyGamma[lid] = sumVal_sumDyGamma_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGamma_2 = (lid < 2u);
  if (shouldReduce_sumDyGamma_2) {
    uint neighborIdx_sumDyGamma_2 = (lid + 2u);
    float myVal_sumDyGamma_2 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_2 = sumDyGamma[neighborIdx_sumDyGamma_2];
    float sumVal_sumDyGamma_2 = (myVal_sumDyGamma_2 + neighborVal_sumDyGamma_2);
    sumDyGamma[lid] = sumVal_sumDyGamma_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGamma_1 = (lid < 1u);
  if (shouldReduce_sumDyGamma_1) {
    uint neighborIdx_sumDyGamma_1 = (lid + 1u);
    float myVal_sumDyGamma_1 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_1 = sumDyGamma[neighborIdx_sumDyGamma_1];
    float sumVal_sumDyGamma_1 = (myVal_sumDyGamma_1 + neighborVal_sumDyGamma_1);
    sumDyGamma[lid] = sumVal_sumDyGamma_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalDyGamma = sumDyGamma[0u];
  sumDyGammaXhat[lid] = ((hiddenValid) ? (dyGammaXhat) : (as_type<float>(0x00000000u)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_128 = (lid < 128u);
  if (shouldReduce_sumDyGammaXhat_128) {
    uint neighborIdx_sumDyGammaXhat_128 = (lid + 128u);
    float myVal_sumDyGammaXhat_128 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_128 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_128];
    float sumVal_sumDyGammaXhat_128 = (myVal_sumDyGammaXhat_128 + neighborVal_sumDyGammaXhat_128);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_64 = (lid < 64u);
  if (shouldReduce_sumDyGammaXhat_64) {
    uint neighborIdx_sumDyGammaXhat_64 = (lid + 64u);
    float myVal_sumDyGammaXhat_64 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_64 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_64];
    float sumVal_sumDyGammaXhat_64 = (myVal_sumDyGammaXhat_64 + neighborVal_sumDyGammaXhat_64);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_32 = (lid < 32u);
  if (shouldReduce_sumDyGammaXhat_32) {
    uint neighborIdx_sumDyGammaXhat_32 = (lid + 32u);
    float myVal_sumDyGammaXhat_32 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_32 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_32];
    float sumVal_sumDyGammaXhat_32 = (myVal_sumDyGammaXhat_32 + neighborVal_sumDyGammaXhat_32);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_16 = (lid < 16u);
  if (shouldReduce_sumDyGammaXhat_16) {
    uint neighborIdx_sumDyGammaXhat_16 = (lid + 16u);
    float myVal_sumDyGammaXhat_16 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_16 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_16];
    float sumVal_sumDyGammaXhat_16 = (myVal_sumDyGammaXhat_16 + neighborVal_sumDyGammaXhat_16);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_8 = (lid < 8u);
  if (shouldReduce_sumDyGammaXhat_8) {
    uint neighborIdx_sumDyGammaXhat_8 = (lid + 8u);
    float myVal_sumDyGammaXhat_8 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_8 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_8];
    float sumVal_sumDyGammaXhat_8 = (myVal_sumDyGammaXhat_8 + neighborVal_sumDyGammaXhat_8);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_4 = (lid < 4u);
  if (shouldReduce_sumDyGammaXhat_4) {
    uint neighborIdx_sumDyGammaXhat_4 = (lid + 4u);
    float myVal_sumDyGammaXhat_4 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_4 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_4];
    float sumVal_sumDyGammaXhat_4 = (myVal_sumDyGammaXhat_4 + neighborVal_sumDyGammaXhat_4);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_2 = (lid < 2u);
  if (shouldReduce_sumDyGammaXhat_2) {
    uint neighborIdx_sumDyGammaXhat_2 = (lid + 2u);
    float myVal_sumDyGammaXhat_2 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_2 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_2];
    float sumVal_sumDyGammaXhat_2 = (myVal_sumDyGammaXhat_2 + neighborVal_sumDyGammaXhat_2);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_1 = (lid < 1u);
  if (shouldReduce_sumDyGammaXhat_1) {
    uint neighborIdx_sumDyGammaXhat_1 = (lid + 1u);
    float myVal_sumDyGammaXhat_1 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_1 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_1];
    float sumVal_sumDyGammaXhat_1 = (myVal_sumDyGammaXhat_1 + neighborVal_sumDyGammaXhat_1);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalDyGammaXhat = sumDyGammaXhat[0u];
  float hsF32 = ((float)(hs));
  float meanDyGamma = (totalDyGamma / hsF32);
  float meanDyGammaXhat = (totalDyGammaXhat / hsF32);
  if ((batchValid && hiddenValid)) {
    float term1 = dyGamma;
    float term2 = meanDyGamma;
    float term3 = (xHat * meanDyGammaXhat);
    float inner = ((term1 - term2) - term3);
    float dxF32 = (rs * inner);
    half dx = ((half)(dxF32));
    gradInput[globalIdx] = dx;
    float dyXhat = (dy * xHat);
    atomic_fetch_add_explicit((device atomic_float*)(&gradGamma[lid]), dyXhat, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_float*)(&gradBeta[lid]), dy, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_transformer_rms_norm_backward_f32 = R"kernel(
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

kernel void navatala_transformer_rms_norm_backward_f32(device const float* _input [[buffer(0)]], device const float* gradOutput [[buffer(1)]], device const float* gamma [[buffer(2)]], device const float* rsigma [[buffer(3)]], device const uint* batchSize [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device float* gradInput [[buffer(6)]], device float* gradGamma [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint hs = hiddenSize[0u];
  threadgroup float sumDyGammaXhat[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  float x = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (as_type<float>(0x00000000u)));
  float dy = (((batchValid && hiddenValid)) ? (gradOutput[globalIdx]) : (as_type<float>(0x00000000u)));
  float g = ((hiddenValid) ? (gamma[lid]) : (as_type<float>(0x00000000u)));
  float rs = ((batchValid) ? (rsigma[batchIdx]) : (as_type<float>(0x3f800000u)));
  float xHat = (x * rs);
  float dyGamma = (dy * g);
  float dyGammaXhat = (dyGamma * xHat);
  sumDyGammaXhat[lid] = ((hiddenValid) ? (dyGammaXhat) : (as_type<float>(0x00000000u)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_128 = (lid < 128u);
  if (shouldReduce_sumDyGammaXhat_128) {
    uint neighborIdx_sumDyGammaXhat_128 = (lid + 128u);
    float myVal_sumDyGammaXhat_128 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_128 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_128];
    float sumVal_sumDyGammaXhat_128 = (myVal_sumDyGammaXhat_128 + neighborVal_sumDyGammaXhat_128);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_64 = (lid < 64u);
  if (shouldReduce_sumDyGammaXhat_64) {
    uint neighborIdx_sumDyGammaXhat_64 = (lid + 64u);
    float myVal_sumDyGammaXhat_64 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_64 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_64];
    float sumVal_sumDyGammaXhat_64 = (myVal_sumDyGammaXhat_64 + neighborVal_sumDyGammaXhat_64);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_32 = (lid < 32u);
  if (shouldReduce_sumDyGammaXhat_32) {
    uint neighborIdx_sumDyGammaXhat_32 = (lid + 32u);
    float myVal_sumDyGammaXhat_32 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_32 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_32];
    float sumVal_sumDyGammaXhat_32 = (myVal_sumDyGammaXhat_32 + neighborVal_sumDyGammaXhat_32);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_16 = (lid < 16u);
  if (shouldReduce_sumDyGammaXhat_16) {
    uint neighborIdx_sumDyGammaXhat_16 = (lid + 16u);
    float myVal_sumDyGammaXhat_16 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_16 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_16];
    float sumVal_sumDyGammaXhat_16 = (myVal_sumDyGammaXhat_16 + neighborVal_sumDyGammaXhat_16);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_8 = (lid < 8u);
  if (shouldReduce_sumDyGammaXhat_8) {
    uint neighborIdx_sumDyGammaXhat_8 = (lid + 8u);
    float myVal_sumDyGammaXhat_8 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_8 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_8];
    float sumVal_sumDyGammaXhat_8 = (myVal_sumDyGammaXhat_8 + neighborVal_sumDyGammaXhat_8);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_4 = (lid < 4u);
  if (shouldReduce_sumDyGammaXhat_4) {
    uint neighborIdx_sumDyGammaXhat_4 = (lid + 4u);
    float myVal_sumDyGammaXhat_4 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_4 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_4];
    float sumVal_sumDyGammaXhat_4 = (myVal_sumDyGammaXhat_4 + neighborVal_sumDyGammaXhat_4);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_2 = (lid < 2u);
  if (shouldReduce_sumDyGammaXhat_2) {
    uint neighborIdx_sumDyGammaXhat_2 = (lid + 2u);
    float myVal_sumDyGammaXhat_2 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_2 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_2];
    float sumVal_sumDyGammaXhat_2 = (myVal_sumDyGammaXhat_2 + neighborVal_sumDyGammaXhat_2);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_1 = (lid < 1u);
  if (shouldReduce_sumDyGammaXhat_1) {
    uint neighborIdx_sumDyGammaXhat_1 = (lid + 1u);
    float myVal_sumDyGammaXhat_1 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_1 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_1];
    float sumVal_sumDyGammaXhat_1 = (myVal_sumDyGammaXhat_1 + neighborVal_sumDyGammaXhat_1);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalDyGammaXhat = sumDyGammaXhat[0u];
  float hsF32 = ((float)(hs));
  float meanDyGammaXhat = (totalDyGammaXhat / hsF32);
  if ((batchValid && hiddenValid)) {
    float term1 = dyGamma;
    float term2 = (xHat * meanDyGammaXhat);
    float inner = (term1 - term2);
    float dx = (rs * inner);
    gradInput[globalIdx] = dx;
    float dyXhat = (dy * xHat);
    atomic_fetch_add_explicit((device atomic_float*)(&gradGamma[lid]), dyXhat, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_transformer_rms_norm_backward_f16 = R"kernel(
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

kernel void navatala_transformer_rms_norm_backward_f16(device const half* _input [[buffer(0)]], device const half* gradOutput [[buffer(1)]], device const half* gamma [[buffer(2)]], device const float* rsigma [[buffer(3)]], device const uint* batchSize [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device half* gradInput [[buffer(6)]], device float* gradGamma [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint hs = hiddenSize[0u];
  threadgroup float sumDyGammaXhat[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  half xF16 = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (half(0.000000)));
  float x = ((float)(xF16));
  half dyF16 = (((batchValid && hiddenValid)) ? (gradOutput[globalIdx]) : (half(0.000000)));
  float dy = ((float)(dyF16));
  half gF16 = ((hiddenValid) ? (gamma[lid]) : (half(0.000000)));
  float g = ((float)(gF16));
  float rs = ((batchValid) ? (rsigma[batchIdx]) : (as_type<float>(0x3f800000u)));
  float xHat = (x * rs);
  float dyGamma = (dy * g);
  float dyGammaXhat = (dyGamma * xHat);
  sumDyGammaXhat[lid] = ((hiddenValid) ? (dyGammaXhat) : (as_type<float>(0x00000000u)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_128 = (lid < 128u);
  if (shouldReduce_sumDyGammaXhat_128) {
    uint neighborIdx_sumDyGammaXhat_128 = (lid + 128u);
    float myVal_sumDyGammaXhat_128 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_128 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_128];
    float sumVal_sumDyGammaXhat_128 = (myVal_sumDyGammaXhat_128 + neighborVal_sumDyGammaXhat_128);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_64 = (lid < 64u);
  if (shouldReduce_sumDyGammaXhat_64) {
    uint neighborIdx_sumDyGammaXhat_64 = (lid + 64u);
    float myVal_sumDyGammaXhat_64 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_64 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_64];
    float sumVal_sumDyGammaXhat_64 = (myVal_sumDyGammaXhat_64 + neighborVal_sumDyGammaXhat_64);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_32 = (lid < 32u);
  if (shouldReduce_sumDyGammaXhat_32) {
    uint neighborIdx_sumDyGammaXhat_32 = (lid + 32u);
    float myVal_sumDyGammaXhat_32 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_32 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_32];
    float sumVal_sumDyGammaXhat_32 = (myVal_sumDyGammaXhat_32 + neighborVal_sumDyGammaXhat_32);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_16 = (lid < 16u);
  if (shouldReduce_sumDyGammaXhat_16) {
    uint neighborIdx_sumDyGammaXhat_16 = (lid + 16u);
    float myVal_sumDyGammaXhat_16 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_16 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_16];
    float sumVal_sumDyGammaXhat_16 = (myVal_sumDyGammaXhat_16 + neighborVal_sumDyGammaXhat_16);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_8 = (lid < 8u);
  if (shouldReduce_sumDyGammaXhat_8) {
    uint neighborIdx_sumDyGammaXhat_8 = (lid + 8u);
    float myVal_sumDyGammaXhat_8 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_8 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_8];
    float sumVal_sumDyGammaXhat_8 = (myVal_sumDyGammaXhat_8 + neighborVal_sumDyGammaXhat_8);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_4 = (lid < 4u);
  if (shouldReduce_sumDyGammaXhat_4) {
    uint neighborIdx_sumDyGammaXhat_4 = (lid + 4u);
    float myVal_sumDyGammaXhat_4 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_4 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_4];
    float sumVal_sumDyGammaXhat_4 = (myVal_sumDyGammaXhat_4 + neighborVal_sumDyGammaXhat_4);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_2 = (lid < 2u);
  if (shouldReduce_sumDyGammaXhat_2) {
    uint neighborIdx_sumDyGammaXhat_2 = (lid + 2u);
    float myVal_sumDyGammaXhat_2 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_2 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_2];
    float sumVal_sumDyGammaXhat_2 = (myVal_sumDyGammaXhat_2 + neighborVal_sumDyGammaXhat_2);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumDyGammaXhat_1 = (lid < 1u);
  if (shouldReduce_sumDyGammaXhat_1) {
    uint neighborIdx_sumDyGammaXhat_1 = (lid + 1u);
    float myVal_sumDyGammaXhat_1 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_1 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_1];
    float sumVal_sumDyGammaXhat_1 = (myVal_sumDyGammaXhat_1 + neighborVal_sumDyGammaXhat_1);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalDyGammaXhat = sumDyGammaXhat[0u];
  float hsF32 = ((float)(hs));
  float meanDyGammaXhat = (totalDyGammaXhat / hsF32);
  if ((batchValid && hiddenValid)) {
    float term1 = dyGamma;
    float term2 = (xHat * meanDyGammaXhat);
    float inner = (term1 - term2);
    float dxF32 = (rs * inner);
    half dx = ((half)(dxF32));
    gradInput[globalIdx] = dx;
    float dyXhat = (dy * xHat);
    atomic_fetch_add_explicit((device atomic_float*)(&gradGamma[lid]), dyXhat, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_transformer_layer_norm_multi_pass_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_layer_norm_multi_pass_f32(device const float* _input [[buffer(0)]], device const float* gamma [[buffer(1)]], device const float* beta [[buffer(2)]], device const float* epsilon [[buffer(3)]], device const uint* batchSize [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device float* _output [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint hs = hiddenSize[0u];
  float eps = epsilon[0u];
  threadgroup float sumBuf[256];
  threadgroup float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  uint baseIdx = (batchIdx * hs);
  float partialSum = as_type<float>(0x00000000u);
  float partialSumSq = as_type<float>(0x00000000u);
  uint iterIdx = lid;
  uint workgroupSize = 256u;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    uint globalIdx = (baseIdx + iterIdx);
    float xVal = ((batchValid) ? (_input[globalIdx]) : (as_type<float>(0x00000000u)));
    partialSum = (partialSum + xVal);
    float xSq = (xVal * xVal);
    partialSumSq = (partialSumSq + xSq);
    iterIdx = (iterIdx + workgroupSize);
  }
  sumBuf[lid] = partialSum;
  sumSqBuf[lid] = partialSumSq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sumBuf_128) {
    uint neighborIdx_sumBuf_128 = (lid + 128u);
    float myVal_sumBuf_128 = sumBuf[lid];
    float neighborVal_sumBuf_128 = sumBuf[neighborIdx_sumBuf_128];
    float sumVal_sumBuf_128 = (myVal_sumBuf_128 + neighborVal_sumBuf_128);
    sumBuf[lid] = sumVal_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sumBuf_64) {
    uint neighborIdx_sumBuf_64 = (lid + 64u);
    float myVal_sumBuf_64 = sumBuf[lid];
    float neighborVal_sumBuf_64 = sumBuf[neighborIdx_sumBuf_64];
    float sumVal_sumBuf_64 = (myVal_sumBuf_64 + neighborVal_sumBuf_64);
    sumBuf[lid] = sumVal_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sumBuf_32) {
    uint neighborIdx_sumBuf_32 = (lid + 32u);
    float myVal_sumBuf_32 = sumBuf[lid];
    float neighborVal_sumBuf_32 = sumBuf[neighborIdx_sumBuf_32];
    float sumVal_sumBuf_32 = (myVal_sumBuf_32 + neighborVal_sumBuf_32);
    sumBuf[lid] = sumVal_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sumBuf_16) {
    uint neighborIdx_sumBuf_16 = (lid + 16u);
    float myVal_sumBuf_16 = sumBuf[lid];
    float neighborVal_sumBuf_16 = sumBuf[neighborIdx_sumBuf_16];
    float sumVal_sumBuf_16 = (myVal_sumBuf_16 + neighborVal_sumBuf_16);
    sumBuf[lid] = sumVal_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sumBuf_8) {
    uint neighborIdx_sumBuf_8 = (lid + 8u);
    float myVal_sumBuf_8 = sumBuf[lid];
    float neighborVal_sumBuf_8 = sumBuf[neighborIdx_sumBuf_8];
    float sumVal_sumBuf_8 = (myVal_sumBuf_8 + neighborVal_sumBuf_8);
    sumBuf[lid] = sumVal_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sumBuf_4) {
    uint neighborIdx_sumBuf_4 = (lid + 4u);
    float myVal_sumBuf_4 = sumBuf[lid];
    float neighborVal_sumBuf_4 = sumBuf[neighborIdx_sumBuf_4];
    float sumVal_sumBuf_4 = (myVal_sumBuf_4 + neighborVal_sumBuf_4);
    sumBuf[lid] = sumVal_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sumBuf_2) {
    uint neighborIdx_sumBuf_2 = (lid + 2u);
    float myVal_sumBuf_2 = sumBuf[lid];
    float neighborVal_sumBuf_2 = sumBuf[neighborIdx_sumBuf_2];
    float sumVal_sumBuf_2 = (myVal_sumBuf_2 + neighborVal_sumBuf_2);
    sumBuf[lid] = sumVal_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sumBuf_1) {
    uint neighborIdx_sumBuf_1 = (lid + 1u);
    float myVal_sumBuf_1 = sumBuf[lid];
    float neighborVal_sumBuf_1 = sumBuf[neighborIdx_sumBuf_1];
    float sumVal_sumBuf_1 = (myVal_sumBuf_1 + neighborVal_sumBuf_1);
    sumBuf[lid] = sumVal_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalSum = sumBuf[0u];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_128 = (lid < 128u);
  if (shouldReduce_sumSqBuf_128) {
    uint neighborIdx_sumSqBuf_128 = (lid + 128u);
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_64 = (lid < 64u);
  if (shouldReduce_sumSqBuf_64) {
    uint neighborIdx_sumSqBuf_64 = (lid + 64u);
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_32 = (lid < 32u);
  if (shouldReduce_sumSqBuf_32) {
    uint neighborIdx_sumSqBuf_32 = (lid + 32u);
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_16 = (lid < 16u);
  if (shouldReduce_sumSqBuf_16) {
    uint neighborIdx_sumSqBuf_16 = (lid + 16u);
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_8 = (lid < 8u);
  if (shouldReduce_sumSqBuf_8) {
    uint neighborIdx_sumSqBuf_8 = (lid + 8u);
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_4 = (lid < 4u);
  if (shouldReduce_sumSqBuf_4) {
    uint neighborIdx_sumSqBuf_4 = (lid + 4u);
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_2 = (lid < 2u);
  if (shouldReduce_sumSqBuf_2) {
    uint neighborIdx_sumSqBuf_2 = (lid + 2u);
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_1 = (lid < 1u);
  if (shouldReduce_sumSqBuf_1) {
    uint neighborIdx_sumSqBuf_1 = (lid + 1u);
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalSumSq = sumSqBuf[0u];
  float hsF32 = ((float)(hs));
  float mean = (totalSum / hsF32);
  float meanSq = (mean * mean);
  float avgSumSq = (totalSumSq / hsF32);
  float variance = (avgSumSq - meanSq);
  float varEps = (variance + eps);
  float std = sqrt(varEps);
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    if (batchValid) {
      uint globalIdx2 = (baseIdx + iterIdx);
      float xVal2 = _input[globalIdx2];
      float g = gamma[iterIdx];
      float b = beta[iterIdx];
      float xNorm = ((xVal2 - mean) / std);
      float scaled = (g * xNorm);
      float result = (scaled + b);
      _output[globalIdx2] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}

)kernel";
const char* k_metal_navatala_transformer_layer_norm_multi_pass_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_layer_norm_multi_pass_f16(device const half* _input [[buffer(0)]], device const half* gamma [[buffer(1)]], device const half* beta [[buffer(2)]], device const float* epsilon [[buffer(3)]], device const uint* batchSize [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device half* _output [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint hs = hiddenSize[0u];
  float eps = epsilon[0u];
  threadgroup float sumBuf[256];
  threadgroup float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  uint baseIdx = (batchIdx * hs);
  float partialSum = as_type<float>(0x00000000u);
  float partialSumSq = as_type<float>(0x00000000u);
  uint iterIdx = lid;
  uint workgroupSize = 256u;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    uint globalIdx = (baseIdx + iterIdx);
    half xF16 = ((batchValid) ? (_input[globalIdx]) : (half(0.000000)));
    float xVal = ((float)(xF16));
    partialSum = (partialSum + xVal);
    float xSq = (xVal * xVal);
    partialSumSq = (partialSumSq + xSq);
    iterIdx = (iterIdx + workgroupSize);
  }
  sumBuf[lid] = partialSum;
  sumSqBuf[lid] = partialSumSq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sumBuf_128) {
    uint neighborIdx_sumBuf_128 = (lid + 128u);
    float myVal_sumBuf_128 = sumBuf[lid];
    float neighborVal_sumBuf_128 = sumBuf[neighborIdx_sumBuf_128];
    float sumVal_sumBuf_128 = (myVal_sumBuf_128 + neighborVal_sumBuf_128);
    sumBuf[lid] = sumVal_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sumBuf_64) {
    uint neighborIdx_sumBuf_64 = (lid + 64u);
    float myVal_sumBuf_64 = sumBuf[lid];
    float neighborVal_sumBuf_64 = sumBuf[neighborIdx_sumBuf_64];
    float sumVal_sumBuf_64 = (myVal_sumBuf_64 + neighborVal_sumBuf_64);
    sumBuf[lid] = sumVal_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sumBuf_32) {
    uint neighborIdx_sumBuf_32 = (lid + 32u);
    float myVal_sumBuf_32 = sumBuf[lid];
    float neighborVal_sumBuf_32 = sumBuf[neighborIdx_sumBuf_32];
    float sumVal_sumBuf_32 = (myVal_sumBuf_32 + neighborVal_sumBuf_32);
    sumBuf[lid] = sumVal_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sumBuf_16) {
    uint neighborIdx_sumBuf_16 = (lid + 16u);
    float myVal_sumBuf_16 = sumBuf[lid];
    float neighborVal_sumBuf_16 = sumBuf[neighborIdx_sumBuf_16];
    float sumVal_sumBuf_16 = (myVal_sumBuf_16 + neighborVal_sumBuf_16);
    sumBuf[lid] = sumVal_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sumBuf_8) {
    uint neighborIdx_sumBuf_8 = (lid + 8u);
    float myVal_sumBuf_8 = sumBuf[lid];
    float neighborVal_sumBuf_8 = sumBuf[neighborIdx_sumBuf_8];
    float sumVal_sumBuf_8 = (myVal_sumBuf_8 + neighborVal_sumBuf_8);
    sumBuf[lid] = sumVal_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sumBuf_4) {
    uint neighborIdx_sumBuf_4 = (lid + 4u);
    float myVal_sumBuf_4 = sumBuf[lid];
    float neighborVal_sumBuf_4 = sumBuf[neighborIdx_sumBuf_4];
    float sumVal_sumBuf_4 = (myVal_sumBuf_4 + neighborVal_sumBuf_4);
    sumBuf[lid] = sumVal_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sumBuf_2) {
    uint neighborIdx_sumBuf_2 = (lid + 2u);
    float myVal_sumBuf_2 = sumBuf[lid];
    float neighborVal_sumBuf_2 = sumBuf[neighborIdx_sumBuf_2];
    float sumVal_sumBuf_2 = (myVal_sumBuf_2 + neighborVal_sumBuf_2);
    sumBuf[lid] = sumVal_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sumBuf_1) {
    uint neighborIdx_sumBuf_1 = (lid + 1u);
    float myVal_sumBuf_1 = sumBuf[lid];
    float neighborVal_sumBuf_1 = sumBuf[neighborIdx_sumBuf_1];
    float sumVal_sumBuf_1 = (myVal_sumBuf_1 + neighborVal_sumBuf_1);
    sumBuf[lid] = sumVal_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalSum = sumBuf[0u];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_128 = (lid < 128u);
  if (shouldReduce_sumSqBuf_128) {
    uint neighborIdx_sumSqBuf_128 = (lid + 128u);
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_64 = (lid < 64u);
  if (shouldReduce_sumSqBuf_64) {
    uint neighborIdx_sumSqBuf_64 = (lid + 64u);
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_32 = (lid < 32u);
  if (shouldReduce_sumSqBuf_32) {
    uint neighborIdx_sumSqBuf_32 = (lid + 32u);
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_16 = (lid < 16u);
  if (shouldReduce_sumSqBuf_16) {
    uint neighborIdx_sumSqBuf_16 = (lid + 16u);
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_8 = (lid < 8u);
  if (shouldReduce_sumSqBuf_8) {
    uint neighborIdx_sumSqBuf_8 = (lid + 8u);
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_4 = (lid < 4u);
  if (shouldReduce_sumSqBuf_4) {
    uint neighborIdx_sumSqBuf_4 = (lid + 4u);
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_2 = (lid < 2u);
  if (shouldReduce_sumSqBuf_2) {
    uint neighborIdx_sumSqBuf_2 = (lid + 2u);
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_1 = (lid < 1u);
  if (shouldReduce_sumSqBuf_1) {
    uint neighborIdx_sumSqBuf_1 = (lid + 1u);
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalSumSq = sumSqBuf[0u];
  float hsF32 = ((float)(hs));
  float mean = (totalSum / hsF32);
  float meanSq = (mean * mean);
  float avgSumSq = (totalSumSq / hsF32);
  float variance = (avgSumSq - meanSq);
  float varEps = (variance + eps);
  float std = sqrt(varEps);
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    if (batchValid) {
      uint globalIdx2 = (baseIdx + iterIdx);
      half xF16_2 = _input[globalIdx2];
      float xVal2 = ((float)(xF16_2));
      half gF16 = gamma[iterIdx];
      half bF16 = beta[iterIdx];
      float g = ((float)(gF16));
      float b = ((float)(bF16));
      float xNorm = ((xVal2 - mean) / std);
      float scaled = (g * xNorm);
      float resultF32 = (scaled + b);
      half result = ((half)(resultF32));
      _output[globalIdx2] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}

)kernel";
const char* k_metal_navatala_transformer_rms_norm_multi_pass_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_rms_norm_multi_pass_f32(device const float* _input [[buffer(0)]], device const float* gamma [[buffer(1)]], device const float* epsilon [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* hiddenSize [[buffer(4)]], device float* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint hs = hiddenSize[0u];
  float eps = epsilon[0u];
  threadgroup float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  uint baseIdx = (batchIdx * hs);
  float partialSumSq = as_type<float>(0x00000000u);
  uint iterIdx = lid;
  uint workgroupSize = 256u;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    uint globalIdx = (baseIdx + iterIdx);
    float xVal = ((batchValid) ? (_input[globalIdx]) : (as_type<float>(0x00000000u)));
    float xSq = (xVal * xVal);
    partialSumSq = (partialSumSq + xSq);
    iterIdx = (iterIdx + workgroupSize);
  }
  sumSqBuf[lid] = partialSumSq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_128 = (lid < 128u);
  if (shouldReduce_sumSqBuf_128) {
    uint neighborIdx_sumSqBuf_128 = (lid + 128u);
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_64 = (lid < 64u);
  if (shouldReduce_sumSqBuf_64) {
    uint neighborIdx_sumSqBuf_64 = (lid + 64u);
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_32 = (lid < 32u);
  if (shouldReduce_sumSqBuf_32) {
    uint neighborIdx_sumSqBuf_32 = (lid + 32u);
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_16 = (lid < 16u);
  if (shouldReduce_sumSqBuf_16) {
    uint neighborIdx_sumSqBuf_16 = (lid + 16u);
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_8 = (lid < 8u);
  if (shouldReduce_sumSqBuf_8) {
    uint neighborIdx_sumSqBuf_8 = (lid + 8u);
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_4 = (lid < 4u);
  if (shouldReduce_sumSqBuf_4) {
    uint neighborIdx_sumSqBuf_4 = (lid + 4u);
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_2 = (lid < 2u);
  if (shouldReduce_sumSqBuf_2) {
    uint neighborIdx_sumSqBuf_2 = (lid + 2u);
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_1 = (lid < 1u);
  if (shouldReduce_sumSqBuf_1) {
    uint neighborIdx_sumSqBuf_1 = (lid + 1u);
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalSumSq = sumSqBuf[0u];
  float hsF32 = ((float)(hs));
  float meanSq = (totalSumSq / hsF32);
  float meanSqEps = (meanSq + eps);
  float rms = sqrt(meanSqEps);
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    if (batchValid) {
      uint globalIdx2 = (baseIdx + iterIdx);
      float xVal2 = _input[globalIdx2];
      float g = gamma[iterIdx];
      float xNorm = (xVal2 / rms);
      float result = (g * xNorm);
      _output[globalIdx2] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}

)kernel";
const char* k_metal_navatala_transformer_rms_norm_multi_pass_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_rms_norm_multi_pass_f16(device const half* _input [[buffer(0)]], device const half* gamma [[buffer(1)]], device const float* epsilon [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* hiddenSize [[buffer(4)]], device half* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint hs = hiddenSize[0u];
  float eps = epsilon[0u];
  threadgroup float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  uint baseIdx = (batchIdx * hs);
  float partialSumSq = as_type<float>(0x00000000u);
  uint iterIdx = lid;
  uint workgroupSize = 256u;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    uint globalIdx = (baseIdx + iterIdx);
    half xF16 = ((batchValid) ? (_input[globalIdx]) : (half(0.000000)));
    float xVal = ((float)(xF16));
    float xSq = (xVal * xVal);
    partialSumSq = (partialSumSq + xSq);
    iterIdx = (iterIdx + workgroupSize);
  }
  sumSqBuf[lid] = partialSumSq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_128 = (lid < 128u);
  if (shouldReduce_sumSqBuf_128) {
    uint neighborIdx_sumSqBuf_128 = (lid + 128u);
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_64 = (lid < 64u);
  if (shouldReduce_sumSqBuf_64) {
    uint neighborIdx_sumSqBuf_64 = (lid + 64u);
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_32 = (lid < 32u);
  if (shouldReduce_sumSqBuf_32) {
    uint neighborIdx_sumSqBuf_32 = (lid + 32u);
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_16 = (lid < 16u);
  if (shouldReduce_sumSqBuf_16) {
    uint neighborIdx_sumSqBuf_16 = (lid + 16u);
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_8 = (lid < 8u);
  if (shouldReduce_sumSqBuf_8) {
    uint neighborIdx_sumSqBuf_8 = (lid + 8u);
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_4 = (lid < 4u);
  if (shouldReduce_sumSqBuf_4) {
    uint neighborIdx_sumSqBuf_4 = (lid + 4u);
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_2 = (lid < 2u);
  if (shouldReduce_sumSqBuf_2) {
    uint neighborIdx_sumSqBuf_2 = (lid + 2u);
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSqBuf_1 = (lid < 1u);
  if (shouldReduce_sumSqBuf_1) {
    uint neighborIdx_sumSqBuf_1 = (lid + 1u);
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float totalSumSq = sumSqBuf[0u];
  float hsF32 = ((float)(hs));
  float meanSq = (totalSumSq / hsF32);
  float meanSqEps = (meanSq + eps);
  float rms = sqrt(meanSqEps);
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    if (batchValid) {
      uint globalIdx2 = (baseIdx + iterIdx);
      half xF16_2 = _input[globalIdx2];
      float xVal2 = ((float)(xF16_2));
      half gF16 = gamma[iterIdx];
      float g = ((float)(gF16));
      float xNorm = (xVal2 / rms);
      float resultF32 = (g * xNorm);
      half result = ((half)(resultF32));
      _output[globalIdx2] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}

)kernel";
const char* k_metal_navatala_transformer_softmax_forward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_softmax_forward_f32(device const float* _input [[buffer(0)]], device const uint* batchSize [[buffer(1)]], device const uint* seqLen [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  threadgroup float maxBuf[256];
  threadgroup float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  uint globalIdx = ((batchIdx * sl) + lid);
  float x = (((batchValid && seqValid)) ? (_input[globalIdx]) : (as_type<float>(0xf149f2cau)));
  maxBuf[lid] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_128 = (lid < 128u);
  if (shouldReduce_max_maxBuf_128) {
    uint neighborIdx_max_maxBuf_128 = (lid + 128u);
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_64 = (lid < 64u);
  if (shouldReduce_max_maxBuf_64) {
    uint neighborIdx_max_maxBuf_64 = (lid + 64u);
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_32 = (lid < 32u);
  if (shouldReduce_max_maxBuf_32) {
    uint neighborIdx_max_maxBuf_32 = (lid + 32u);
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_16 = (lid < 16u);
  if (shouldReduce_max_maxBuf_16) {
    uint neighborIdx_max_maxBuf_16 = (lid + 16u);
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_8 = (lid < 8u);
  if (shouldReduce_max_maxBuf_8) {
    uint neighborIdx_max_maxBuf_8 = (lid + 8u);
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_4 = (lid < 4u);
  if (shouldReduce_max_maxBuf_4) {
    uint neighborIdx_max_maxBuf_4 = (lid + 4u);
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_2 = (lid < 2u);
  if (shouldReduce_max_maxBuf_2) {
    uint neighborIdx_max_maxBuf_2 = (lid + 2u);
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_1 = (lid < 1u);
  if (shouldReduce_max_maxBuf_1) {
    uint neighborIdx_max_maxBuf_1 = (lid + 1u);
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxVal = maxBuf[0u];
  float xShifted = (x - maxVal);
  float expVal = exp(xShifted);
  float expValMasked = ((seqValid) ? (expVal) : (as_type<float>(0x00000000u)));
  sumBuf[lid] = expValMasked;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_sumBuf_128) {
    uint neighborIdx_sum_sumBuf_128 = (lid + 128u);
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_sumBuf_64) {
    uint neighborIdx_sum_sumBuf_64 = (lid + 64u);
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_sumBuf_32) {
    uint neighborIdx_sum_sumBuf_32 = (lid + 32u);
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_sumBuf_16) {
    uint neighborIdx_sum_sumBuf_16 = (lid + 16u);
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_sumBuf_8) {
    uint neighborIdx_sum_sumBuf_8 = (lid + 8u);
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_sumBuf_4) {
    uint neighborIdx_sum_sumBuf_4 = (lid + 4u);
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_sumBuf_2) {
    uint neighborIdx_sum_sumBuf_2 = (lid + 2u);
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_sumBuf_1) {
    uint neighborIdx_sum_sumBuf_1 = (lid + 1u);
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = sumBuf[0u];
  if ((batchValid && seqValid)) {
    float result = (expVal / sumExp);
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_softmax_forward_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_softmax_forward_f16(device const half* _input [[buffer(0)]], device const uint* batchSize [[buffer(1)]], device const uint* seqLen [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  threadgroup float maxBuf[256];
  threadgroup float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  uint globalIdx = ((batchIdx * sl) + lid);
  half xF16 = (((batchValid && seqValid)) ? (_input[globalIdx]) : (half(0.000000)));
  float x = ((seqValid) ? (((float)(xF16))) : (as_type<float>(0xf149f2cau)));
  maxBuf[lid] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_128 = (lid < 128u);
  if (shouldReduce_max_maxBuf_128) {
    uint neighborIdx_max_maxBuf_128 = (lid + 128u);
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_64 = (lid < 64u);
  if (shouldReduce_max_maxBuf_64) {
    uint neighborIdx_max_maxBuf_64 = (lid + 64u);
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_32 = (lid < 32u);
  if (shouldReduce_max_maxBuf_32) {
    uint neighborIdx_max_maxBuf_32 = (lid + 32u);
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_16 = (lid < 16u);
  if (shouldReduce_max_maxBuf_16) {
    uint neighborIdx_max_maxBuf_16 = (lid + 16u);
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_8 = (lid < 8u);
  if (shouldReduce_max_maxBuf_8) {
    uint neighborIdx_max_maxBuf_8 = (lid + 8u);
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_4 = (lid < 4u);
  if (shouldReduce_max_maxBuf_4) {
    uint neighborIdx_max_maxBuf_4 = (lid + 4u);
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_2 = (lid < 2u);
  if (shouldReduce_max_maxBuf_2) {
    uint neighborIdx_max_maxBuf_2 = (lid + 2u);
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_1 = (lid < 1u);
  if (shouldReduce_max_maxBuf_1) {
    uint neighborIdx_max_maxBuf_1 = (lid + 1u);
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxVal = maxBuf[0u];
  float xShifted = (x - maxVal);
  float expVal = exp(xShifted);
  float expValMasked = ((seqValid) ? (expVal) : (as_type<float>(0x00000000u)));
  sumBuf[lid] = expValMasked;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_sumBuf_128) {
    uint neighborIdx_sum_sumBuf_128 = (lid + 128u);
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_sumBuf_64) {
    uint neighborIdx_sum_sumBuf_64 = (lid + 64u);
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_sumBuf_32) {
    uint neighborIdx_sum_sumBuf_32 = (lid + 32u);
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_sumBuf_16) {
    uint neighborIdx_sum_sumBuf_16 = (lid + 16u);
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_sumBuf_8) {
    uint neighborIdx_sum_sumBuf_8 = (lid + 8u);
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_sumBuf_4) {
    uint neighborIdx_sum_sumBuf_4 = (lid + 4u);
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_sumBuf_2) {
    uint neighborIdx_sum_sumBuf_2 = (lid + 2u);
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_sumBuf_1) {
    uint neighborIdx_sum_sumBuf_1 = (lid + 1u);
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = sumBuf[0u];
  if ((batchValid && seqValid)) {
    float resultF32 = (expVal / sumExp);
    half result = ((half)(resultF32));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_softmax_backward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_softmax_backward_f32(device const float* _output [[buffer(0)]], device const float* gradOutput [[buffer(1)]], device const uint* batchSize [[buffer(2)]], device const uint* seqLen [[buffer(3)]], device float* gradInput [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  threadgroup float dotBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  uint globalIdx = ((batchIdx * sl) + lid);
  float y = (((batchValid && seqValid)) ? (_output[globalIdx]) : (as_type<float>(0x00000000u)));
  float dy = (((batchValid && seqValid)) ? (gradOutput[globalIdx]) : (as_type<float>(0x00000000u)));
  float yDy = (y * dy);
  dotBuf[lid] = yDy;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_128 = (lid < 128u);
  if (shouldReduce_sum_dotBuf_128) {
    uint neighborIdx_sum_dotBuf_128 = (lid + 128u);
    float myVal_sum_dotBuf_128 = dotBuf[lid];
    float neighborVal_sum_dotBuf_128 = dotBuf[neighborIdx_sum_dotBuf_128];
    float sumVal_sum_dotBuf_128 = (myVal_sum_dotBuf_128 + neighborVal_sum_dotBuf_128);
    dotBuf[lid] = sumVal_sum_dotBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_64 = (lid < 64u);
  if (shouldReduce_sum_dotBuf_64) {
    uint neighborIdx_sum_dotBuf_64 = (lid + 64u);
    float myVal_sum_dotBuf_64 = dotBuf[lid];
    float neighborVal_sum_dotBuf_64 = dotBuf[neighborIdx_sum_dotBuf_64];
    float sumVal_sum_dotBuf_64 = (myVal_sum_dotBuf_64 + neighborVal_sum_dotBuf_64);
    dotBuf[lid] = sumVal_sum_dotBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_32 = (lid < 32u);
  if (shouldReduce_sum_dotBuf_32) {
    uint neighborIdx_sum_dotBuf_32 = (lid + 32u);
    float myVal_sum_dotBuf_32 = dotBuf[lid];
    float neighborVal_sum_dotBuf_32 = dotBuf[neighborIdx_sum_dotBuf_32];
    float sumVal_sum_dotBuf_32 = (myVal_sum_dotBuf_32 + neighborVal_sum_dotBuf_32);
    dotBuf[lid] = sumVal_sum_dotBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_16 = (lid < 16u);
  if (shouldReduce_sum_dotBuf_16) {
    uint neighborIdx_sum_dotBuf_16 = (lid + 16u);
    float myVal_sum_dotBuf_16 = dotBuf[lid];
    float neighborVal_sum_dotBuf_16 = dotBuf[neighborIdx_sum_dotBuf_16];
    float sumVal_sum_dotBuf_16 = (myVal_sum_dotBuf_16 + neighborVal_sum_dotBuf_16);
    dotBuf[lid] = sumVal_sum_dotBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_8 = (lid < 8u);
  if (shouldReduce_sum_dotBuf_8) {
    uint neighborIdx_sum_dotBuf_8 = (lid + 8u);
    float myVal_sum_dotBuf_8 = dotBuf[lid];
    float neighborVal_sum_dotBuf_8 = dotBuf[neighborIdx_sum_dotBuf_8];
    float sumVal_sum_dotBuf_8 = (myVal_sum_dotBuf_8 + neighborVal_sum_dotBuf_8);
    dotBuf[lid] = sumVal_sum_dotBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_4 = (lid < 4u);
  if (shouldReduce_sum_dotBuf_4) {
    uint neighborIdx_sum_dotBuf_4 = (lid + 4u);
    float myVal_sum_dotBuf_4 = dotBuf[lid];
    float neighborVal_sum_dotBuf_4 = dotBuf[neighborIdx_sum_dotBuf_4];
    float sumVal_sum_dotBuf_4 = (myVal_sum_dotBuf_4 + neighborVal_sum_dotBuf_4);
    dotBuf[lid] = sumVal_sum_dotBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_2 = (lid < 2u);
  if (shouldReduce_sum_dotBuf_2) {
    uint neighborIdx_sum_dotBuf_2 = (lid + 2u);
    float myVal_sum_dotBuf_2 = dotBuf[lid];
    float neighborVal_sum_dotBuf_2 = dotBuf[neighborIdx_sum_dotBuf_2];
    float sumVal_sum_dotBuf_2 = (myVal_sum_dotBuf_2 + neighborVal_sum_dotBuf_2);
    dotBuf[lid] = sumVal_sum_dotBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_1 = (lid < 1u);
  if (shouldReduce_sum_dotBuf_1) {
    uint neighborIdx_sum_dotBuf_1 = (lid + 1u);
    float myVal_sum_dotBuf_1 = dotBuf[lid];
    float neighborVal_sum_dotBuf_1 = dotBuf[neighborIdx_sum_dotBuf_1];
    float sumVal_sum_dotBuf_1 = (myVal_sum_dotBuf_1 + neighborVal_sum_dotBuf_1);
    dotBuf[lid] = sumVal_sum_dotBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float dot = dotBuf[0u];
  if ((batchValid && seqValid)) {
    float dyMinusDot = (dy - dot);
    float dx = (y * dyMinusDot);
    gradInput[globalIdx] = dx;
  }
}

)kernel";
const char* k_metal_navatala_transformer_softmax_backward_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_softmax_backward_f16(device const half* _output [[buffer(0)]], device const half* gradOutput [[buffer(1)]], device const uint* batchSize [[buffer(2)]], device const uint* seqLen [[buffer(3)]], device half* gradInput [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  threadgroup float dotBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  uint globalIdx = ((batchIdx * sl) + lid);
  half yF16 = (((batchValid && seqValid)) ? (_output[globalIdx]) : (half(0.000000)));
  half dyF16 = (((batchValid && seqValid)) ? (gradOutput[globalIdx]) : (half(0.000000)));
  float y = ((float)(yF16));
  float dy = ((float)(dyF16));
  float yDy = (y * dy);
  dotBuf[lid] = yDy;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_128 = (lid < 128u);
  if (shouldReduce_sum_dotBuf_128) {
    uint neighborIdx_sum_dotBuf_128 = (lid + 128u);
    float myVal_sum_dotBuf_128 = dotBuf[lid];
    float neighborVal_sum_dotBuf_128 = dotBuf[neighborIdx_sum_dotBuf_128];
    float sumVal_sum_dotBuf_128 = (myVal_sum_dotBuf_128 + neighborVal_sum_dotBuf_128);
    dotBuf[lid] = sumVal_sum_dotBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_64 = (lid < 64u);
  if (shouldReduce_sum_dotBuf_64) {
    uint neighborIdx_sum_dotBuf_64 = (lid + 64u);
    float myVal_sum_dotBuf_64 = dotBuf[lid];
    float neighborVal_sum_dotBuf_64 = dotBuf[neighborIdx_sum_dotBuf_64];
    float sumVal_sum_dotBuf_64 = (myVal_sum_dotBuf_64 + neighborVal_sum_dotBuf_64);
    dotBuf[lid] = sumVal_sum_dotBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_32 = (lid < 32u);
  if (shouldReduce_sum_dotBuf_32) {
    uint neighborIdx_sum_dotBuf_32 = (lid + 32u);
    float myVal_sum_dotBuf_32 = dotBuf[lid];
    float neighborVal_sum_dotBuf_32 = dotBuf[neighborIdx_sum_dotBuf_32];
    float sumVal_sum_dotBuf_32 = (myVal_sum_dotBuf_32 + neighborVal_sum_dotBuf_32);
    dotBuf[lid] = sumVal_sum_dotBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_16 = (lid < 16u);
  if (shouldReduce_sum_dotBuf_16) {
    uint neighborIdx_sum_dotBuf_16 = (lid + 16u);
    float myVal_sum_dotBuf_16 = dotBuf[lid];
    float neighborVal_sum_dotBuf_16 = dotBuf[neighborIdx_sum_dotBuf_16];
    float sumVal_sum_dotBuf_16 = (myVal_sum_dotBuf_16 + neighborVal_sum_dotBuf_16);
    dotBuf[lid] = sumVal_sum_dotBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_8 = (lid < 8u);
  if (shouldReduce_sum_dotBuf_8) {
    uint neighborIdx_sum_dotBuf_8 = (lid + 8u);
    float myVal_sum_dotBuf_8 = dotBuf[lid];
    float neighborVal_sum_dotBuf_8 = dotBuf[neighborIdx_sum_dotBuf_8];
    float sumVal_sum_dotBuf_8 = (myVal_sum_dotBuf_8 + neighborVal_sum_dotBuf_8);
    dotBuf[lid] = sumVal_sum_dotBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_4 = (lid < 4u);
  if (shouldReduce_sum_dotBuf_4) {
    uint neighborIdx_sum_dotBuf_4 = (lid + 4u);
    float myVal_sum_dotBuf_4 = dotBuf[lid];
    float neighborVal_sum_dotBuf_4 = dotBuf[neighborIdx_sum_dotBuf_4];
    float sumVal_sum_dotBuf_4 = (myVal_sum_dotBuf_4 + neighborVal_sum_dotBuf_4);
    dotBuf[lid] = sumVal_sum_dotBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_2 = (lid < 2u);
  if (shouldReduce_sum_dotBuf_2) {
    uint neighborIdx_sum_dotBuf_2 = (lid + 2u);
    float myVal_sum_dotBuf_2 = dotBuf[lid];
    float neighborVal_sum_dotBuf_2 = dotBuf[neighborIdx_sum_dotBuf_2];
    float sumVal_sum_dotBuf_2 = (myVal_sum_dotBuf_2 + neighborVal_sum_dotBuf_2);
    dotBuf[lid] = sumVal_sum_dotBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_1 = (lid < 1u);
  if (shouldReduce_sum_dotBuf_1) {
    uint neighborIdx_sum_dotBuf_1 = (lid + 1u);
    float myVal_sum_dotBuf_1 = dotBuf[lid];
    float neighborVal_sum_dotBuf_1 = dotBuf[neighborIdx_sum_dotBuf_1];
    float sumVal_sum_dotBuf_1 = (myVal_sum_dotBuf_1 + neighborVal_sum_dotBuf_1);
    dotBuf[lid] = sumVal_sum_dotBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float dot = dotBuf[0u];
  if ((batchValid && seqValid)) {
    float dyMinusDot = (dy - dot);
    float dxF32 = (y * dyMinusDot);
    half dx = ((half)(dxF32));
    gradInput[globalIdx] = dx;
  }
}

)kernel";
const char* k_metal_navatala_transformer_softmax_with_mask_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_softmax_with_mask_f32(device const float* _input [[buffer(0)]], device const uint* batchSize [[buffer(1)]], device const uint* seqLen [[buffer(2)]], device const bool* isCausal [[buffer(3)]], device float* _output [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint rowIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  bool causal = isCausal[0u];
  threadgroup float maxBuf[256];
  threadgroup float sumBuf[256];
  bool rowValid = (rowIdx < bs);
  bool colValid = (lid < sl);
  bool valid = (rowValid && colValid);
  uint col = lid;
  uint row = (rowIdx % sl);
  uint globalIdx = ((rowIdx * sl) + lid);
  float rawX = ((valid) ? (_input[globalIdx]) : (as_type<float>(0xf149f2cau)));
  bool isMasked = (causal && (col > row));
  float x = ((isMasked) ? (as_type<float>(0xf149f2cau)) : (rawX));
  maxBuf[lid] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_128 = (lid < 128u);
  if (shouldReduce_max_maxBuf_128) {
    uint neighborIdx_max_maxBuf_128 = (lid + 128u);
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_64 = (lid < 64u);
  if (shouldReduce_max_maxBuf_64) {
    uint neighborIdx_max_maxBuf_64 = (lid + 64u);
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_32 = (lid < 32u);
  if (shouldReduce_max_maxBuf_32) {
    uint neighborIdx_max_maxBuf_32 = (lid + 32u);
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_16 = (lid < 16u);
  if (shouldReduce_max_maxBuf_16) {
    uint neighborIdx_max_maxBuf_16 = (lid + 16u);
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_8 = (lid < 8u);
  if (shouldReduce_max_maxBuf_8) {
    uint neighborIdx_max_maxBuf_8 = (lid + 8u);
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_4 = (lid < 4u);
  if (shouldReduce_max_maxBuf_4) {
    uint neighborIdx_max_maxBuf_4 = (lid + 4u);
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_2 = (lid < 2u);
  if (shouldReduce_max_maxBuf_2) {
    uint neighborIdx_max_maxBuf_2 = (lid + 2u);
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_1 = (lid < 1u);
  if (shouldReduce_max_maxBuf_1) {
    uint neighborIdx_max_maxBuf_1 = (lid + 1u);
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxVal = maxBuf[0u];
  float xShifted = (x - maxVal);
  float expVal = exp(xShifted);
  float expValMasked = (((colValid && (!isMasked))) ? (expVal) : (as_type<float>(0x00000000u)));
  sumBuf[lid] = expValMasked;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_sumBuf_128) {
    uint neighborIdx_sum_sumBuf_128 = (lid + 128u);
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_sumBuf_64) {
    uint neighborIdx_sum_sumBuf_64 = (lid + 64u);
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_sumBuf_32) {
    uint neighborIdx_sum_sumBuf_32 = (lid + 32u);
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_sumBuf_16) {
    uint neighborIdx_sum_sumBuf_16 = (lid + 16u);
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_sumBuf_8) {
    uint neighborIdx_sum_sumBuf_8 = (lid + 8u);
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_sumBuf_4) {
    uint neighborIdx_sum_sumBuf_4 = (lid + 4u);
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_sumBuf_2) {
    uint neighborIdx_sum_sumBuf_2 = (lid + 2u);
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_sumBuf_1) {
    uint neighborIdx_sum_sumBuf_1 = (lid + 1u);
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = sumBuf[0u];
  if (valid) {
    float result = ((isMasked) ? (as_type<float>(0x00000000u)) : ((expVal / sumExp)));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_softmax_with_mask_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_softmax_with_mask_f16(device const half* _input [[buffer(0)]], device const uint* batchSize [[buffer(1)]], device const uint* seqLen [[buffer(2)]], device const bool* isCausal [[buffer(3)]], device half* _output [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint rowIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  bool causal = isCausal[0u];
  threadgroup float maxBuf[256];
  threadgroup float sumBuf[256];
  bool rowValid = (rowIdx < bs);
  bool colValid = (lid < sl);
  bool valid = (rowValid && colValid);
  uint col = lid;
  uint row = (rowIdx % sl);
  uint globalIdx = ((rowIdx * sl) + lid);
  half rawXF16 = ((valid) ? (_input[globalIdx]) : (half(0.000000)));
  float rawX = ((colValid) ? (((float)(rawXF16))) : (as_type<float>(0xf149f2cau)));
  bool isMasked = (causal && (col > row));
  float x = ((isMasked) ? (as_type<float>(0xf149f2cau)) : (rawX));
  maxBuf[lid] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_128 = (lid < 128u);
  if (shouldReduce_max_maxBuf_128) {
    uint neighborIdx_max_maxBuf_128 = (lid + 128u);
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_64 = (lid < 64u);
  if (shouldReduce_max_maxBuf_64) {
    uint neighborIdx_max_maxBuf_64 = (lid + 64u);
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_32 = (lid < 32u);
  if (shouldReduce_max_maxBuf_32) {
    uint neighborIdx_max_maxBuf_32 = (lid + 32u);
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_16 = (lid < 16u);
  if (shouldReduce_max_maxBuf_16) {
    uint neighborIdx_max_maxBuf_16 = (lid + 16u);
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_8 = (lid < 8u);
  if (shouldReduce_max_maxBuf_8) {
    uint neighborIdx_max_maxBuf_8 = (lid + 8u);
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_4 = (lid < 4u);
  if (shouldReduce_max_maxBuf_4) {
    uint neighborIdx_max_maxBuf_4 = (lid + 4u);
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_2 = (lid < 2u);
  if (shouldReduce_max_maxBuf_2) {
    uint neighborIdx_max_maxBuf_2 = (lid + 2u);
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_1 = (lid < 1u);
  if (shouldReduce_max_maxBuf_1) {
    uint neighborIdx_max_maxBuf_1 = (lid + 1u);
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxVal = maxBuf[0u];
  float xShifted = (x - maxVal);
  float expVal = exp(xShifted);
  float expValMasked = (((colValid && (!isMasked))) ? (expVal) : (as_type<float>(0x00000000u)));
  sumBuf[lid] = expValMasked;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_sumBuf_128) {
    uint neighborIdx_sum_sumBuf_128 = (lid + 128u);
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_sumBuf_64) {
    uint neighborIdx_sum_sumBuf_64 = (lid + 64u);
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_sumBuf_32) {
    uint neighborIdx_sum_sumBuf_32 = (lid + 32u);
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_sumBuf_16) {
    uint neighborIdx_sum_sumBuf_16 = (lid + 16u);
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_sumBuf_8) {
    uint neighborIdx_sum_sumBuf_8 = (lid + 8u);
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_sumBuf_4) {
    uint neighborIdx_sum_sumBuf_4 = (lid + 4u);
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_sumBuf_2) {
    uint neighborIdx_sum_sumBuf_2 = (lid + 2u);
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_sumBuf_1) {
    uint neighborIdx_sum_sumBuf_1 = (lid + 1u);
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = sumBuf[0u];
  if (valid) {
    float resultF32 = ((isMasked) ? (as_type<float>(0x00000000u)) : ((expVal / sumExp)));
    half result = ((half)(resultF32));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_softmax_with_mask_backward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_softmax_with_mask_backward_f32(device const float* _output [[buffer(0)]], device const float* gradOutput [[buffer(1)]], device const uint* batchSize [[buffer(2)]], device const uint* seqLen [[buffer(3)]], device const bool* isCausal [[buffer(4)]], device float* gradInput [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint rowIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  bool causal = isCausal[0u];
  threadgroup float dotBuf[256];
  bool rowValid = (rowIdx < bs);
  bool colValid = (lid < sl);
  bool valid = (rowValid && colValid);
  uint col = lid;
  uint row = (rowIdx % sl);
  bool isMasked = (causal && (col > row));
  uint globalIdx = ((rowIdx * sl) + lid);
  float y = ((valid) ? (_output[globalIdx]) : (as_type<float>(0x00000000u)));
  float dy = ((valid) ? (gradOutput[globalIdx]) : (as_type<float>(0x00000000u)));
  float yDy = ((isMasked) ? (as_type<float>(0x00000000u)) : ((y * dy)));
  dotBuf[lid] = yDy;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_128 = (lid < 128u);
  if (shouldReduce_sum_dotBuf_128) {
    uint neighborIdx_sum_dotBuf_128 = (lid + 128u);
    float myVal_sum_dotBuf_128 = dotBuf[lid];
    float neighborVal_sum_dotBuf_128 = dotBuf[neighborIdx_sum_dotBuf_128];
    float sumVal_sum_dotBuf_128 = (myVal_sum_dotBuf_128 + neighborVal_sum_dotBuf_128);
    dotBuf[lid] = sumVal_sum_dotBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_64 = (lid < 64u);
  if (shouldReduce_sum_dotBuf_64) {
    uint neighborIdx_sum_dotBuf_64 = (lid + 64u);
    float myVal_sum_dotBuf_64 = dotBuf[lid];
    float neighborVal_sum_dotBuf_64 = dotBuf[neighborIdx_sum_dotBuf_64];
    float sumVal_sum_dotBuf_64 = (myVal_sum_dotBuf_64 + neighborVal_sum_dotBuf_64);
    dotBuf[lid] = sumVal_sum_dotBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_32 = (lid < 32u);
  if (shouldReduce_sum_dotBuf_32) {
    uint neighborIdx_sum_dotBuf_32 = (lid + 32u);
    float myVal_sum_dotBuf_32 = dotBuf[lid];
    float neighborVal_sum_dotBuf_32 = dotBuf[neighborIdx_sum_dotBuf_32];
    float sumVal_sum_dotBuf_32 = (myVal_sum_dotBuf_32 + neighborVal_sum_dotBuf_32);
    dotBuf[lid] = sumVal_sum_dotBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_16 = (lid < 16u);
  if (shouldReduce_sum_dotBuf_16) {
    uint neighborIdx_sum_dotBuf_16 = (lid + 16u);
    float myVal_sum_dotBuf_16 = dotBuf[lid];
    float neighborVal_sum_dotBuf_16 = dotBuf[neighborIdx_sum_dotBuf_16];
    float sumVal_sum_dotBuf_16 = (myVal_sum_dotBuf_16 + neighborVal_sum_dotBuf_16);
    dotBuf[lid] = sumVal_sum_dotBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_8 = (lid < 8u);
  if (shouldReduce_sum_dotBuf_8) {
    uint neighborIdx_sum_dotBuf_8 = (lid + 8u);
    float myVal_sum_dotBuf_8 = dotBuf[lid];
    float neighborVal_sum_dotBuf_8 = dotBuf[neighborIdx_sum_dotBuf_8];
    float sumVal_sum_dotBuf_8 = (myVal_sum_dotBuf_8 + neighborVal_sum_dotBuf_8);
    dotBuf[lid] = sumVal_sum_dotBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_4 = (lid < 4u);
  if (shouldReduce_sum_dotBuf_4) {
    uint neighborIdx_sum_dotBuf_4 = (lid + 4u);
    float myVal_sum_dotBuf_4 = dotBuf[lid];
    float neighborVal_sum_dotBuf_4 = dotBuf[neighborIdx_sum_dotBuf_4];
    float sumVal_sum_dotBuf_4 = (myVal_sum_dotBuf_4 + neighborVal_sum_dotBuf_4);
    dotBuf[lid] = sumVal_sum_dotBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_2 = (lid < 2u);
  if (shouldReduce_sum_dotBuf_2) {
    uint neighborIdx_sum_dotBuf_2 = (lid + 2u);
    float myVal_sum_dotBuf_2 = dotBuf[lid];
    float neighborVal_sum_dotBuf_2 = dotBuf[neighborIdx_sum_dotBuf_2];
    float sumVal_sum_dotBuf_2 = (myVal_sum_dotBuf_2 + neighborVal_sum_dotBuf_2);
    dotBuf[lid] = sumVal_sum_dotBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_1 = (lid < 1u);
  if (shouldReduce_sum_dotBuf_1) {
    uint neighborIdx_sum_dotBuf_1 = (lid + 1u);
    float myVal_sum_dotBuf_1 = dotBuf[lid];
    float neighborVal_sum_dotBuf_1 = dotBuf[neighborIdx_sum_dotBuf_1];
    float sumVal_sum_dotBuf_1 = (myVal_sum_dotBuf_1 + neighborVal_sum_dotBuf_1);
    dotBuf[lid] = sumVal_sum_dotBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float dot = dotBuf[0u];
  if (valid) {
    float dyMinusDot = (dy - dot);
    float dxUnmasked = (y * dyMinusDot);
    float dx = ((isMasked) ? (as_type<float>(0x00000000u)) : (dxUnmasked));
    gradInput[globalIdx] = dx;
  }
}

)kernel";
const char* k_metal_navatala_transformer_softmax_with_mask_backward_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_softmax_with_mask_backward_f16(device const half* _output [[buffer(0)]], device const half* gradOutput [[buffer(1)]], device const uint* batchSize [[buffer(2)]], device const uint* seqLen [[buffer(3)]], device const bool* isCausal [[buffer(4)]], device half* gradInput [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint rowIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  bool causal = isCausal[0u];
  threadgroup float dotBuf[256];
  bool rowValid = (rowIdx < bs);
  bool colValid = (lid < sl);
  bool valid = (rowValid && colValid);
  uint col = lid;
  uint row = (rowIdx % sl);
  bool isMasked = (causal && (col > row));
  uint globalIdx = ((rowIdx * sl) + lid);
  half yF16 = ((valid) ? (_output[globalIdx]) : (half(0.000000)));
  half dyF16 = ((valid) ? (gradOutput[globalIdx]) : (half(0.000000)));
  float y = ((float)(yF16));
  float dy = ((float)(dyF16));
  float yDy = ((isMasked) ? (as_type<float>(0x00000000u)) : ((y * dy)));
  dotBuf[lid] = yDy;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_128 = (lid < 128u);
  if (shouldReduce_sum_dotBuf_128) {
    uint neighborIdx_sum_dotBuf_128 = (lid + 128u);
    float myVal_sum_dotBuf_128 = dotBuf[lid];
    float neighborVal_sum_dotBuf_128 = dotBuf[neighborIdx_sum_dotBuf_128];
    float sumVal_sum_dotBuf_128 = (myVal_sum_dotBuf_128 + neighborVal_sum_dotBuf_128);
    dotBuf[lid] = sumVal_sum_dotBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_64 = (lid < 64u);
  if (shouldReduce_sum_dotBuf_64) {
    uint neighborIdx_sum_dotBuf_64 = (lid + 64u);
    float myVal_sum_dotBuf_64 = dotBuf[lid];
    float neighborVal_sum_dotBuf_64 = dotBuf[neighborIdx_sum_dotBuf_64];
    float sumVal_sum_dotBuf_64 = (myVal_sum_dotBuf_64 + neighborVal_sum_dotBuf_64);
    dotBuf[lid] = sumVal_sum_dotBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_32 = (lid < 32u);
  if (shouldReduce_sum_dotBuf_32) {
    uint neighborIdx_sum_dotBuf_32 = (lid + 32u);
    float myVal_sum_dotBuf_32 = dotBuf[lid];
    float neighborVal_sum_dotBuf_32 = dotBuf[neighborIdx_sum_dotBuf_32];
    float sumVal_sum_dotBuf_32 = (myVal_sum_dotBuf_32 + neighborVal_sum_dotBuf_32);
    dotBuf[lid] = sumVal_sum_dotBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_16 = (lid < 16u);
  if (shouldReduce_sum_dotBuf_16) {
    uint neighborIdx_sum_dotBuf_16 = (lid + 16u);
    float myVal_sum_dotBuf_16 = dotBuf[lid];
    float neighborVal_sum_dotBuf_16 = dotBuf[neighborIdx_sum_dotBuf_16];
    float sumVal_sum_dotBuf_16 = (myVal_sum_dotBuf_16 + neighborVal_sum_dotBuf_16);
    dotBuf[lid] = sumVal_sum_dotBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_8 = (lid < 8u);
  if (shouldReduce_sum_dotBuf_8) {
    uint neighborIdx_sum_dotBuf_8 = (lid + 8u);
    float myVal_sum_dotBuf_8 = dotBuf[lid];
    float neighborVal_sum_dotBuf_8 = dotBuf[neighborIdx_sum_dotBuf_8];
    float sumVal_sum_dotBuf_8 = (myVal_sum_dotBuf_8 + neighborVal_sum_dotBuf_8);
    dotBuf[lid] = sumVal_sum_dotBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_4 = (lid < 4u);
  if (shouldReduce_sum_dotBuf_4) {
    uint neighborIdx_sum_dotBuf_4 = (lid + 4u);
    float myVal_sum_dotBuf_4 = dotBuf[lid];
    float neighborVal_sum_dotBuf_4 = dotBuf[neighborIdx_sum_dotBuf_4];
    float sumVal_sum_dotBuf_4 = (myVal_sum_dotBuf_4 + neighborVal_sum_dotBuf_4);
    dotBuf[lid] = sumVal_sum_dotBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_2 = (lid < 2u);
  if (shouldReduce_sum_dotBuf_2) {
    uint neighborIdx_sum_dotBuf_2 = (lid + 2u);
    float myVal_sum_dotBuf_2 = dotBuf[lid];
    float neighborVal_sum_dotBuf_2 = dotBuf[neighborIdx_sum_dotBuf_2];
    float sumVal_sum_dotBuf_2 = (myVal_sum_dotBuf_2 + neighborVal_sum_dotBuf_2);
    dotBuf[lid] = sumVal_sum_dotBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_1 = (lid < 1u);
  if (shouldReduce_sum_dotBuf_1) {
    uint neighborIdx_sum_dotBuf_1 = (lid + 1u);
    float myVal_sum_dotBuf_1 = dotBuf[lid];
    float neighborVal_sum_dotBuf_1 = dotBuf[neighborIdx_sum_dotBuf_1];
    float sumVal_sum_dotBuf_1 = (myVal_sum_dotBuf_1 + neighborVal_sum_dotBuf_1);
    dotBuf[lid] = sumVal_sum_dotBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float dot = dotBuf[0u];
  if (valid) {
    float dyMinusDot = (dy - dot);
    float dxF32Unmasked = (y * dyMinusDot);
    float dxF32 = ((isMasked) ? (as_type<float>(0x00000000u)) : (dxF32Unmasked));
    half dx = ((half)(dxF32));
    gradInput[globalIdx] = dx;
  }
}

)kernel";
const char* k_metal_navatala_transformer_softmax_multi_pass_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_softmax_multi_pass_f32(device const float* _input [[buffer(0)]], device const uint* batchSize [[buffer(1)]], device const uint* seqLen [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  threadgroup float maxBuf[256];
  threadgroup float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  uint baseIdx = (batchIdx * sl);
  uint workgroupSize = 256u;
  float partialMax = as_type<float>(0xf149f2cau);
  float partialSum = as_type<float>(0x00000000u);
  uint iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    uint globalIdx = (baseIdx + iterIdx);
    float xVal = ((batchValid) ? (_input[globalIdx]) : (as_type<float>(0xf149f2cau)));
    bool isGreater = (xVal > partialMax);
    partialMax = ((isGreater) ? (xVal) : (partialMax));
    iterIdx = (iterIdx + workgroupSize);
  }
  maxBuf[lid] = partialMax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_128 = (lid < 128u);
  if (shouldReduce_max_maxBuf_128) {
    uint neighborIdx_max_maxBuf_128 = (lid + 128u);
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_64 = (lid < 64u);
  if (shouldReduce_max_maxBuf_64) {
    uint neighborIdx_max_maxBuf_64 = (lid + 64u);
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_32 = (lid < 32u);
  if (shouldReduce_max_maxBuf_32) {
    uint neighborIdx_max_maxBuf_32 = (lid + 32u);
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_16 = (lid < 16u);
  if (shouldReduce_max_maxBuf_16) {
    uint neighborIdx_max_maxBuf_16 = (lid + 16u);
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_8 = (lid < 8u);
  if (shouldReduce_max_maxBuf_8) {
    uint neighborIdx_max_maxBuf_8 = (lid + 8u);
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_4 = (lid < 4u);
  if (shouldReduce_max_maxBuf_4) {
    uint neighborIdx_max_maxBuf_4 = (lid + 4u);
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_2 = (lid < 2u);
  if (shouldReduce_max_maxBuf_2) {
    uint neighborIdx_max_maxBuf_2 = (lid + 2u);
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_1 = (lid < 1u);
  if (shouldReduce_max_maxBuf_1) {
    uint neighborIdx_max_maxBuf_1 = (lid + 1u);
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float globalMax = maxBuf[0u];
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    if (batchValid) {
      uint globalIdx2 = (baseIdx + iterIdx);
      float xVal2 = _input[globalIdx2];
      float xShifted = (xVal2 - globalMax);
      float expVal = exp(xShifted);
      partialSum = (partialSum + expVal);
    }
    iterIdx = (iterIdx + workgroupSize);
  }
  sumBuf[lid] = partialSum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_sumBuf_128) {
    uint neighborIdx_sum_sumBuf_128 = (lid + 128u);
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_sumBuf_64) {
    uint neighborIdx_sum_sumBuf_64 = (lid + 64u);
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_sumBuf_32) {
    uint neighborIdx_sum_sumBuf_32 = (lid + 32u);
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_sumBuf_16) {
    uint neighborIdx_sum_sumBuf_16 = (lid + 16u);
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_sumBuf_8) {
    uint neighborIdx_sum_sumBuf_8 = (lid + 8u);
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_sumBuf_4) {
    uint neighborIdx_sum_sumBuf_4 = (lid + 4u);
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_sumBuf_2) {
    uint neighborIdx_sum_sumBuf_2 = (lid + 2u);
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_sumBuf_1) {
    uint neighborIdx_sum_sumBuf_1 = (lid + 1u);
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float globalSum = sumBuf[0u];
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    if (batchValid) {
      uint globalIdx3 = (baseIdx + iterIdx);
      float xVal3 = _input[globalIdx3];
      float xShifted3 = (xVal3 - globalMax);
      float expVal3 = exp(xShifted3);
      float result = (expVal3 / globalSum);
      _output[globalIdx3] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}

)kernel";
const char* k_metal_navatala_transformer_softmax_multi_pass_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_softmax_multi_pass_f16(device const half* _input [[buffer(0)]], device const uint* batchSize [[buffer(1)]], device const uint* seqLen [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  threadgroup float maxBuf[256];
  threadgroup float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  uint baseIdx = (batchIdx * sl);
  uint workgroupSize = 256u;
  float partialMax = as_type<float>(0xf149f2cau);
  float partialSum = as_type<float>(0x00000000u);
  uint iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    uint globalIdx = (baseIdx + iterIdx);
    half xF16 = ((batchValid) ? (_input[globalIdx]) : (half(0.000000)));
    float xVal = ((float)(xF16));
    float xValSafe = ((batchValid) ? (xVal) : (as_type<float>(0xf149f2cau)));
    bool isGreater = (xValSafe > partialMax);
    partialMax = ((isGreater) ? (xValSafe) : (partialMax));
    iterIdx = (iterIdx + workgroupSize);
  }
  maxBuf[lid] = partialMax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_128 = (lid < 128u);
  if (shouldReduce_max_maxBuf_128) {
    uint neighborIdx_max_maxBuf_128 = (lid + 128u);
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_64 = (lid < 64u);
  if (shouldReduce_max_maxBuf_64) {
    uint neighborIdx_max_maxBuf_64 = (lid + 64u);
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_32 = (lid < 32u);
  if (shouldReduce_max_maxBuf_32) {
    uint neighborIdx_max_maxBuf_32 = (lid + 32u);
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_16 = (lid < 16u);
  if (shouldReduce_max_maxBuf_16) {
    uint neighborIdx_max_maxBuf_16 = (lid + 16u);
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_8 = (lid < 8u);
  if (shouldReduce_max_maxBuf_8) {
    uint neighborIdx_max_maxBuf_8 = (lid + 8u);
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_4 = (lid < 4u);
  if (shouldReduce_max_maxBuf_4) {
    uint neighborIdx_max_maxBuf_4 = (lid + 4u);
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_2 = (lid < 2u);
  if (shouldReduce_max_maxBuf_2) {
    uint neighborIdx_max_maxBuf_2 = (lid + 2u);
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_1 = (lid < 1u);
  if (shouldReduce_max_maxBuf_1) {
    uint neighborIdx_max_maxBuf_1 = (lid + 1u);
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float globalMax = maxBuf[0u];
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    if (batchValid) {
      uint globalIdx2 = (baseIdx + iterIdx);
      half xF16_2 = _input[globalIdx2];
      float xVal2 = ((float)(xF16_2));
      float xShifted = (xVal2 - globalMax);
      float expVal = exp(xShifted);
      partialSum = (partialSum + expVal);
    }
    iterIdx = (iterIdx + workgroupSize);
  }
  sumBuf[lid] = partialSum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_sumBuf_128) {
    uint neighborIdx_sum_sumBuf_128 = (lid + 128u);
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_sumBuf_64) {
    uint neighborIdx_sum_sumBuf_64 = (lid + 64u);
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_sumBuf_32) {
    uint neighborIdx_sum_sumBuf_32 = (lid + 32u);
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_sumBuf_16) {
    uint neighborIdx_sum_sumBuf_16 = (lid + 16u);
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_sumBuf_8) {
    uint neighborIdx_sum_sumBuf_8 = (lid + 8u);
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_sumBuf_4) {
    uint neighborIdx_sum_sumBuf_4 = (lid + 4u);
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_sumBuf_2) {
    uint neighborIdx_sum_sumBuf_2 = (lid + 2u);
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_sumBuf_1) {
    uint neighborIdx_sum_sumBuf_1 = (lid + 1u);
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float globalSum = sumBuf[0u];
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    if (batchValid) {
      uint globalIdx3 = (baseIdx + iterIdx);
      half xF16_3 = _input[globalIdx3];
      float xVal3 = ((float)(xF16_3));
      float xShifted3 = (xVal3 - globalMax);
      float expVal3 = exp(xShifted3);
      float resultF32 = (expVal3 / globalSum);
      half result = ((half)(resultF32));
      _output[globalIdx3] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}

)kernel";
const char* k_metal_navatala_transformer_cast_f32_to_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_cast_f32_to_f16(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    half result = ((half)(x));
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_cast_f16_to_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_cast_f16_to_f32(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    float result = ((float)(x));
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_transpose_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_transpose_f32(device const float* _input [[buffer(0)]], device const uint* rows [[buffer(1)]], device const uint* cols [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint r = rows[0u];
  uint c = cols[0u];
  uint total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / c);
    uint col = (gid % c);
    float x = _input[gid];
    uint outIdx = ((col * r) + row);
    _output[outIdx] = x;
  }
}

)kernel";
const char* k_metal_navatala_transformer_transpose_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_transpose_f16(device const half* _input [[buffer(0)]], device const uint* rows [[buffer(1)]], device const uint* cols [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint r = rows[0u];
  uint c = cols[0u];
  uint total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / c);
    uint col = (gid % c);
    half x = _input[gid];
    uint outIdx = ((col * r) + row);
    _output[outIdx] = x;
  }
}

)kernel";
const char* k_metal_navatala_transformer_fused_cast_transpose_f32_to_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_fused_cast_transpose_f32_to_f16(device const float* _input [[buffer(0)]], device const uint* rows [[buffer(1)]], device const uint* cols [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint r = rows[0u];
  uint c = cols[0u];
  uint total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / c);
    uint col = (gid % c);
    float x = _input[gid];
    half xF16 = ((half)(x));
    uint outIdx = ((col * r) + row);
    _output[outIdx] = xF16;
  }
}

)kernel";
const char* k_metal_navatala_transformer_cast_f32_to_f_p8_e4_m3 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_cast_f32_to_f_p8_e4_m3(device const float* _input [[buffer(0)]], device const float* scale [[buffer(1)]], device const uint* count [[buffer(2)]], device uchar* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float s = scale[0u];
    float scaled = (x * s);
    float minVal = as_type<float>(0xc3700000u);
    float maxVal = as_type<float>(0x43700000u);
    float clamped = ((((((scaled < maxVal)) ? (scaled) : (maxVal)) > minVal)) ? ((((scaled < maxVal)) ? (scaled) : (maxVal))) : (minVal));
    float rounded = round(clamped);
    uchar fp8Val = ((uchar)(rounded));
    _output[gid] = fp8Val;
  }
}

)kernel";
const char* k_metal_navatala_transformer_cast_f_p8_e4_m3_to_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_cast_f_p8_e4_m3_to_f32(device const uchar* _input [[buffer(0)]], device const float* scale [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uchar fp8Val = _input[gid];
    float s = scale[0u];
    float f32Val = ((float)(fp8Val));
    float result = (f32Val / s);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_cast_f32_to_f_p8_e5_m2 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_cast_f32_to_f_p8_e5_m2(device const float* _input [[buffer(0)]], device const float* scale [[buffer(1)]], device const uint* count [[buffer(2)]], device uchar* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float s = scale[0u];
    float scaled = (x * s);
    float minVal = as_type<float>(0xc7600000u);
    float maxVal = as_type<float>(0x47600000u);
    float clamped = ((((((scaled < maxVal)) ? (scaled) : (maxVal)) > minVal)) ? ((((scaled < maxVal)) ? (scaled) : (maxVal))) : (minVal));
    float rounded = round(clamped);
    uchar fp8Val = ((uchar)(rounded));
    _output[gid] = fp8Val;
  }
}

)kernel";
const char* k_metal_navatala_transformer_cast_f_p8_e5_m2_to_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_cast_f_p8_e5_m2_to_f32(device const uchar* _input [[buffer(0)]], device const float* scale [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uchar fp8Val = _input[gid];
    float s = scale[0u];
    float f32Val = ((float)(fp8Val));
    float result = (f32Val / s);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_fused_cast_transpose_f_p8 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_fused_cast_transpose_f_p8(device const float* _input [[buffer(0)]], device const float* scale [[buffer(1)]], device const uint* rows [[buffer(2)]], device const uint* cols [[buffer(3)]], device uchar* _output [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint r = rows[0u];
  uint c = cols[0u];
  uint total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / c);
    uint col = (gid % c);
    float x = _input[gid];
    float s = scale[0u];
    float scaled = (x * s);
    float minVal = as_type<float>(0xc3700000u);
    float maxVal = as_type<float>(0x43700000u);
    float clamped = ((((((scaled < maxVal)) ? (scaled) : (maxVal)) > minVal)) ? ((((scaled < maxVal)) ? (scaled) : (maxVal))) : (minVal));
    float rounded = round(clamped);
    uchar fp8Val = ((uchar)(rounded));
    uint outIdx = ((col * r) + row);
    _output[outIdx] = fp8Val;
  }
}

)kernel";
const char* k_metal_navatala_transformer_transpose4_d_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_transpose4_d_f32(device const float* _input [[buffer(0)]], device const uint* dim0 [[buffer(1)]], device const uint* dim1 [[buffer(2)]], device const uint* dim2 [[buffer(3)]], device const uint* dim3 [[buffer(4)]], device const uint* perm0 [[buffer(5)]], device const uint* perm1 [[buffer(6)]], device const uint* perm2 [[buffer(7)]], device const uint* perm3 [[buffer(8)]], device float* _output [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint d0 = dim0[0u];
  uint d1 = dim1[0u];
  uint d2 = dim2[0u];
  uint d3 = dim3[0u];
  uint p0 = perm0[0u];
  uint p1 = perm1[0u];
  uint p2 = perm2[0u];
  uint p3 = perm3[0u];
  uint total = (((d0 * d1) * d2) * d3);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint stride1 = ((d1 * d2) * d3);
    uint stride2 = (d2 * d3);
    uint stride3 = d3;
    uint i0 = (gid / stride1);
    uint rem0 = (gid % stride1);
    uint i1 = (rem0 / stride2);
    uint rem1 = (rem0 % stride2);
    uint i2 = (rem1 / stride3);
    uint i3 = (rem1 % stride3);
    uint outD0 = (((p0 == 0u)) ? (d0) : ((((p0 == 1u)) ? (d1) : ((((p0 == 2u)) ? (d2) : (d3))))));
    uint outD1 = (((p1 == 0u)) ? (d0) : ((((p1 == 1u)) ? (d1) : ((((p1 == 2u)) ? (d2) : (d3))))));
    uint outD2 = (((p2 == 0u)) ? (d0) : ((((p2 == 1u)) ? (d1) : ((((p2 == 2u)) ? (d2) : (d3))))));
    uint outD3 = (((p3 == 0u)) ? (d0) : ((((p3 == 1u)) ? (d1) : ((((p3 == 2u)) ? (d2) : (d3))))));
    uint outI0 = (((p0 == 0u)) ? (i0) : ((((p0 == 1u)) ? (i1) : ((((p0 == 2u)) ? (i2) : (i3))))));
    uint outI1 = (((p1 == 0u)) ? (i0) : ((((p1 == 1u)) ? (i1) : ((((p1 == 2u)) ? (i2) : (i3))))));
    uint outI2 = (((p2 == 0u)) ? (i0) : ((((p2 == 1u)) ? (i1) : ((((p2 == 2u)) ? (i2) : (i3))))));
    uint outI3 = (((p3 == 0u)) ? (i0) : ((((p3 == 1u)) ? (i1) : ((((p3 == 2u)) ? (i2) : (i3))))));
    uint outStride1 = ((outD1 * outD2) * outD3);
    uint outStride2 = (outD2 * outD3);
    uint outStride3 = outD3;
    uint outIdx = ((((outI0 * outStride1) + (outI1 * outStride2)) + (outI2 * outStride3)) + outI3);
    float x = _input[gid];
    _output[outIdx] = x;
  }
}

)kernel";
const char* k_metal_navatala_transformer_transpose4_d_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_transpose4_d_f16(device const half* _input [[buffer(0)]], device const uint* dim0 [[buffer(1)]], device const uint* dim1 [[buffer(2)]], device const uint* dim2 [[buffer(3)]], device const uint* dim3 [[buffer(4)]], device const uint* perm0 [[buffer(5)]], device const uint* perm1 [[buffer(6)]], device const uint* perm2 [[buffer(7)]], device const uint* perm3 [[buffer(8)]], device half* _output [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint d0 = dim0[0u];
  uint d1 = dim1[0u];
  uint d2 = dim2[0u];
  uint d3 = dim3[0u];
  uint p0 = perm0[0u];
  uint p1 = perm1[0u];
  uint p2 = perm2[0u];
  uint p3 = perm3[0u];
  uint total = (((d0 * d1) * d2) * d3);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint stride1 = ((d1 * d2) * d3);
    uint stride2 = (d2 * d3);
    uint stride3 = d3;
    uint i0 = (gid / stride1);
    uint rem0 = (gid % stride1);
    uint i1 = (rem0 / stride2);
    uint rem1 = (rem0 % stride2);
    uint i2 = (rem1 / stride3);
    uint i3 = (rem1 % stride3);
    uint outD0 = (((p0 == 0u)) ? (d0) : ((((p0 == 1u)) ? (d1) : ((((p0 == 2u)) ? (d2) : (d3))))));
    uint outD1 = (((p1 == 0u)) ? (d0) : ((((p1 == 1u)) ? (d1) : ((((p1 == 2u)) ? (d2) : (d3))))));
    uint outD2 = (((p2 == 0u)) ? (d0) : ((((p2 == 1u)) ? (d1) : ((((p2 == 2u)) ? (d2) : (d3))))));
    uint outD3 = (((p3 == 0u)) ? (d0) : ((((p3 == 1u)) ? (d1) : ((((p3 == 2u)) ? (d2) : (d3))))));
    uint outI0 = (((p0 == 0u)) ? (i0) : ((((p0 == 1u)) ? (i1) : ((((p0 == 2u)) ? (i2) : (i3))))));
    uint outI1 = (((p1 == 0u)) ? (i0) : ((((p1 == 1u)) ? (i1) : ((((p1 == 2u)) ? (i2) : (i3))))));
    uint outI2 = (((p2 == 0u)) ? (i0) : ((((p2 == 1u)) ? (i1) : ((((p2 == 2u)) ? (i2) : (i3))))));
    uint outI3 = (((p3 == 0u)) ? (i0) : ((((p3 == 1u)) ? (i1) : ((((p3 == 2u)) ? (i2) : (i3))))));
    uint outStride1 = ((outD1 * outD2) * outD3);
    uint outStride2 = (outD2 * outD3);
    uint outStride3 = outD3;
    uint outIdx = ((((outI0 * outStride1) + (outI1 * outStride2)) + (outI2 * outStride3)) + outI3);
    half x = _input[gid];
    _output[outIdx] = x;
  }
}

)kernel";
const char* k_metal_navatala_transformer_dropout_forward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_dropout_forward_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device const float* prob [[buffer(2)]], device const uint* seed [[buffer(3)]], device float* _output [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  float p = prob[0u];
  uint s = seed[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    uint h0 = (gid ^ s);
    uint h1 = (h0 * 3432918353u);
    uint h2 = (h1 ^ (h1 >> 15u));
    uint h3 = (h2 * 461845907u);
    uint hashVal = (h3 ^ (h3 >> 13u));
    float randF32 = (((float)(hashVal)) / as_type<float>(0x4f800000u));
    bool keep = (randF32 > p);
    float oneMinusP = (as_type<float>(0x3f800000u) - p);
    float scale = (as_type<float>(0x3f800000u) / oneMinusP);
    float scaled = (x * scale);
    float result = ((keep) ? (scaled) : (as_type<float>(0x00000000u)));
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_dropout_forward_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_dropout_forward_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device const float* prob [[buffer(2)]], device const uint* seed [[buffer(3)]], device half* _output [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  float p = prob[0u];
  uint s = seed[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    half xF16 = _input[gid];
    float x = ((float)(xF16));
    uint h0 = (gid ^ s);
    uint h1 = (h0 * 3432918353u);
    uint h2 = (h1 ^ (h1 >> 15u));
    uint h3 = (h2 * 461845907u);
    uint hashVal = (h3 ^ (h3 >> 13u));
    float randF32 = (((float)(hashVal)) / as_type<float>(0x4f800000u));
    bool keep = (randF32 > p);
    float oneMinusP = (as_type<float>(0x3f800000u) - p);
    float scale = (as_type<float>(0x3f800000u) / oneMinusP);
    float scaled = (x * scale);
    float resultF32 = ((keep) ? (scaled) : (as_type<float>(0x00000000u)));
    half result = ((half)(resultF32));
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_dropout_backward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_dropout_backward_f32(device const float* gradOutput [[buffer(0)]], device const uint* count [[buffer(1)]], device const float* prob [[buffer(2)]], device const uint* seed [[buffer(3)]], device float* gradInput [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  float p = prob[0u];
  uint s = seed[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float dy = gradOutput[gid];
    uint h0 = (gid ^ s);
    uint h1 = (h0 * 3432918353u);
    uint h2 = (h1 ^ (h1 >> 15u));
    uint h3 = (h2 * 461845907u);
    uint hashVal = (h3 ^ (h3 >> 13u));
    float randF32 = (((float)(hashVal)) / as_type<float>(0x4f800000u));
    bool keep = (randF32 > p);
    float oneMinusP = (as_type<float>(0x3f800000u) - p);
    float scale = (as_type<float>(0x3f800000u) / oneMinusP);
    float scaled = (dy * scale);
    float dx = ((keep) ? (scaled) : (as_type<float>(0x00000000u)));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_metal_navatala_transformer_dropout_backward_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_dropout_backward_f16(device const half* gradOutput [[buffer(0)]], device const uint* count [[buffer(1)]], device const float* prob [[buffer(2)]], device const uint* seed [[buffer(3)]], device half* gradInput [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  float p = prob[0u];
  uint s = seed[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    half dyF16 = gradOutput[gid];
    float dy = ((float)(dyF16));
    uint h0 = (gid ^ s);
    uint h1 = (h0 * 3432918353u);
    uint h2 = (h1 ^ (h1 >> 15u));
    uint h3 = (h2 * 461845907u);
    uint hashVal = (h3 ^ (h3 >> 13u));
    float randF32 = (((float)(hashVal)) / as_type<float>(0x4f800000u));
    bool keep = (randF32 > p);
    float oneMinusP = (as_type<float>(0x3f800000u) - p);
    float scale = (as_type<float>(0x3f800000u) / oneMinusP);
    float scaled = (dy * scale);
    float dxF32 = ((keep) ? (scaled) : (as_type<float>(0x00000000u)));
    half dx = ((half)(dxF32));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_metal_navatala_transformer_rope_forward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_rope_forward_f32(device const float* _input [[buffer(0)]], device const uint* positions [[buffer(1)]], device const uint* batchSize [[buffer(2)]], device const uint* seqLen [[buffer(3)]], device const uint* headDim [[buffer(4)]], device const float* base [[buffer(5)]], device float* _output [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  uint hd = headDim[0u];
  float ropeBase = base[0u];
  uint halfHd = (hd / 2u);
  uint totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint batchSeqPairs = (sl * halfHd);
    uint batchIdx = (gid / batchSeqPairs);
    uint remainder = (gid % batchSeqPairs);
    uint seqIdx = (remainder / halfHd);
    uint dimPair = (remainder % halfHd);
    uint posIdx = ((batchIdx * sl) + seqIdx);
    uint pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (as_type<float>(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint dim0 = (dimPair * 2u);
    uint dim1 = (dim0 + 1u);
    uint idx0 = (baseOffset + (seqOffset + dim0));
    uint idx1 = (baseOffset + (seqOffset + dim1));
    float x0 = _input[idx0];
    float x1 = _input[idx1];
    float x0Cos = (x0 * cosTheta);
    float x1Sin = (x1 * sinTheta);
    float y0 = (x0Cos - x1Sin);
    float x0Sin = (x0 * sinTheta);
    float x1Cos = (x1 * cosTheta);
    float y1 = (x0Sin + x1Cos);
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}

)kernel";
const char* k_metal_navatala_transformer_rope_forward_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_rope_forward_f16(device const half* _input [[buffer(0)]], device const uint* positions [[buffer(1)]], device const uint* batchSize [[buffer(2)]], device const uint* seqLen [[buffer(3)]], device const uint* headDim [[buffer(4)]], device const float* base [[buffer(5)]], device half* _output [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  uint hd = headDim[0u];
  float ropeBase = base[0u];
  uint halfHd = (hd / 2u);
  uint totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint batchSeqPairs = (sl * halfHd);
    uint batchIdx = (gid / batchSeqPairs);
    uint remainder = (gid % batchSeqPairs);
    uint seqIdx = (remainder / halfHd);
    uint dimPair = (remainder % halfHd);
    uint posIdx = ((batchIdx * sl) + seqIdx);
    uint pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (as_type<float>(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint dim0 = (dimPair * 2u);
    uint dim1 = (dim0 + 1u);
    uint idx0 = (baseOffset + (seqOffset + dim0));
    uint idx1 = (baseOffset + (seqOffset + dim1));
    half x0F16 = _input[idx0];
    half x1F16 = _input[idx1];
    float x0 = ((float)(x0F16));
    float x1 = ((float)(x1F16));
    float x0Cos = (x0 * cosTheta);
    float x1Sin = (x1 * sinTheta);
    float y0F32 = (x0Cos - x1Sin);
    float x0Sin = (x0 * sinTheta);
    float x1Cos = (x1 * cosTheta);
    float y1F32 = (x0Sin + x1Cos);
    half y0 = ((half)(y0F32));
    half y1 = ((half)(y1F32));
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}

)kernel";
const char* k_metal_navatala_transformer_rope_with_cache_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_rope_with_cache_f32(device const float* _input [[buffer(0)]], device const float* cosCache [[buffer(1)]], device const float* sinCache [[buffer(2)]], device const uint* positions [[buffer(3)]], device const uint* batchSize [[buffer(4)]], device const uint* seqLen [[buffer(5)]], device const uint* headDim [[buffer(6)]], device float* _output [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  uint hd = headDim[0u];
  uint halfHd = (hd / 2u);
  uint totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint batchSeqPairs = (sl * halfHd);
    uint batchIdx = (gid / batchSeqPairs);
    uint remainder = (gid % batchSeqPairs);
    uint seqIdx = (remainder / halfHd);
    uint dimPair = (remainder % halfHd);
    uint posIdx = ((batchIdx * sl) + seqIdx);
    uint pos = positions[posIdx];
    uint cacheIdx = ((pos * halfHd) + dimPair);
    float cosTheta = cosCache[cacheIdx];
    float sinTheta = sinCache[cacheIdx];
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint dim0 = (dimPair * 2u);
    uint dim1 = (dim0 + 1u);
    uint idx0 = (baseOffset + (seqOffset + dim0));
    uint idx1 = (baseOffset + (seqOffset + dim1));
    float x0 = _input[idx0];
    float x1 = _input[idx1];
    float y0 = ((x0 * cosTheta) - (x1 * sinTheta));
    float y1 = ((x0 * sinTheta) + (x1 * cosTheta));
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}

)kernel";
const char* k_metal_navatala_transformer_rope_backward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_rope_backward_f32(device const float* gradOutput [[buffer(0)]], device const uint* positions [[buffer(1)]], device const uint* batchSize [[buffer(2)]], device const uint* seqLen [[buffer(3)]], device const uint* headDim [[buffer(4)]], device const float* base [[buffer(5)]], device float* gradInput [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  uint hd = headDim[0u];
  float ropeBase = base[0u];
  uint halfHd = (hd / 2u);
  uint totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint batchSeqPairs = (sl * halfHd);
    uint batchIdx = (gid / batchSeqPairs);
    uint remainder = (gid % batchSeqPairs);
    uint seqIdx = (remainder / halfHd);
    uint dimPair = (remainder % halfHd);
    uint posIdx = ((batchIdx * sl) + seqIdx);
    uint pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (as_type<float>(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint dim0 = (dimPair * 2u);
    uint dim1 = (dim0 + 1u);
    uint idx0 = (baseOffset + (seqOffset + dim0));
    uint idx1 = (baseOffset + (seqOffset + dim1));
    float dy0 = gradOutput[idx0];
    float dy1 = gradOutput[idx1];
    float dy0Cos = (dy0 * cosTheta);
    float dy1Sin = (dy1 * sinTheta);
    float dx0 = (dy0Cos + dy1Sin);
    float dy0Sin = (dy0 * sinTheta);
    float dy1Cos = (dy1 * cosTheta);
    float negDy0Sin = (as_type<float>(0x00000000u) - dy0Sin);
    float dx1 = (negDy0Sin + dy1Cos);
    gradInput[idx0] = dx0;
    gradInput[idx1] = dx1;
  }
}

)kernel";
const char* k_metal_navatala_transformer_rope_backward_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_rope_backward_f16(device const half* gradOutput [[buffer(0)]], device const uint* positions [[buffer(1)]], device const uint* batchSize [[buffer(2)]], device const uint* seqLen [[buffer(3)]], device const uint* headDim [[buffer(4)]], device const float* base [[buffer(5)]], device half* gradInput [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  uint hd = headDim[0u];
  float ropeBase = base[0u];
  uint halfHd = (hd / 2u);
  uint totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint batchSeqPairs = (sl * halfHd);
    uint batchIdx = (gid / batchSeqPairs);
    uint remainder = (gid % batchSeqPairs);
    uint seqIdx = (remainder / halfHd);
    uint dimPair = (remainder % halfHd);
    uint posIdx = ((batchIdx * sl) + seqIdx);
    uint pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (as_type<float>(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint dim0 = (dimPair * 2u);
    uint dim1 = (dim0 + 1u);
    uint idx0 = (baseOffset + (seqOffset + dim0));
    uint idx1 = (baseOffset + (seqOffset + dim1));
    half dy0F16 = gradOutput[idx0];
    half dy1F16 = gradOutput[idx1];
    float dy0 = ((float)(dy0F16));
    float dy1 = ((float)(dy1F16));
    float dy0Cos = (dy0 * cosTheta);
    float dy1Sin = (dy1 * sinTheta);
    float dx0F32 = (dy0Cos + dy1Sin);
    float dy0Sin = (dy0 * sinTheta);
    float dy1Cos = (dy1 * cosTheta);
    float negDy0Sin = (as_type<float>(0x00000000u) - dy0Sin);
    float dx1F32 = (negDy0Sin + dy1Cos);
    half dx0 = ((half)(dx0F32));
    half dx1 = ((half)(dx1F32));
    gradInput[idx0] = dx0;
    gradInput[idx1] = dx1;
  }
}

)kernel";
const char* k_metal_navatala_transformer_rope_interleaved_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_rope_interleaved_f32(device const float* _input [[buffer(0)]], device const uint* positions [[buffer(1)]], device const uint* batchSize [[buffer(2)]], device const uint* seqLen [[buffer(3)]], device const uint* headDim [[buffer(4)]], device const float* base [[buffer(5)]], device float* _output [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  uint hd = headDim[0u];
  float ropeBase = base[0u];
  uint halfHd = (hd / 2u);
  uint totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint batchSeqPairs = (sl * halfHd);
    uint batchIdx = (gid / batchSeqPairs);
    uint remainder = (gid % batchSeqPairs);
    uint seqIdx = (remainder / halfHd);
    uint dimIdx = (remainder % halfHd);
    uint posIdx = ((batchIdx * sl) + seqIdx);
    uint pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimIdxF32 = ((float)(dimIdx));
    float hdF32 = ((float)(hd));
    float twoD = (as_type<float>(0x40000000u) * dimIdxF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint idx0 = (baseOffset + (seqOffset + dimIdx));
    uint idx1 = (idx0 + halfHd);
    float x0 = _input[idx0];
    float x1 = _input[idx1];
    float x0Cos = (x0 * cosTheta);
    float x1Sin = (x1 * sinTheta);
    float y0 = (x0Cos - x1Sin);
    float x0Sin = (x0 * sinTheta);
    float x1Cos = (x1 * cosTheta);
    float y1 = (x0Sin + x1Cos);
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}

)kernel";
const char* k_metal_navatala_transformer_rope_interleaved_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_rope_interleaved_f16(device const half* _input [[buffer(0)]], device const uint* positions [[buffer(1)]], device const uint* batchSize [[buffer(2)]], device const uint* seqLen [[buffer(3)]], device const uint* headDim [[buffer(4)]], device const float* base [[buffer(5)]], device half* _output [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  uint hd = headDim[0u];
  float ropeBase = base[0u];
  uint halfHd = (hd / 2u);
  uint totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint batchSeqPairs = (sl * halfHd);
    uint batchIdx = (gid / batchSeqPairs);
    uint remainder = (gid % batchSeqPairs);
    uint seqIdx = (remainder / halfHd);
    uint dimIdx = (remainder % halfHd);
    uint posIdx = ((batchIdx * sl) + seqIdx);
    uint pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimIdxF32 = ((float)(dimIdx));
    float hdF32 = ((float)(hd));
    float twoD = (as_type<float>(0x40000000u) * dimIdxF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint idx0 = (baseOffset + (seqOffset + dimIdx));
    uint idx1 = (idx0 + halfHd);
    half x0F16 = _input[idx0];
    half x1F16 = _input[idx1];
    float x0 = ((float)(x0F16));
    float x1 = ((float)(x1F16));
    float x0Cos = (x0 * cosTheta);
    float x1Sin = (x1 * sinTheta);
    float y0F32 = (x0Cos - x1Sin);
    float x0Sin = (x0 * sinTheta);
    float x1Cos = (x1 * cosTheta);
    float y1F32 = (x0Sin + x1Cos);
    half y0 = ((half)(y0F32));
    half y1 = ((half)(y1F32));
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}

)kernel";
const char* k_metal_navatala_transformer_simple_attention_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_simple_attention_f16(device const half* query [[buffer(0)]], device const half* key [[buffer(1)]], device const half* value [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* seqLen [[buffer(4)]], device const uint* numHeads [[buffer(5)]], device const uint* headDim [[buffer(6)]], device half* _output [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint groupId = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  uint nh = numHeads[0u];
  uint hd = headDim[0u];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_type<float>(0x3f800000u) / sqrtHd);
  threadgroup float attnScores[256];
  threadgroup float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if (valid) {
    float dotProd = as_type<float>(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      half qValF16 = query[qIdx];
      half kValF16 = key[kIdx];
      float qVal = ((float)(qValF16));
      float kVal = ((float)(kValF16));
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = as_type<float>(0xf149f2cau);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxScore = attnScores[0u];
  if (valid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = sumBuf[0u];
  if (valid) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    half v0F16 = value[vBase];
    float v0 = ((float)(v0F16));
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p2_sumBuf_128) {
    uint neighborIdx_sum_p2_sumBuf_128 = (lid + 128u);
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p2_sumBuf_64) {
    uint neighborIdx_sum_p2_sumBuf_64 = (lid + 64u);
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p2_sumBuf_32) {
    uint neighborIdx_sum_p2_sumBuf_32 = (lid + 32u);
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p2_sumBuf_16) {
    uint neighborIdx_sum_p2_sumBuf_16 = (lid + 16u);
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p2_sumBuf_8) {
    uint neighborIdx_sum_p2_sumBuf_8 = (lid + 8u);
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p2_sumBuf_4) {
    uint neighborIdx_sum_p2_sumBuf_4 = (lid + 4u);
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p2_sumBuf_2) {
    uint neighborIdx_sum_p2_sumBuf_2 = (lid + 2u);
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p2_sumBuf_1) {
    uint neighborIdx_sum_p2_sumBuf_1 = (lid + 1u);
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((batchValid && (lid == 0u))) {
    float outVal = sumBuf[0u];
    half outF16 = ((half)(outVal));
    uint outIdx = (qBase + 0u);
    _output[outIdx] = outF16;
  }
}

)kernel";
const char* k_metal_navatala_transformer_simple_attention_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_simple_attention_f32(device const float* query [[buffer(0)]], device const float* key [[buffer(1)]], device const float* value [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* seqLen [[buffer(4)]], device const uint* numHeads [[buffer(5)]], device const uint* headDim [[buffer(6)]], device float* _output [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint groupId = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  uint nh = numHeads[0u];
  uint hd = headDim[0u];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_type<float>(0x3f800000u) / sqrtHd);
  threadgroup float attnScores[256];
  threadgroup float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if (valid) {
    float dotProd = as_type<float>(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      float qVal = query[qIdx];
      float kVal = key[kIdx];
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = as_type<float>(0xf149f2cau);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxScore = attnScores[0u];
  if (valid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = sumBuf[0u];
  if (valid) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float v0 = value[vBase];
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p2_sumBuf_128) {
    uint neighborIdx_sum_p2_sumBuf_128 = (lid + 128u);
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p2_sumBuf_64) {
    uint neighborIdx_sum_p2_sumBuf_64 = (lid + 64u);
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p2_sumBuf_32) {
    uint neighborIdx_sum_p2_sumBuf_32 = (lid + 32u);
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p2_sumBuf_16) {
    uint neighborIdx_sum_p2_sumBuf_16 = (lid + 16u);
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p2_sumBuf_8) {
    uint neighborIdx_sum_p2_sumBuf_8 = (lid + 8u);
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p2_sumBuf_4) {
    uint neighborIdx_sum_p2_sumBuf_4 = (lid + 4u);
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p2_sumBuf_2) {
    uint neighborIdx_sum_p2_sumBuf_2 = (lid + 2u);
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p2_sumBuf_1) {
    uint neighborIdx_sum_p2_sumBuf_1 = (lid + 1u);
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((batchValid && (lid == 0u))) {
    float outVal = sumBuf[0u];
    uint outIdx = (qBase + 0u);
    _output[outIdx] = outVal;
  }
}

)kernel";
const char* k_metal_navatala_transformer_paged_attention_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_paged_attention_f32(device const float* query [[buffer(0)]], device const float* kvCache [[buffer(1)]], device const uint* blockTables [[buffer(2)]], device const uint* contextLens [[buffer(3)]], device const uint* numHeads [[buffer(4)]], device const uint* headDim [[buffer(5)]], device const uint* pageSize [[buffer(6)]], device const uint* maxBlocks [[buffer(7)]], device float* _output [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint groupId = ((uint)(int(__tgid.x)));
  uint nh = numHeads[0u];
  uint hd = headDim[0u];
  uint ps = pageSize[0u];
  uint mb = maxBlocks[0u];
  uint batchIdx = (groupId / nh);
  uint headIdx = (groupId % nh);
  uint ctxLen = contextLens[batchIdx];
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_type<float>(0x3f800000u) / sqrtHd);
  threadgroup float attnScores[256];
  threadgroup float sumBuf[256];
  bool kvValid = (lid < ctxLen);
  uint qBase = ((batchIdx * (nh * hd)) + (headIdx * hd));
  uint blockIdx = (lid / ps);
  uint blockOffset = (lid % ps);
  uint blockTableBase = (batchIdx * mb);
  uint physicalBlock = blockTables[(blockTableBase + blockIdx)];
  uint kvBlockStride = (2u * (nh * (ps * hd)));
  uint kvSlotStride = (nh * (ps * hd));
  uint kvHeadStride = (ps * hd);
  uint kBase = ((physicalBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (blockOffset * hd)));
  uint vBase = ((physicalBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (blockOffset * hd))));
  if (kvValid) {
    float dotProd = as_type<float>(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      float qVal = query[qIdx];
      float kVal = kvCache[kIdx];
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = as_type<float>(0xf149f2cau);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxScore = attnScores[0u];
  if (kvValid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = sumBuf[0u];
  for (int outD = 0; outD < (int)(hd); ++outD) {
    if (kvValid) {
      float myExpScore = attnScores[lid];
      float shiftedExp = (myExpScore - maxScore);
      float expVal = exp(shiftedExp);
      float attnWeight = (expVal / sumExp);
      uint vIdx = (vBase + outD);
      float vVal = kvCache[vIdx];
      float weighted = (attnWeight * vVal);
      sumBuf[lid] = weighted;
    } else {
      sumBuf[lid] = as_type<float>(0x00000000u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool shouldReduce_sum_p2_sumBuf_128 = (lid < 128u);
    if (shouldReduce_sum_p2_sumBuf_128) {
      uint neighborIdx_sum_p2_sumBuf_128 = (lid + 128u);
      float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
      float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool shouldReduce_sum_p2_sumBuf_64 = (lid < 64u);
    if (shouldReduce_sum_p2_sumBuf_64) {
      uint neighborIdx_sum_p2_sumBuf_64 = (lid + 64u);
      float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
      float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool shouldReduce_sum_p2_sumBuf_32 = (lid < 32u);
    if (shouldReduce_sum_p2_sumBuf_32) {
      uint neighborIdx_sum_p2_sumBuf_32 = (lid + 32u);
      float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
      float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool shouldReduce_sum_p2_sumBuf_16 = (lid < 16u);
    if (shouldReduce_sum_p2_sumBuf_16) {
      uint neighborIdx_sum_p2_sumBuf_16 = (lid + 16u);
      float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
      float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool shouldReduce_sum_p2_sumBuf_8 = (lid < 8u);
    if (shouldReduce_sum_p2_sumBuf_8) {
      uint neighborIdx_sum_p2_sumBuf_8 = (lid + 8u);
      float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
      float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool shouldReduce_sum_p2_sumBuf_4 = (lid < 4u);
    if (shouldReduce_sum_p2_sumBuf_4) {
      uint neighborIdx_sum_p2_sumBuf_4 = (lid + 4u);
      float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
      float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool shouldReduce_sum_p2_sumBuf_2 = (lid < 2u);
    if (shouldReduce_sum_p2_sumBuf_2) {
      uint neighborIdx_sum_p2_sumBuf_2 = (lid + 2u);
      float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
      float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool shouldReduce_sum_p2_sumBuf_1 = (lid < 1u);
    if (shouldReduce_sum_p2_sumBuf_1) {
      uint neighborIdx_sum_p2_sumBuf_1 = (lid + 1u);
      float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
      float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((lid == 0u)) {
      float outVal = sumBuf[0u];
      uint outIdx = (qBase + outD);
      _output[outIdx] = outVal;
    }
  }
}

)kernel";
const char* k_metal_navatala_transformer_simple_attention_with_mask_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_simple_attention_with_mask_f32(device const float* query [[buffer(0)]], device const float* key [[buffer(1)]], device const float* value [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* seqLen [[buffer(4)]], device const uint* numHeads [[buffer(5)]], device const uint* headDim [[buffer(6)]], device float* _output [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint groupId = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  uint nh = numHeads[0u];
  uint hd = headDim[0u];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_type<float>(0x3f800000u) / sqrtHd);
  threadgroup float attnScores[256];
  threadgroup float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  bool causalMasked = (lid > queryPos);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if ((valid && (!causalMasked))) {
    float dotProd = as_type<float>(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      float qVal = query[qIdx];
      float kVal = key[kIdx];
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = as_type<float>(0xf149f2cau);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxScore = attnScores[0u];
  if ((valid && (!causalMasked))) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = sumBuf[0u];
  if ((valid && (!causalMasked))) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float v0 = value[vBase];
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p2_sumBuf_128) {
    uint neighborIdx_sum_p2_sumBuf_128 = (lid + 128u);
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p2_sumBuf_64) {
    uint neighborIdx_sum_p2_sumBuf_64 = (lid + 64u);
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p2_sumBuf_32) {
    uint neighborIdx_sum_p2_sumBuf_32 = (lid + 32u);
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p2_sumBuf_16) {
    uint neighborIdx_sum_p2_sumBuf_16 = (lid + 16u);
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p2_sumBuf_8) {
    uint neighborIdx_sum_p2_sumBuf_8 = (lid + 8u);
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p2_sumBuf_4) {
    uint neighborIdx_sum_p2_sumBuf_4 = (lid + 4u);
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p2_sumBuf_2) {
    uint neighborIdx_sum_p2_sumBuf_2 = (lid + 2u);
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p2_sumBuf_1) {
    uint neighborIdx_sum_p2_sumBuf_1 = (lid + 1u);
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((batchValid && (lid == 0u))) {
    float outVal = sumBuf[0u];
    uint outIdx = (qBase + 0u);
    _output[outIdx] = outVal;
  }
}

)kernel";
const char* k_metal_navatala_transformer_simple_attention_with_mask_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_simple_attention_with_mask_f16(device const half* query [[buffer(0)]], device const half* key [[buffer(1)]], device const half* value [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* seqLen [[buffer(4)]], device const uint* numHeads [[buffer(5)]], device const uint* headDim [[buffer(6)]], device half* _output [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint groupId = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  uint nh = numHeads[0u];
  uint hd = headDim[0u];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_type<float>(0x3f800000u) / sqrtHd);
  threadgroup float attnScores[256];
  threadgroup float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  bool causalMasked = (lid > queryPos);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if ((valid && (!causalMasked))) {
    float dotProd = as_type<float>(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      half qValF16 = query[qIdx];
      half kValF16 = key[kIdx];
      float qVal = ((float)(qValF16));
      float kVal = ((float)(kValF16));
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = as_type<float>(0xf149f2cau);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxScore = attnScores[0u];
  if ((valid && (!causalMasked))) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = sumBuf[0u];
  if ((valid && (!causalMasked))) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    half v0F16 = value[vBase];
    float v0 = ((float)(v0F16));
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p2_sumBuf_128) {
    uint neighborIdx_sum_p2_sumBuf_128 = (lid + 128u);
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p2_sumBuf_64) {
    uint neighborIdx_sum_p2_sumBuf_64 = (lid + 64u);
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p2_sumBuf_32) {
    uint neighborIdx_sum_p2_sumBuf_32 = (lid + 32u);
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p2_sumBuf_16) {
    uint neighborIdx_sum_p2_sumBuf_16 = (lid + 16u);
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p2_sumBuf_8) {
    uint neighborIdx_sum_p2_sumBuf_8 = (lid + 8u);
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p2_sumBuf_4) {
    uint neighborIdx_sum_p2_sumBuf_4 = (lid + 4u);
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p2_sumBuf_2) {
    uint neighborIdx_sum_p2_sumBuf_2 = (lid + 2u);
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p2_sumBuf_1) {
    uint neighborIdx_sum_p2_sumBuf_1 = (lid + 1u);
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((batchValid && (lid == 0u))) {
    float outVal = sumBuf[0u];
    half outF16 = ((half)(outVal));
    uint outIdx = (qBase + 0u);
    _output[outIdx] = outF16;
  }
}

)kernel";
const char* k_metal_navatala_transformer_simple_attention_with_padding_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_simple_attention_with_padding_f32(device const float* query [[buffer(0)]], device const float* key [[buffer(1)]], device const float* value [[buffer(2)]], device const uint* seqLengths [[buffer(3)]], device const uint* batchSize [[buffer(4)]], device const uint* maxSeqLen [[buffer(5)]], device const uint* numHeads [[buffer(6)]], device const uint* headDim [[buffer(7)]], device float* _output [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint groupId = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = maxSeqLen[0u];
  uint nh = numHeads[0u];
  uint hd = headDim[0u];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  uint actualSeqLen = seqLengths[batchIdx];
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_type<float>(0x3f800000u) / sqrtHd);
  threadgroup float attnScores[256];
  threadgroup float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  bool paddingMasked = (lid >= actualSeqLen);
  bool queryMasked = (queryPos >= actualSeqLen);
  bool anyMasked = (paddingMasked || queryMasked);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if ((valid && (!anyMasked))) {
    float dotProd = as_type<float>(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      float qVal = query[qIdx];
      float kVal = key[kIdx];
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = as_type<float>(0xf149f2cau);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxScore = attnScores[0u];
  if ((valid && (!anyMasked))) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = sumBuf[0u];
  if ((valid && (!anyMasked))) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float v0 = value[vBase];
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p2_sumBuf_128) {
    uint neighborIdx_sum_p2_sumBuf_128 = (lid + 128u);
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p2_sumBuf_64) {
    uint neighborIdx_sum_p2_sumBuf_64 = (lid + 64u);
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p2_sumBuf_32) {
    uint neighborIdx_sum_p2_sumBuf_32 = (lid + 32u);
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p2_sumBuf_16) {
    uint neighborIdx_sum_p2_sumBuf_16 = (lid + 16u);
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p2_sumBuf_8) {
    uint neighborIdx_sum_p2_sumBuf_8 = (lid + 8u);
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p2_sumBuf_4) {
    uint neighborIdx_sum_p2_sumBuf_4 = (lid + 4u);
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p2_sumBuf_2) {
    uint neighborIdx_sum_p2_sumBuf_2 = (lid + 2u);
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p2_sumBuf_1) {
    uint neighborIdx_sum_p2_sumBuf_1 = (lid + 1u);
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((batchValid && (lid == 0u))) {
    float outVal = sumBuf[0u];
    uint outIdx = (qBase + 0u);
    _output[outIdx] = outVal;
  }
}

)kernel";
const char* k_metal_navatala_transformer_simple_attention_with_padding_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_simple_attention_with_padding_f16(device const half* query [[buffer(0)]], device const half* key [[buffer(1)]], device const half* value [[buffer(2)]], device const uint* seqLengths [[buffer(3)]], device const uint* batchSize [[buffer(4)]], device const uint* maxSeqLen [[buffer(5)]], device const uint* numHeads [[buffer(6)]], device const uint* headDim [[buffer(7)]], device half* _output [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint groupId = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = maxSeqLen[0u];
  uint nh = numHeads[0u];
  uint hd = headDim[0u];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  uint actualSeqLen = seqLengths[batchIdx];
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_type<float>(0x3f800000u) / sqrtHd);
  threadgroup float attnScores[256];
  threadgroup float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  bool paddingMasked = (lid >= actualSeqLen);
  bool queryMasked = (queryPos >= actualSeqLen);
  bool anyMasked = (paddingMasked || queryMasked);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if ((valid && (!anyMasked))) {
    float dotProd = as_type<float>(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      half qValF16 = query[qIdx];
      half kValF16 = key[kIdx];
      float qVal = ((float)(qValF16));
      float kVal = ((float)(kValF16));
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = as_type<float>(0xf149f2cau);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxScore = attnScores[0u];
  if ((valid && (!anyMasked))) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = sumBuf[0u];
  if ((valid && (!anyMasked))) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    half v0F16 = value[vBase];
    float v0 = ((float)(v0F16));
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p2_sumBuf_128) {
    uint neighborIdx_sum_p2_sumBuf_128 = (lid + 128u);
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p2_sumBuf_64) {
    uint neighborIdx_sum_p2_sumBuf_64 = (lid + 64u);
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p2_sumBuf_32) {
    uint neighborIdx_sum_p2_sumBuf_32 = (lid + 32u);
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p2_sumBuf_16) {
    uint neighborIdx_sum_p2_sumBuf_16 = (lid + 16u);
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p2_sumBuf_8) {
    uint neighborIdx_sum_p2_sumBuf_8 = (lid + 8u);
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p2_sumBuf_4) {
    uint neighborIdx_sum_p2_sumBuf_4 = (lid + 4u);
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p2_sumBuf_2) {
    uint neighborIdx_sum_p2_sumBuf_2 = (lid + 2u);
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p2_sumBuf_1) {
    uint neighborIdx_sum_p2_sumBuf_1 = (lid + 1u);
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((batchValid && (lid == 0u))) {
    float outVal = sumBuf[0u];
    half outF16 = ((half)(outVal));
    uint outIdx = (qBase + 0u);
    _output[outIdx] = outF16;
  }
}

)kernel";
const char* k_metal_navatala_transformer_simple_attention_backward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_simple_attention_backward_f32(device const float* gradOutput [[buffer(0)]], device const float* query [[buffer(1)]], device const float* key [[buffer(2)]], device const float* value [[buffer(3)]], device const uint* batchSize [[buffer(4)]], device const uint* seqLen [[buffer(5)]], device const uint* numHeads [[buffer(6)]], device const uint* headDim [[buffer(7)]], device float* gradQuery [[buffer(8)]], device float* gradKey [[buffer(9)]], device float* gradValue [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint groupId = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  uint nh = numHeads[0u];
  uint hd = headDim[0u];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_type<float>(0x3f800000u) / sqrtHd);
  threadgroup float attnScores[256];
  threadgroup float sumBuf[256];
  threadgroup float gradBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if (valid) {
    float dotProdQK = as_type<float>(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      float qVal = query[qIdx];
      float kVal = key[kIdx];
      float prod = (qVal * kVal);
      float oldDotProdQK = dotProdQK;
      dotProdQK = (oldDotProdQK + prod);
    }
    float finalDotProdQK = dotProdQK;
    float scaledScore = (finalDotProdQK * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = as_type<float>(0xf149f2cau);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxScore = attnScores[0u];
  if (valid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = sumBuf[0u];
  if (valid) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float dotProdGradAttn = as_type<float>(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint gradOutIdx = (qBase + ((uint)(d)));
      uint vIdx = (vBase + ((uint)(d)));
      float gradOutVal = gradOutput[gradOutIdx];
      float vVal = value[vIdx];
      float prodGradV = (gradOutVal * vVal);
      float oldDotProdGradAttn = dotProdGradAttn;
      dotProdGradAttn = (oldDotProdGradAttn + prodGradV);
      float gradVContrib = (attnWeight * gradOutVal);
      float oldGradV = gradValue[vIdx];
      float newGradV = (oldGradV + gradVContrib);
      gradValue[vIdx] = newGradV;
    }
    float finalGradAttn = dotProdGradAttn;
    gradBuf[lid] = finalGradAttn;
  } else {
    gradBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (valid) {
    float myAttnWeight = (exp((attnScores[lid] - maxScore)) / sumExp);
    float myGradAttn = gradBuf[lid];
    float gradScore = (myAttnWeight * myGradAttn);
    float scaledGradScore = (gradScore * scale);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      float kVal = key[kIdx];
      float qVal = query[qIdx];
      float gradQContrib = (scaledGradScore * kVal);
      float oldGradQ = gradQuery[qIdx];
      float newGradQ = (oldGradQ + gradQContrib);
      gradQuery[qIdx] = newGradQ;
      float gradKContrib = (scaledGradScore * qVal);
      float oldGradK = gradKey[kIdx];
      float newGradK = (oldGradK + gradKContrib);
      gradKey[kIdx] = newGradK;
    }
  }
}

)kernel";
const char* k_metal_navatala_transformer_simple_attention_backward_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_simple_attention_backward_f16(device const half* gradOutput [[buffer(0)]], device const half* query [[buffer(1)]], device const half* key [[buffer(2)]], device const half* value [[buffer(3)]], device const uint* batchSize [[buffer(4)]], device const uint* seqLen [[buffer(5)]], device const uint* numHeads [[buffer(6)]], device const uint* headDim [[buffer(7)]], device half* gradQuery [[buffer(8)]], device half* gradKey [[buffer(9)]], device half* gradValue [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint groupId = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  uint nh = numHeads[0u];
  uint hd = headDim[0u];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_type<float>(0x3f800000u) / sqrtHd);
  threadgroup float attnScores[256];
  threadgroup float sumBuf[256];
  threadgroup float gradBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if (valid) {
    float dotProd = as_type<float>(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      half qValF16 = query[qIdx];
      half kValF16 = key[kIdx];
      float qVal = ((float)(qValF16));
      float kVal = ((float)(kValF16));
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = as_type<float>(0xf149f2cau);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxScore = attnScores[0u];
  if (valid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = sumBuf[0u];
  if (valid) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float gradAttnAccum = as_type<float>(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint gradOutIdx = (qBase + ((uint)(d)));
      uint vIdx = (vBase + ((uint)(d)));
      half gradOutF16 = gradOutput[gradOutIdx];
      float gradOutVal = ((float)(gradOutF16));
      half vValF16 = value[vIdx];
      float vVal = ((float)(vValF16));
      float gradVContrib = (attnWeight * gradOutVal);
      half oldGradVF16 = gradValue[vIdx];
      float oldGradV = ((float)(oldGradVF16));
      float newGradV = (oldGradV + gradVContrib);
      half newGradVF16 = ((half)(newGradV));
      gradValue[vIdx] = newGradVF16;
      float gradAttnProd = (gradOutVal * vVal);
      float oldGradAttnAccum = gradAttnAccum;
      gradAttnAccum = (oldGradAttnAccum + gradAttnProd);
    }
    float gradAttn = gradAttnAccum;
    gradBuf[lid] = gradAttn;
  } else {
    gradBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (valid) {
    float myAttnWeight = (exp((attnScores[lid] - maxScore)) / sumExp);
    float myGradAttn = gradBuf[lid];
    float gradScore = (myAttnWeight * myGradAttn);
    float scaledGradScore = (gradScore * scale);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      half kValF16 = key[kIdx];
      float kVal = ((float)(kValF16));
      float gradQContrib = (scaledGradScore * kVal);
      half oldGradQF16 = gradQuery[qIdx];
      float oldGradQ = ((float)(oldGradQF16));
      float newGradQ = (oldGradQ + gradQContrib);
      half newGradQF16 = ((half)(newGradQ));
      gradQuery[qIdx] = newGradQF16;
      half qValF16 = query[qIdx];
      float qVal = ((float)(qValF16));
      float gradKContrib = (scaledGradScore * qVal);
      half oldGradKF16 = gradKey[kIdx];
      float oldGradK = ((float)(oldGradKF16));
      float newGradK = (oldGradK + gradKContrib);
      half newGradKF16 = ((half)(newGradK));
      gradKey[kIdx] = newGradKF16;
    }
  }
}

)kernel";
const char* k_metal_navatala_transformer_kv_cache_append_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_kv_cache_append_f16(device const half* newK [[buffer(0)]], device const half* newV [[buffer(1)]], device const uint* blockTables [[buffer(2)]], device const uint* seqPositions [[buffer(3)]], device const uint* numHeads [[buffer(4)]], device const uint* headDim [[buffer(5)]], device const uint* pageSize [[buffer(6)]], device const uint* maxBlocks [[buffer(7)]], device half* kvCache [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint groupId = ((uint)(int(__tgid.x)));
  uint nh = numHeads[0u];
  uint hd = headDim[0u];
  uint ps = pageSize[0u];
  uint mb = maxBlocks[0u];
  uint batchIdx = (groupId / nh);
  uint headIdx = (groupId % nh);
  uint seqPos = seqPositions[batchIdx];
  uint blockIdx = (seqPos / ps);
  uint blockOffset = (seqPos % ps);
  uint blockTableBase = (batchIdx * mb);
  uint physicalBlock = blockTables[(blockTableBase + blockIdx)];
  uint kvBlockStride = (2u * (nh * (ps * hd)));
  uint kvSlotStride = (nh * (ps * hd));
  uint kvHeadStride = (ps * hd);
  uint kCacheBase = ((physicalBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (blockOffset * hd)));
  uint vCacheBase = ((physicalBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (blockOffset * hd))));
  uint srcBase = ((batchIdx * (nh * hd)) + (headIdx * hd));
  bool dimValid = (lid < hd);
  if (dimValid) {
    uint srcIdx = (srcBase + lid);
    uint kDstIdx = (kCacheBase + lid);
    uint vDstIdx = (vCacheBase + lid);
    half kVal = newK[srcIdx];
    half vVal = newV[srcIdx];
    kvCache[kDstIdx] = kVal;
    kvCache[vDstIdx] = vVal;
  }
}

)kernel";
const char* k_metal_navatala_transformer_kv_cache_append_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_kv_cache_append_f32(device const float* newK [[buffer(0)]], device const float* newV [[buffer(1)]], device const uint* blockTables [[buffer(2)]], device const uint* seqPositions [[buffer(3)]], device const uint* numHeads [[buffer(4)]], device const uint* headDim [[buffer(5)]], device const uint* pageSize [[buffer(6)]], device const uint* maxBlocks [[buffer(7)]], device float* kvCache [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint groupId = ((uint)(int(__tgid.x)));
  uint nh = numHeads[0u];
  uint hd = headDim[0u];
  uint ps = pageSize[0u];
  uint mb = maxBlocks[0u];
  uint batchIdx = (groupId / nh);
  uint headIdx = (groupId % nh);
  uint seqPos = seqPositions[batchIdx];
  uint blockIdx = (seqPos / ps);
  uint blockOffset = (seqPos % ps);
  uint blockTableBase = (batchIdx * mb);
  uint physicalBlock = blockTables[(blockTableBase + blockIdx)];
  uint kvBlockStride = (2u * (nh * (ps * hd)));
  uint kvSlotStride = (nh * (ps * hd));
  uint kvHeadStride = (ps * hd);
  uint kCacheBase = ((physicalBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (blockOffset * hd)));
  uint vCacheBase = ((physicalBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (blockOffset * hd))));
  uint srcBase = ((batchIdx * (nh * hd)) + (headIdx * hd));
  bool dimValid = (lid < hd);
  if (dimValid) {
    uint srcIdx = (srcBase + lid);
    uint kDstIdx = (kCacheBase + lid);
    uint vDstIdx = (vCacheBase + lid);
    float kVal = newK[srcIdx];
    float vVal = newV[srcIdx];
    kvCache[kDstIdx] = kVal;
    kvCache[vDstIdx] = vVal;
  }
}

)kernel";
const char* k_metal_navatala_transformer_kv_cache_rotate_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_kv_cache_rotate_f16(device const uint* blockTables [[buffer(0)]], device const uint* contextLens [[buffer(1)]], device const uint* shiftAmount [[buffer(2)]], device const uint* numHeads [[buffer(3)]], device const uint* headDim [[buffer(4)]], device const uint* pageSize [[buffer(5)]], device const uint* maxBlocks [[buffer(6)]], device half* kvCache [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint groupId = ((uint)(int(__tgid.x)));
  uint nh = numHeads[0u];
  uint hd = headDim[0u];
  uint ps = pageSize[0u];
  uint mb = maxBlocks[0u];
  uint shift = shiftAmount[0u];
  uint batchIdx = (groupId / nh);
  uint headIdx = (groupId % nh);
  uint ctxLen = contextLens[batchIdx];
  bool posValid = (lid < ctxLen);
  uint srcPos = (lid + shift);
  bool srcValid = (srcPos < ctxLen);
  bool valid = (posValid && srcValid);
  uint blockTableBase = (batchIdx * mb);
  uint kvBlockStride = (2u * (nh * (ps * hd)));
  uint kvSlotStride = (nh * (ps * hd));
  uint kvHeadStride = (ps * hd);
  if (valid) {
    uint srcBlockIdx = (srcPos / ps);
    uint srcBlockOffset = (srcPos % ps);
    uint srcPhysBlock = blockTables[(blockTableBase + srcBlockIdx)];
    uint dstBlockIdx = (lid / ps);
    uint dstBlockOffset = (lid % ps);
    uint dstPhysBlock = blockTables[(blockTableBase + dstBlockIdx)];
    uint kSrcBase = ((srcPhysBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (srcBlockOffset * hd)));
    uint kDstBase = ((dstPhysBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (dstBlockOffset * hd)));
    uint vSrcBase = ((srcPhysBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (srcBlockOffset * hd))));
    uint vDstBase = ((dstPhysBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (dstBlockOffset * hd))));
    for (int d = 0; d < (int)(hd); ++d) {
      uint kSrcIdx = (kSrcBase + ((uint)(d)));
      uint kDstIdx = (kDstBase + ((uint)(d)));
      uint vSrcIdx = (vSrcBase + ((uint)(d)));
      uint vDstIdx = (vDstBase + ((uint)(d)));
      half kVal = kvCache[kSrcIdx];
      half vVal = kvCache[vSrcIdx];
      kvCache[kDstIdx] = kVal;
      kvCache[vDstIdx] = vVal;
    }
  }
}

)kernel";
const char* k_metal_navatala_transformer_tiled_gemm_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_tiled_gemm_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* m [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device const float* alpha [[buffer(5)]], device const float* beta [[buffer(6)]], device half* c [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tidX = ((uint)(int(__tid.x)));
  uint tidY = ((uint)(int(__tid.y)));
  uint gidX = ((uint)(int(__tgid.x)));
  uint gidY = ((uint)(int(__tgid.y)));
  uint mDim = m[0u];
  uint nDim = n[0u];
  uint kDim = k[0u];
  float alphaVal = alpha[0u];
  float betaVal = beta[0u];
  uint tileSize = 16u;
  uint row = ((gidY * tileSize) + tidY);
  uint col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < nDim);
  bool valid = (rowValid && colValid);
  threadgroup float tileA[256];
  threadgroup float tileB[256];
  float acc = as_type<float>(0x00000000u);
  uint tileCount = ((kDim + (tileSize - 1u)) / tileSize);
  for (int tileIter = 0; tileIter < (int)(tileCount); ++tileIter) {
    uint tileIterU32 = ((uint)(tileIter));
    uint tileBase = (tileIterU32 * tileSize);
    uint aK = (tileBase + tidX);
    uint bK = (tileBase + tidY);
    uint tileOffset = ((tidY * tileSize) + tidX);
    bool aKValid = (aK < kDim);
    bool bKValid = (bK < kDim);
    bool aLoadValid = (rowValid && aKValid);
    bool bLoadValid = (colValid && bKValid);
    uint aIdx = ((row * kDim) + aK);
    uint bIdx = ((bK * nDim) + col);
    half aValF16 = ((aLoadValid) ? (a[aIdx]) : (half(0.000000)));
    half bValF16 = ((bLoadValid) ? (b[bIdx]) : (half(0.000000)));
    float aTileVal = ((float)(aValF16));
    float bTileVal = ((float)(bValF16));
    tileA[tileOffset] = aTileVal;
    tileB[tileOffset] = bTileVal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < (int)(tileSize); ++kk) {
      uint kkU32 = ((uint)(kk));
      uint tileAIdx = ((tidY * tileSize) + kkU32);
      uint tileBIdx = ((kkU32 * tileSize) + tidX);
      float aTile = tileA[tileAIdx];
      float bTile = tileB[tileBIdx];
      float prod = (aTile * bTile);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float finalAcc = acc;
  if (valid) {
    uint cIdx = ((row * nDim) + col);
    half cOldF16 = c[cIdx];
    float cOld = ((float)(cOldF16));
    float scaledProd = (alphaVal * finalAcc);
    float scaledOld = (betaVal * cOld);
    float result = (scaledProd + scaledOld);
    half resultF16 = ((half)(result));
    c[cIdx] = resultF16;
  }
}

)kernel";
const char* k_metal_navatala_transformer_tiled_gemm_f16_f32_out = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_tiled_gemm_f16_f32_out(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* m [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device const float* alpha [[buffer(5)]], device const float* beta [[buffer(6)]], device float* c [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tidX = ((uint)(int(__tid.x)));
  uint tidY = ((uint)(int(__tid.y)));
  uint gidX = ((uint)(int(__tgid.x)));
  uint gidY = ((uint)(int(__tgid.y)));
  uint mDim = m[0u];
  uint nDim = n[0u];
  uint kDim = k[0u];
  float alphaVal = alpha[0u];
  float betaVal = beta[0u];
  uint tileSize = 16u;
  uint row = ((gidY * tileSize) + tidY);
  uint col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < nDim);
  bool valid = (rowValid && colValid);
  threadgroup float tileA[256];
  threadgroup float tileB[256];
  float acc = as_type<float>(0x00000000u);
  uint tileCount = ((kDim + (tileSize - 1u)) / tileSize);
  for (int tileIter = 0; tileIter < (int)(tileCount); ++tileIter) {
    uint tileIterU32 = ((uint)(tileIter));
    uint tileBase = (tileIterU32 * tileSize);
    uint aK = (tileBase + tidX);
    uint bK = (tileBase + tidY);
    uint tileOffset = ((tidY * tileSize) + tidX);
    bool aKValid = (aK < kDim);
    bool bKValid = (bK < kDim);
    bool aLoadValid = (rowValid && aKValid);
    bool bLoadValid = (colValid && bKValid);
    uint aIdx = ((row * kDim) + aK);
    uint bIdx = ((bK * nDim) + col);
    half aValF16 = ((aLoadValid) ? (a[aIdx]) : (half(0.000000)));
    half bValF16 = ((bLoadValid) ? (b[bIdx]) : (half(0.000000)));
    float aTileVal = ((float)(aValF16));
    float bTileVal = ((float)(bValF16));
    tileA[tileOffset] = aTileVal;
    tileB[tileOffset] = bTileVal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < (int)(tileSize); ++kk) {
      uint kkU32 = ((uint)(kk));
      uint tileAIdx = ((tidY * tileSize) + kkU32);
      uint tileBIdx = ((kkU32 * tileSize) + tidX);
      float aTile = tileA[tileAIdx];
      float bTile = tileB[tileBIdx];
      float prod = (aTile * bTile);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float finalAcc = acc;
  if (valid) {
    uint cIdx = ((row * nDim) + col);
    float cOld = c[cIdx];
    float scaledProd = (alphaVal * finalAcc);
    float scaledOld = (betaVal * cOld);
    float result = (scaledProd + scaledOld);
    c[cIdx] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_tiled_gemm_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_tiled_gemm_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* m [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device const float* alpha [[buffer(5)]], device const float* beta [[buffer(6)]], device float* c [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tidX = ((uint)(int(__tid.x)));
  uint tidY = ((uint)(int(__tid.y)));
  uint gidX = ((uint)(int(__tgid.x)));
  uint gidY = ((uint)(int(__tgid.y)));
  uint mDim = m[0u];
  uint nDim = n[0u];
  uint kDim = k[0u];
  float alphaVal = alpha[0u];
  float betaVal = beta[0u];
  uint tileSize = 16u;
  uint row = ((gidY * tileSize) + tidY);
  uint col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < nDim);
  bool valid = (rowValid && colValid);
  threadgroup float tileA[256];
  threadgroup float tileB[256];
  float acc = as_type<float>(0x00000000u);
  uint tileCount = ((kDim + (tileSize - 1u)) / tileSize);
  for (int tileIter = 0; tileIter < (int)(tileCount); ++tileIter) {
    uint tileIterU32 = ((uint)(tileIter));
    uint tileBase = (tileIterU32 * tileSize);
    uint aK = (tileBase + tidX);
    uint bK = (tileBase + tidY);
    uint tileOffset = ((tidY * tileSize) + tidX);
    bool aKValid = (aK < kDim);
    bool bKValid = (bK < kDim);
    bool aLoadValid = (rowValid && aKValid);
    bool bLoadValid = (colValid && bKValid);
    uint aIdx = ((row * kDim) + aK);
    uint bIdx = ((bK * nDim) + col);
    float aTileVal = ((aLoadValid) ? (a[aIdx]) : (as_type<float>(0x00000000u)));
    float bTileVal = ((bLoadValid) ? (b[bIdx]) : (as_type<float>(0x00000000u)));
    tileA[tileOffset] = aTileVal;
    tileB[tileOffset] = bTileVal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < (int)(tileSize); ++kk) {
      uint kkU32 = ((uint)(kk));
      uint tileAIdx = ((tidY * tileSize) + kkU32);
      uint tileBIdx = ((kkU32 * tileSize) + tidX);
      float aTile = tileA[tileAIdx];
      float bTile = tileB[tileBIdx];
      float prod = (aTile * bTile);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float finalAcc = acc;
  if (valid) {
    uint cIdx = ((row * nDim) + col);
    float cOld = c[cIdx];
    float scaledProd = (alphaVal * finalAcc);
    float scaledOld = (betaVal * cOld);
    float result = (scaledProd + scaledOld);
    c[cIdx] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_tiled_gemm_backward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_tiled_gemm_backward_f32(device const float* gradC [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* m [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device float* gradA [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tidX = ((uint)(int(__tid.x)));
  uint tidY = ((uint)(int(__tid.y)));
  uint gidX = ((uint)(int(__tgid.x)));
  uint gidY = ((uint)(int(__tgid.y)));
  uint mDim = m[0u];
  uint nDim = n[0u];
  uint kDim = k[0u];
  uint tileSize = 16u;
  uint row = ((gidY * tileSize) + tidY);
  uint col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < kDim);
  bool valid = (rowValid && colValid);
  threadgroup float tileGradC[256];
  threadgroup float tileBT[256];
  if (valid) {
    float acc = as_type<float>(0x00000000u);
    for (int jIter = 0; jIter < (int)(nDim); ++jIter) {
      uint gradCIdx = ((row * nDim) + jIter);
      uint bIdx = ((col * nDim) + jIter);
      float gradCVal = gradC[gradCIdx];
      float bVal = b[bIdx];
      float prod = (gradCVal * bVal);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    float finalAcc = acc;
    uint gradAIdx = ((row * kDim) + col);
    gradA[gradAIdx] = finalAcc;
  }
}

)kernel";
const char* k_metal_navatala_transformer_tiled_gemm_backward_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_tiled_gemm_backward_f16(device const half* gradC [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* m [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device half* gradA [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tidX = ((uint)(int(__tid.x)));
  uint tidY = ((uint)(int(__tid.y)));
  uint gidX = ((uint)(int(__tgid.x)));
  uint gidY = ((uint)(int(__tgid.y)));
  uint mDim = m[0u];
  uint nDim = n[0u];
  uint kDim = k[0u];
  uint tileSize = 16u;
  uint row = ((gidY * tileSize) + tidY);
  uint col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < kDim);
  bool valid = (rowValid && colValid);
  threadgroup float tileGradC[256];
  threadgroup float tileBT[256];
  if (valid) {
    float acc = as_type<float>(0x00000000u);
    for (int jIter = 0; jIter < (int)(nDim); ++jIter) {
      uint gradCIdx = ((row * nDim) + jIter);
      uint bIdx = ((col * nDim) + jIter);
      half gradCValF16 = gradC[gradCIdx];
      half bValF16 = b[bIdx];
      float gradCVal = ((float)(gradCValF16));
      float bVal = ((float)(bValF16));
      float prod = (gradCVal * bVal);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    float finalAcc = acc;
    half finalAccF16 = ((half)(finalAcc));
    uint gradAIdx = ((row * kDim) + col);
    gradA[gradAIdx] = finalAccF16;
  }
}

)kernel";
const char* k_metal_navatala_transformer_multi_tensor_scale_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_multi_tensor_scale_f32(device const float* _input [[buffer(0)]], device const float* scale [[buffer(1)]], device const uint* totalElements [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((((uint)(int(__tgid.x))) * 256u) + ((uint)(int(__tid.x))));
  uint total = totalElements[0u];
  float s = scale[0u];
  bool valid = (gid < total);
  if (valid) {
    float x = _input[gid];
    float result = (x * s);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_multi_tensor_scale_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_multi_tensor_scale_f16(device const half* _input [[buffer(0)]], device const float* scale [[buffer(1)]], device const uint* totalElements [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((((uint)(int(__tgid.x))) * 256u) + ((uint)(int(__tid.x))));
  uint total = totalElements[0u];
  float s = scale[0u];
  bool valid = (gid < total);
  if (valid) {
    half xF16 = _input[gid];
    float x = ((float)(xF16));
    float resultF32 = (x * s);
    half result = ((half)(resultF32));
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_multi_tensor_add_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_multi_tensor_add_f32(device const float* _input [[buffer(0)]], device const float* value [[buffer(1)]], device const uint* totalElements [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((((uint)(int(__tgid.x))) * 256u) + ((uint)(int(__tid.x))));
  uint total = totalElements[0u];
  float v = value[0u];
  bool valid = (gid < total);
  if (valid) {
    float x = _input[gid];
    float result = (x + v);
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_multi_tensor_add_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_multi_tensor_add_f16(device const half* _input [[buffer(0)]], device const float* value [[buffer(1)]], device const uint* totalElements [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((((uint)(int(__tgid.x))) * 256u) + ((uint)(int(__tid.x))));
  uint total = totalElements[0u];
  float v = value[0u];
  bool valid = (gid < total);
  if (valid) {
    half xF16 = _input[gid];
    float x = ((float)(xF16));
    float resultF32 = (x + v);
    half result = ((half)(resultF32));
    _output[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_transformer_multi_tensor_copy_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_multi_tensor_copy_f32(device const float* _input [[buffer(0)]], device const uint* totalElements [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((((uint)(int(__tgid.x))) * 256u) + ((uint)(int(__tid.x))));
  uint total = totalElements[0u];
  bool valid = (gid < total);
  if (valid) {
    float x = _input[gid];
    _output[gid] = x;
  }
}

)kernel";
const char* k_metal_navatala_transformer_multi_tensor_copy_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_multi_tensor_copy_f16(device const half* _input [[buffer(0)]], device const uint* totalElements [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((((uint)(int(__tgid.x))) * 256u) + ((uint)(int(__tid.x))));
  uint total = totalElements[0u];
  bool valid = (gid < total);
  if (valid) {
    half x = _input[gid];
    _output[gid] = x;
  }
}

)kernel";
const char* k_metal_navatala_transformer_multi_tensor_l2_norm_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_multi_tensor_l2_norm_f32(device const float* _input [[buffer(0)]], device const uint* totalElements [[buffer(1)]], device float* partialSums [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint gid = ((uint)(int(__tgid.x)));
  uint globalIdx = ((gid * 256u) + lid);
  uint total = totalElements[0u];
  threadgroup float sumSq[256];
  bool valid = (globalIdx < total);
  float x = ((valid) ? (_input[globalIdx]) : (as_type<float>(0x00000000u)));
  float xSq = (x * x);
  sumSq[lid] = xSq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_128 = (lid < 128u);
  if (shouldReduce_sumSq_128) {
    uint neighborIdx_sumSq_128 = (lid + 128u);
    float myVal_sumSq_128 = sumSq[lid];
    float neighborVal_sumSq_128 = sumSq[neighborIdx_sumSq_128];
    float sumVal_sumSq_128 = (myVal_sumSq_128 + neighborVal_sumSq_128);
    sumSq[lid] = sumVal_sumSq_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_64 = (lid < 64u);
  if (shouldReduce_sumSq_64) {
    uint neighborIdx_sumSq_64 = (lid + 64u);
    float myVal_sumSq_64 = sumSq[lid];
    float neighborVal_sumSq_64 = sumSq[neighborIdx_sumSq_64];
    float sumVal_sumSq_64 = (myVal_sumSq_64 + neighborVal_sumSq_64);
    sumSq[lid] = sumVal_sumSq_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_32 = (lid < 32u);
  if (shouldReduce_sumSq_32) {
    uint neighborIdx_sumSq_32 = (lid + 32u);
    float myVal_sumSq_32 = sumSq[lid];
    float neighborVal_sumSq_32 = sumSq[neighborIdx_sumSq_32];
    float sumVal_sumSq_32 = (myVal_sumSq_32 + neighborVal_sumSq_32);
    sumSq[lid] = sumVal_sumSq_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_16 = (lid < 16u);
  if (shouldReduce_sumSq_16) {
    uint neighborIdx_sumSq_16 = (lid + 16u);
    float myVal_sumSq_16 = sumSq[lid];
    float neighborVal_sumSq_16 = sumSq[neighborIdx_sumSq_16];
    float sumVal_sumSq_16 = (myVal_sumSq_16 + neighborVal_sumSq_16);
    sumSq[lid] = sumVal_sumSq_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_8 = (lid < 8u);
  if (shouldReduce_sumSq_8) {
    uint neighborIdx_sumSq_8 = (lid + 8u);
    float myVal_sumSq_8 = sumSq[lid];
    float neighborVal_sumSq_8 = sumSq[neighborIdx_sumSq_8];
    float sumVal_sumSq_8 = (myVal_sumSq_8 + neighborVal_sumSq_8);
    sumSq[lid] = sumVal_sumSq_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_4 = (lid < 4u);
  if (shouldReduce_sumSq_4) {
    uint neighborIdx_sumSq_4 = (lid + 4u);
    float myVal_sumSq_4 = sumSq[lid];
    float neighborVal_sumSq_4 = sumSq[neighborIdx_sumSq_4];
    float sumVal_sumSq_4 = (myVal_sumSq_4 + neighborVal_sumSq_4);
    sumSq[lid] = sumVal_sumSq_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_2 = (lid < 2u);
  if (shouldReduce_sumSq_2) {
    uint neighborIdx_sumSq_2 = (lid + 2u);
    float myVal_sumSq_2 = sumSq[lid];
    float neighborVal_sumSq_2 = sumSq[neighborIdx_sumSq_2];
    float sumVal_sumSq_2 = (myVal_sumSq_2 + neighborVal_sumSq_2);
    sumSq[lid] = sumVal_sumSq_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_1 = (lid < 1u);
  if (shouldReduce_sumSq_1) {
    uint neighborIdx_sumSq_1 = (lid + 1u);
    float myVal_sumSq_1 = sumSq[lid];
    float neighborVal_sumSq_1 = sumSq[neighborIdx_sumSq_1];
    float sumVal_sumSq_1 = (myVal_sumSq_1 + neighborVal_sumSq_1);
    sumSq[lid] = sumVal_sumSq_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((lid == 0u)) {
    float partialSum = sumSq[0u];
    partialSums[gid] = partialSum;
  }
}

)kernel";
const char* k_metal_navatala_transformer_multi_tensor_l2_norm_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_multi_tensor_l2_norm_f16(device const half* _input [[buffer(0)]], device const uint* totalElements [[buffer(1)]], device float* partialSums [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint gid = ((uint)(int(__tgid.x)));
  uint globalIdx = ((gid * 256u) + lid);
  uint total = totalElements[0u];
  threadgroup float sumSq[256];
  bool valid = (globalIdx < total);
  half xF16 = ((valid) ? (_input[globalIdx]) : (half(0.000000)));
  float x = ((float)(xF16));
  float xSq = (x * x);
  sumSq[lid] = xSq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_128 = (lid < 128u);
  if (shouldReduce_sumSq_128) {
    uint neighborIdx_sumSq_128 = (lid + 128u);
    float myVal_sumSq_128 = sumSq[lid];
    float neighborVal_sumSq_128 = sumSq[neighborIdx_sumSq_128];
    float sumVal_sumSq_128 = (myVal_sumSq_128 + neighborVal_sumSq_128);
    sumSq[lid] = sumVal_sumSq_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_64 = (lid < 64u);
  if (shouldReduce_sumSq_64) {
    uint neighborIdx_sumSq_64 = (lid + 64u);
    float myVal_sumSq_64 = sumSq[lid];
    float neighborVal_sumSq_64 = sumSq[neighborIdx_sumSq_64];
    float sumVal_sumSq_64 = (myVal_sumSq_64 + neighborVal_sumSq_64);
    sumSq[lid] = sumVal_sumSq_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_32 = (lid < 32u);
  if (shouldReduce_sumSq_32) {
    uint neighborIdx_sumSq_32 = (lid + 32u);
    float myVal_sumSq_32 = sumSq[lid];
    float neighborVal_sumSq_32 = sumSq[neighborIdx_sumSq_32];
    float sumVal_sumSq_32 = (myVal_sumSq_32 + neighborVal_sumSq_32);
    sumSq[lid] = sumVal_sumSq_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_16 = (lid < 16u);
  if (shouldReduce_sumSq_16) {
    uint neighborIdx_sumSq_16 = (lid + 16u);
    float myVal_sumSq_16 = sumSq[lid];
    float neighborVal_sumSq_16 = sumSq[neighborIdx_sumSq_16];
    float sumVal_sumSq_16 = (myVal_sumSq_16 + neighborVal_sumSq_16);
    sumSq[lid] = sumVal_sumSq_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_8 = (lid < 8u);
  if (shouldReduce_sumSq_8) {
    uint neighborIdx_sumSq_8 = (lid + 8u);
    float myVal_sumSq_8 = sumSq[lid];
    float neighborVal_sumSq_8 = sumSq[neighborIdx_sumSq_8];
    float sumVal_sumSq_8 = (myVal_sumSq_8 + neighborVal_sumSq_8);
    sumSq[lid] = sumVal_sumSq_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_4 = (lid < 4u);
  if (shouldReduce_sumSq_4) {
    uint neighborIdx_sumSq_4 = (lid + 4u);
    float myVal_sumSq_4 = sumSq[lid];
    float neighborVal_sumSq_4 = sumSq[neighborIdx_sumSq_4];
    float sumVal_sumSq_4 = (myVal_sumSq_4 + neighborVal_sumSq_4);
    sumSq[lid] = sumVal_sumSq_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_2 = (lid < 2u);
  if (shouldReduce_sumSq_2) {
    uint neighborIdx_sumSq_2 = (lid + 2u);
    float myVal_sumSq_2 = sumSq[lid];
    float neighborVal_sumSq_2 = sumSq[neighborIdx_sumSq_2];
    float sumVal_sumSq_2 = (myVal_sumSq_2 + neighborVal_sumSq_2);
    sumSq[lid] = sumVal_sumSq_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_1 = (lid < 1u);
  if (shouldReduce_sumSq_1) {
    uint neighborIdx_sumSq_1 = (lid + 1u);
    float myVal_sumSq_1 = sumSq[lid];
    float neighborVal_sumSq_1 = sumSq[neighborIdx_sumSq_1];
    float sumVal_sumSq_1 = (myVal_sumSq_1 + neighborVal_sumSq_1);
    sumSq[lid] = sumVal_sumSq_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((lid == 0u)) {
    float partialSum = sumSq[0u];
    partialSums[gid] = partialSum;
  }
}

)kernel";
const char* k_metal_navatala_transformer_multi_tensor_clip_grad_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_multi_tensor_clip_grad_f32(device const float* totalNorm [[buffer(0)]], device const float* maxNorm [[buffer(1)]], device const uint* totalElements [[buffer(2)]], device float* grad [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((((uint)(int(__tgid.x))) * 256u) + ((uint)(int(__tid.x))));
  uint total = totalElements[0u];
  float tNorm = totalNorm[0u];
  float mNorm = maxNorm[0u];
  float epsilon = as_type<float>(0x358637bdu);
  float safeTNorm = (tNorm + epsilon);
  float ratio = (mNorm / safeTNorm);
  bool needsClip = (tNorm > mNorm);
  float clipCoef = ((needsClip) ? (ratio) : (as_type<float>(0x3f800000u)));
  bool valid = (gid < total);
  if (valid) {
    float g = grad[gid];
    float clipped = (g * clipCoef);
    grad[gid] = clipped;
  }
}

)kernel";
const char* k_metal_navatala_transformer_multi_tensor_clip_grad_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_multi_tensor_clip_grad_f16(device const float* totalNorm [[buffer(0)]], device const float* maxNorm [[buffer(1)]], device const uint* totalElements [[buffer(2)]], device half* grad [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((((uint)(int(__tgid.x))) * 256u) + ((uint)(int(__tid.x))));
  uint total = totalElements[0u];
  float tNorm = totalNorm[0u];
  float mNorm = maxNorm[0u];
  float epsilon = as_type<float>(0x358637bdu);
  float safeTNorm = (tNorm + epsilon);
  float ratio = (mNorm / safeTNorm);
  bool needsClip = (tNorm > mNorm);
  float clipCoef = ((needsClip) ? (ratio) : (as_type<float>(0x3f800000u)));
  bool valid = (gid < total);
  if (valid) {
    half gF16 = grad[gid];
    float g = ((float)(gF16));
    float clippedF32 = (g * clipCoef);
    half clipped = ((half)(clippedF32));
    grad[gid] = clipped;
  }
}

)kernel";
const char* k_metal_navatala_transformer_top_k_gating_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_top_k_gating_f32(device const float* routerLogits [[buffer(0)]], device const uint* numTokens [[buffer(1)]], device const uint* numExperts [[buffer(2)]], device const uint* topK [[buffer(3)]], device int* expertIndices [[buffer(4)]], device float* expertWeights [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint tokenIdx = ((uint)(int(__tgid.x)));
  uint nt = numTokens[0u];
  uint ne = numExperts[0u];
  uint k = topK[0u];
  threadgroup float logitsBuf[256];
  threadgroup float expBuf[256];
  threadgroup int indexBuf[256];
  bool tokenValid = (tokenIdx < nt);
  bool expertValid = (lid < ne);
  bool valid = (tokenValid && expertValid);
  uint globalIdx = ((tokenIdx * ne) + lid);
  float logit = ((valid) ? (routerLogits[globalIdx]) : (as_type<float>(0xf149f2cau)));
  logitsBuf[lid] = logit;
  int lidI32 = ((int)(lid));
  indexBuf[lid] = lidI32;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_128 = (lid < 128u);
  if (shouldReduce_max_logitsBuf_128) {
    uint neighborIdx_max_logitsBuf_128 = (lid + 128u);
    float myVal_max_logitsBuf_128 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_128 = logitsBuf[neighborIdx_max_logitsBuf_128];
    bool isGreater_max_logitsBuf_128 = (neighborVal_max_logitsBuf_128 > myVal_max_logitsBuf_128);
    float maxVal_max_logitsBuf_128 = ((isGreater_max_logitsBuf_128) ? (neighborVal_max_logitsBuf_128) : (myVal_max_logitsBuf_128));
    logitsBuf[lid] = maxVal_max_logitsBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_64 = (lid < 64u);
  if (shouldReduce_max_logitsBuf_64) {
    uint neighborIdx_max_logitsBuf_64 = (lid + 64u);
    float myVal_max_logitsBuf_64 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_64 = logitsBuf[neighborIdx_max_logitsBuf_64];
    bool isGreater_max_logitsBuf_64 = (neighborVal_max_logitsBuf_64 > myVal_max_logitsBuf_64);
    float maxVal_max_logitsBuf_64 = ((isGreater_max_logitsBuf_64) ? (neighborVal_max_logitsBuf_64) : (myVal_max_logitsBuf_64));
    logitsBuf[lid] = maxVal_max_logitsBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_32 = (lid < 32u);
  if (shouldReduce_max_logitsBuf_32) {
    uint neighborIdx_max_logitsBuf_32 = (lid + 32u);
    float myVal_max_logitsBuf_32 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_32 = logitsBuf[neighborIdx_max_logitsBuf_32];
    bool isGreater_max_logitsBuf_32 = (neighborVal_max_logitsBuf_32 > myVal_max_logitsBuf_32);
    float maxVal_max_logitsBuf_32 = ((isGreater_max_logitsBuf_32) ? (neighborVal_max_logitsBuf_32) : (myVal_max_logitsBuf_32));
    logitsBuf[lid] = maxVal_max_logitsBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_16 = (lid < 16u);
  if (shouldReduce_max_logitsBuf_16) {
    uint neighborIdx_max_logitsBuf_16 = (lid + 16u);
    float myVal_max_logitsBuf_16 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_16 = logitsBuf[neighborIdx_max_logitsBuf_16];
    bool isGreater_max_logitsBuf_16 = (neighborVal_max_logitsBuf_16 > myVal_max_logitsBuf_16);
    float maxVal_max_logitsBuf_16 = ((isGreater_max_logitsBuf_16) ? (neighborVal_max_logitsBuf_16) : (myVal_max_logitsBuf_16));
    logitsBuf[lid] = maxVal_max_logitsBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_8 = (lid < 8u);
  if (shouldReduce_max_logitsBuf_8) {
    uint neighborIdx_max_logitsBuf_8 = (lid + 8u);
    float myVal_max_logitsBuf_8 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_8 = logitsBuf[neighborIdx_max_logitsBuf_8];
    bool isGreater_max_logitsBuf_8 = (neighborVal_max_logitsBuf_8 > myVal_max_logitsBuf_8);
    float maxVal_max_logitsBuf_8 = ((isGreater_max_logitsBuf_8) ? (neighborVal_max_logitsBuf_8) : (myVal_max_logitsBuf_8));
    logitsBuf[lid] = maxVal_max_logitsBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_4 = (lid < 4u);
  if (shouldReduce_max_logitsBuf_4) {
    uint neighborIdx_max_logitsBuf_4 = (lid + 4u);
    float myVal_max_logitsBuf_4 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_4 = logitsBuf[neighborIdx_max_logitsBuf_4];
    bool isGreater_max_logitsBuf_4 = (neighborVal_max_logitsBuf_4 > myVal_max_logitsBuf_4);
    float maxVal_max_logitsBuf_4 = ((isGreater_max_logitsBuf_4) ? (neighborVal_max_logitsBuf_4) : (myVal_max_logitsBuf_4));
    logitsBuf[lid] = maxVal_max_logitsBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_2 = (lid < 2u);
  if (shouldReduce_max_logitsBuf_2) {
    uint neighborIdx_max_logitsBuf_2 = (lid + 2u);
    float myVal_max_logitsBuf_2 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_2 = logitsBuf[neighborIdx_max_logitsBuf_2];
    bool isGreater_max_logitsBuf_2 = (neighborVal_max_logitsBuf_2 > myVal_max_logitsBuf_2);
    float maxVal_max_logitsBuf_2 = ((isGreater_max_logitsBuf_2) ? (neighborVal_max_logitsBuf_2) : (myVal_max_logitsBuf_2));
    logitsBuf[lid] = maxVal_max_logitsBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_1 = (lid < 1u);
  if (shouldReduce_max_logitsBuf_1) {
    uint neighborIdx_max_logitsBuf_1 = (lid + 1u);
    float myVal_max_logitsBuf_1 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_1 = logitsBuf[neighborIdx_max_logitsBuf_1];
    bool isGreater_max_logitsBuf_1 = (neighborVal_max_logitsBuf_1 > myVal_max_logitsBuf_1);
    float maxVal_max_logitsBuf_1 = ((isGreater_max_logitsBuf_1) ? (neighborVal_max_logitsBuf_1) : (myVal_max_logitsBuf_1));
    logitsBuf[lid] = maxVal_max_logitsBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxLogit = logitsBuf[0u];
  float originalLogit = ((valid) ? (routerLogits[globalIdx]) : (as_type<float>(0xf149f2cau)));
  float shiftedLogit = (originalLogit - maxLogit);
  float expLogit = exp(shiftedLogit);
  float expMasked = ((expertValid) ? (expLogit) : (as_type<float>(0x00000000u)));
  expBuf[lid] = expMasked;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_128 = (lid < 128u);
  if (shouldReduce_sum_expBuf_128) {
    uint neighborIdx_sum_expBuf_128 = (lid + 128u);
    float myVal_sum_expBuf_128 = expBuf[lid];
    float neighborVal_sum_expBuf_128 = expBuf[neighborIdx_sum_expBuf_128];
    float sumVal_sum_expBuf_128 = (myVal_sum_expBuf_128 + neighborVal_sum_expBuf_128);
    expBuf[lid] = sumVal_sum_expBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_64 = (lid < 64u);
  if (shouldReduce_sum_expBuf_64) {
    uint neighborIdx_sum_expBuf_64 = (lid + 64u);
    float myVal_sum_expBuf_64 = expBuf[lid];
    float neighborVal_sum_expBuf_64 = expBuf[neighborIdx_sum_expBuf_64];
    float sumVal_sum_expBuf_64 = (myVal_sum_expBuf_64 + neighborVal_sum_expBuf_64);
    expBuf[lid] = sumVal_sum_expBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_32 = (lid < 32u);
  if (shouldReduce_sum_expBuf_32) {
    uint neighborIdx_sum_expBuf_32 = (lid + 32u);
    float myVal_sum_expBuf_32 = expBuf[lid];
    float neighborVal_sum_expBuf_32 = expBuf[neighborIdx_sum_expBuf_32];
    float sumVal_sum_expBuf_32 = (myVal_sum_expBuf_32 + neighborVal_sum_expBuf_32);
    expBuf[lid] = sumVal_sum_expBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_16 = (lid < 16u);
  if (shouldReduce_sum_expBuf_16) {
    uint neighborIdx_sum_expBuf_16 = (lid + 16u);
    float myVal_sum_expBuf_16 = expBuf[lid];
    float neighborVal_sum_expBuf_16 = expBuf[neighborIdx_sum_expBuf_16];
    float sumVal_sum_expBuf_16 = (myVal_sum_expBuf_16 + neighborVal_sum_expBuf_16);
    expBuf[lid] = sumVal_sum_expBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_8 = (lid < 8u);
  if (shouldReduce_sum_expBuf_8) {
    uint neighborIdx_sum_expBuf_8 = (lid + 8u);
    float myVal_sum_expBuf_8 = expBuf[lid];
    float neighborVal_sum_expBuf_8 = expBuf[neighborIdx_sum_expBuf_8];
    float sumVal_sum_expBuf_8 = (myVal_sum_expBuf_8 + neighborVal_sum_expBuf_8);
    expBuf[lid] = sumVal_sum_expBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_4 = (lid < 4u);
  if (shouldReduce_sum_expBuf_4) {
    uint neighborIdx_sum_expBuf_4 = (lid + 4u);
    float myVal_sum_expBuf_4 = expBuf[lid];
    float neighborVal_sum_expBuf_4 = expBuf[neighborIdx_sum_expBuf_4];
    float sumVal_sum_expBuf_4 = (myVal_sum_expBuf_4 + neighborVal_sum_expBuf_4);
    expBuf[lid] = sumVal_sum_expBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_2 = (lid < 2u);
  if (shouldReduce_sum_expBuf_2) {
    uint neighborIdx_sum_expBuf_2 = (lid + 2u);
    float myVal_sum_expBuf_2 = expBuf[lid];
    float neighborVal_sum_expBuf_2 = expBuf[neighborIdx_sum_expBuf_2];
    float sumVal_sum_expBuf_2 = (myVal_sum_expBuf_2 + neighborVal_sum_expBuf_2);
    expBuf[lid] = sumVal_sum_expBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_1 = (lid < 1u);
  if (shouldReduce_sum_expBuf_1) {
    uint neighborIdx_sum_expBuf_1 = (lid + 1u);
    float myVal_sum_expBuf_1 = expBuf[lid];
    float neighborVal_sum_expBuf_1 = expBuf[neighborIdx_sum_expBuf_1];
    float sumVal_sum_expBuf_1 = (myVal_sum_expBuf_1 + neighborVal_sum_expBuf_1);
    expBuf[lid] = sumVal_sum_expBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = expBuf[0u];
  float prob = (expLogit / sumExp);
  logitsBuf[lid] = prob;
  indexBuf[lid] = lidI32;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int kIter = 0; kIter < (int)(k); ++kIter) {
    if ((lid == 0u)) {
      int bestIdx = 0;
      float bestVal = as_type<float>(0xf149f2cau);
      for (int scanIdx = 0; scanIdx < (int)(ne); ++scanIdx) {
        float scanVal = logitsBuf[scanIdx];
        bool isBetter = (scanVal > bestVal);
        if (isBetter) {
          bestVal = scanVal;
          int scanIdxI32 = ((int)(scanIdx));
          bestIdx = scanIdxI32;
        }
      }
      uint kIterU32 = kIter;
      uint outIdx = ((tokenIdx * k) + kIterU32);
      int bestIdxFinal = bestIdx;
      float bestValFinal = bestVal;
      expertIndices[outIdx] = bestIdxFinal;
      expertWeights[outIdx] = bestValFinal;
      uint bestIdxU32 = ((uint)(bestIdxFinal));
      logitsBuf[bestIdxU32] = as_type<float>(0xf149f2cau);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

)kernel";
const char* k_metal_navatala_transformer_top_k_gating_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_top_k_gating_f16(device const half* routerLogits [[buffer(0)]], device const uint* numTokens [[buffer(1)]], device const uint* numExperts [[buffer(2)]], device const uint* topK [[buffer(3)]], device int* expertIndices [[buffer(4)]], device half* expertWeights [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint tokenIdx = ((uint)(int(__tgid.x)));
  uint nt = numTokens[0u];
  uint ne = numExperts[0u];
  uint k = topK[0u];
  threadgroup float logitsBuf[256];
  threadgroup float expBuf[256];
  threadgroup int indexBuf[256];
  bool tokenValid = (tokenIdx < nt);
  bool expertValid = (lid < ne);
  bool valid = (tokenValid && expertValid);
  uint globalIdx = ((tokenIdx * ne) + lid);
  half logitF16 = ((valid) ? (routerLogits[globalIdx]) : (half(0.000000)));
  float logit = ((expertValid) ? (((float)(logitF16))) : (as_type<float>(0xf149f2cau)));
  logitsBuf[lid] = logit;
  int lidI32 = ((int)(lid));
  indexBuf[lid] = lidI32;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_128 = (lid < 128u);
  if (shouldReduce_max_logitsBuf_128) {
    uint neighborIdx_max_logitsBuf_128 = (lid + 128u);
    float myVal_max_logitsBuf_128 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_128 = logitsBuf[neighborIdx_max_logitsBuf_128];
    bool isGreater_max_logitsBuf_128 = (neighborVal_max_logitsBuf_128 > myVal_max_logitsBuf_128);
    float maxVal_max_logitsBuf_128 = ((isGreater_max_logitsBuf_128) ? (neighborVal_max_logitsBuf_128) : (myVal_max_logitsBuf_128));
    logitsBuf[lid] = maxVal_max_logitsBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_64 = (lid < 64u);
  if (shouldReduce_max_logitsBuf_64) {
    uint neighborIdx_max_logitsBuf_64 = (lid + 64u);
    float myVal_max_logitsBuf_64 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_64 = logitsBuf[neighborIdx_max_logitsBuf_64];
    bool isGreater_max_logitsBuf_64 = (neighborVal_max_logitsBuf_64 > myVal_max_logitsBuf_64);
    float maxVal_max_logitsBuf_64 = ((isGreater_max_logitsBuf_64) ? (neighborVal_max_logitsBuf_64) : (myVal_max_logitsBuf_64));
    logitsBuf[lid] = maxVal_max_logitsBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_32 = (lid < 32u);
  if (shouldReduce_max_logitsBuf_32) {
    uint neighborIdx_max_logitsBuf_32 = (lid + 32u);
    float myVal_max_logitsBuf_32 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_32 = logitsBuf[neighborIdx_max_logitsBuf_32];
    bool isGreater_max_logitsBuf_32 = (neighborVal_max_logitsBuf_32 > myVal_max_logitsBuf_32);
    float maxVal_max_logitsBuf_32 = ((isGreater_max_logitsBuf_32) ? (neighborVal_max_logitsBuf_32) : (myVal_max_logitsBuf_32));
    logitsBuf[lid] = maxVal_max_logitsBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_16 = (lid < 16u);
  if (shouldReduce_max_logitsBuf_16) {
    uint neighborIdx_max_logitsBuf_16 = (lid + 16u);
    float myVal_max_logitsBuf_16 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_16 = logitsBuf[neighborIdx_max_logitsBuf_16];
    bool isGreater_max_logitsBuf_16 = (neighborVal_max_logitsBuf_16 > myVal_max_logitsBuf_16);
    float maxVal_max_logitsBuf_16 = ((isGreater_max_logitsBuf_16) ? (neighborVal_max_logitsBuf_16) : (myVal_max_logitsBuf_16));
    logitsBuf[lid] = maxVal_max_logitsBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_8 = (lid < 8u);
  if (shouldReduce_max_logitsBuf_8) {
    uint neighborIdx_max_logitsBuf_8 = (lid + 8u);
    float myVal_max_logitsBuf_8 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_8 = logitsBuf[neighborIdx_max_logitsBuf_8];
    bool isGreater_max_logitsBuf_8 = (neighborVal_max_logitsBuf_8 > myVal_max_logitsBuf_8);
    float maxVal_max_logitsBuf_8 = ((isGreater_max_logitsBuf_8) ? (neighborVal_max_logitsBuf_8) : (myVal_max_logitsBuf_8));
    logitsBuf[lid] = maxVal_max_logitsBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_4 = (lid < 4u);
  if (shouldReduce_max_logitsBuf_4) {
    uint neighborIdx_max_logitsBuf_4 = (lid + 4u);
    float myVal_max_logitsBuf_4 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_4 = logitsBuf[neighborIdx_max_logitsBuf_4];
    bool isGreater_max_logitsBuf_4 = (neighborVal_max_logitsBuf_4 > myVal_max_logitsBuf_4);
    float maxVal_max_logitsBuf_4 = ((isGreater_max_logitsBuf_4) ? (neighborVal_max_logitsBuf_4) : (myVal_max_logitsBuf_4));
    logitsBuf[lid] = maxVal_max_logitsBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_2 = (lid < 2u);
  if (shouldReduce_max_logitsBuf_2) {
    uint neighborIdx_max_logitsBuf_2 = (lid + 2u);
    float myVal_max_logitsBuf_2 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_2 = logitsBuf[neighborIdx_max_logitsBuf_2];
    bool isGreater_max_logitsBuf_2 = (neighborVal_max_logitsBuf_2 > myVal_max_logitsBuf_2);
    float maxVal_max_logitsBuf_2 = ((isGreater_max_logitsBuf_2) ? (neighborVal_max_logitsBuf_2) : (myVal_max_logitsBuf_2));
    logitsBuf[lid] = maxVal_max_logitsBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_1 = (lid < 1u);
  if (shouldReduce_max_logitsBuf_1) {
    uint neighborIdx_max_logitsBuf_1 = (lid + 1u);
    float myVal_max_logitsBuf_1 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_1 = logitsBuf[neighborIdx_max_logitsBuf_1];
    bool isGreater_max_logitsBuf_1 = (neighborVal_max_logitsBuf_1 > myVal_max_logitsBuf_1);
    float maxVal_max_logitsBuf_1 = ((isGreater_max_logitsBuf_1) ? (neighborVal_max_logitsBuf_1) : (myVal_max_logitsBuf_1));
    logitsBuf[lid] = maxVal_max_logitsBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxLogit = logitsBuf[0u];
  half originalLogitF16 = ((valid) ? (routerLogits[globalIdx]) : (half(0.000000)));
  float originalLogit = ((expertValid) ? (((float)(originalLogitF16))) : (as_type<float>(0xf149f2cau)));
  float shiftedLogit = (originalLogit - maxLogit);
  float expLogit = exp(shiftedLogit);
  float expMasked = ((expertValid) ? (expLogit) : (as_type<float>(0x00000000u)));
  expBuf[lid] = expMasked;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_128 = (lid < 128u);
  if (shouldReduce_sum_expBuf_128) {
    uint neighborIdx_sum_expBuf_128 = (lid + 128u);
    float myVal_sum_expBuf_128 = expBuf[lid];
    float neighborVal_sum_expBuf_128 = expBuf[neighborIdx_sum_expBuf_128];
    float sumVal_sum_expBuf_128 = (myVal_sum_expBuf_128 + neighborVal_sum_expBuf_128);
    expBuf[lid] = sumVal_sum_expBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_64 = (lid < 64u);
  if (shouldReduce_sum_expBuf_64) {
    uint neighborIdx_sum_expBuf_64 = (lid + 64u);
    float myVal_sum_expBuf_64 = expBuf[lid];
    float neighborVal_sum_expBuf_64 = expBuf[neighborIdx_sum_expBuf_64];
    float sumVal_sum_expBuf_64 = (myVal_sum_expBuf_64 + neighborVal_sum_expBuf_64);
    expBuf[lid] = sumVal_sum_expBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_32 = (lid < 32u);
  if (shouldReduce_sum_expBuf_32) {
    uint neighborIdx_sum_expBuf_32 = (lid + 32u);
    float myVal_sum_expBuf_32 = expBuf[lid];
    float neighborVal_sum_expBuf_32 = expBuf[neighborIdx_sum_expBuf_32];
    float sumVal_sum_expBuf_32 = (myVal_sum_expBuf_32 + neighborVal_sum_expBuf_32);
    expBuf[lid] = sumVal_sum_expBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_16 = (lid < 16u);
  if (shouldReduce_sum_expBuf_16) {
    uint neighborIdx_sum_expBuf_16 = (lid + 16u);
    float myVal_sum_expBuf_16 = expBuf[lid];
    float neighborVal_sum_expBuf_16 = expBuf[neighborIdx_sum_expBuf_16];
    float sumVal_sum_expBuf_16 = (myVal_sum_expBuf_16 + neighborVal_sum_expBuf_16);
    expBuf[lid] = sumVal_sum_expBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_8 = (lid < 8u);
  if (shouldReduce_sum_expBuf_8) {
    uint neighborIdx_sum_expBuf_8 = (lid + 8u);
    float myVal_sum_expBuf_8 = expBuf[lid];
    float neighborVal_sum_expBuf_8 = expBuf[neighborIdx_sum_expBuf_8];
    float sumVal_sum_expBuf_8 = (myVal_sum_expBuf_8 + neighborVal_sum_expBuf_8);
    expBuf[lid] = sumVal_sum_expBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_4 = (lid < 4u);
  if (shouldReduce_sum_expBuf_4) {
    uint neighborIdx_sum_expBuf_4 = (lid + 4u);
    float myVal_sum_expBuf_4 = expBuf[lid];
    float neighborVal_sum_expBuf_4 = expBuf[neighborIdx_sum_expBuf_4];
    float sumVal_sum_expBuf_4 = (myVal_sum_expBuf_4 + neighborVal_sum_expBuf_4);
    expBuf[lid] = sumVal_sum_expBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_2 = (lid < 2u);
  if (shouldReduce_sum_expBuf_2) {
    uint neighborIdx_sum_expBuf_2 = (lid + 2u);
    float myVal_sum_expBuf_2 = expBuf[lid];
    float neighborVal_sum_expBuf_2 = expBuf[neighborIdx_sum_expBuf_2];
    float sumVal_sum_expBuf_2 = (myVal_sum_expBuf_2 + neighborVal_sum_expBuf_2);
    expBuf[lid] = sumVal_sum_expBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_1 = (lid < 1u);
  if (shouldReduce_sum_expBuf_1) {
    uint neighborIdx_sum_expBuf_1 = (lid + 1u);
    float myVal_sum_expBuf_1 = expBuf[lid];
    float neighborVal_sum_expBuf_1 = expBuf[neighborIdx_sum_expBuf_1];
    float sumVal_sum_expBuf_1 = (myVal_sum_expBuf_1 + neighborVal_sum_expBuf_1);
    expBuf[lid] = sumVal_sum_expBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = expBuf[0u];
  float prob = (expLogit / sumExp);
  logitsBuf[lid] = prob;
  int lidI32_2 = ((int)(lid));
  indexBuf[lid] = lidI32_2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int kIter = 0; kIter < (int)(k); ++kIter) {
    if ((lid == 0u)) {
      int bestIdx = 0;
      float bestVal = as_type<float>(0xf149f2cau);
      for (int scanIdx = 0; scanIdx < (int)(ne); ++scanIdx) {
        float scanVal = logitsBuf[scanIdx];
        bool isBetter = (scanVal > bestVal);
        if (isBetter) {
          bestVal = scanVal;
          int scanIdxI32 = ((int)(scanIdx));
          bestIdx = scanIdxI32;
        }
      }
      uint kIterU32 = kIter;
      uint outIdx = ((tokenIdx * k) + kIterU32);
      int bestIdxFinal = bestIdx;
      float bestValFinal = bestVal;
      half bestValF16 = ((half)(bestValFinal));
      expertIndices[outIdx] = bestIdxFinal;
      expertWeights[outIdx] = bestValF16;
      uint bestIdxU32 = ((uint)(bestIdxFinal));
      logitsBuf[bestIdxU32] = as_type<float>(0xf149f2cau);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

)kernel";
const char* k_metal_navatala_transformer_expert_capacity_mask_f32 = R"kernel(
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

kernel void navatala_transformer_expert_capacity_mask_f32(device const int* expertIndices [[buffer(0)]], device const uint* numTokens [[buffer(1)]], device const uint* numExperts [[buffer(2)]], device const uint* topK [[buffer(3)]], device const uint* capacity [[buffer(4)]], device float* mask [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nt = numTokens[0u];
  uint ne = numExperts[0u];
  uint k = topK[0u];
  uint cap = capacity[0u];
  threadgroup uint expertCounts[256];
  uint lid = ((uint)(int(__tid.x)));
  if ((lid < ne)) {
    expertCounts[lid] = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint totalElements = (nt * k);
  bool valid = (gid < totalElements);
  if (valid) {
    int expertIdx = expertIndices[gid];
    uint expertIdxU32 = ((uint)(expertIdx));
    bool expertIdxValid = (expertIdxU32 < ne);
    if (expertIdxValid) {
      uint prevCount = atomic_fetch_add_explicit(((device atomic_uint*)(&(expertCounts[expertIdxU32]))), 1u, memory_order_relaxed);
      bool withinCapacity = (prevCount < cap);
      float maskVal = ((withinCapacity) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
      mask[gid] = maskVal;
    } else {
      mask[gid] = as_type<float>(0x00000000u);
    }
  }
}

)kernel";
const char* k_metal_navatala_transformer_expert_capacity_mask_f16 = R"kernel(
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

kernel void navatala_transformer_expert_capacity_mask_f16(device const int* expertIndices [[buffer(0)]], device const uint* numTokens [[buffer(1)]], device const uint* numExperts [[buffer(2)]], device const uint* topK [[buffer(3)]], device const uint* capacity [[buffer(4)]], device half* mask [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nt = numTokens[0u];
  uint ne = numExperts[0u];
  uint k = topK[0u];
  uint cap = capacity[0u];
  threadgroup uint expertCounts[256];
  uint lid = ((uint)(int(__tid.x)));
  if ((lid < ne)) {
    expertCounts[lid] = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint totalElements = (nt * k);
  bool valid = (gid < totalElements);
  if (valid) {
    int expertIdx = expertIndices[gid];
    uint expertIdxU32 = ((uint)(expertIdx));
    bool expertIdxValid = (expertIdxU32 < ne);
    if (expertIdxValid) {
      uint prevCount = atomic_fetch_add_explicit(((device atomic_uint*)(&(expertCounts[expertIdxU32]))), 1u, memory_order_relaxed);
      bool withinCapacity = (prevCount < cap);
      half maskVal = ((withinCapacity) ? (half(1.000000)) : (half(0.000000)));
      mask[gid] = maskVal;
    } else {
      mask[gid] = half(0.000000);
    }
  }
}

)kernel";
const char* k_metal_navatala_transformer_permute_tokens_f32 = R"kernel(
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

kernel void navatala_transformer_permute_tokens_f32(device const float* hiddenStates [[buffer(0)]], device const int* expertIndices [[buffer(1)]], device const uint* expertOffsets [[buffer(2)]], device atomic_uint* expertCounters [[buffer(3)]], device const uint* numTokens [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device const uint* numExperts [[buffer(6)]], device float* permutedStates [[buffer(7)]], device int* tokenIndices [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint tokenIdx = ((uint)(int(__tgid.x)));
  uint nt = numTokens[0u];
  uint hs = hiddenSize[0u];
  uint ne = numExperts[0u];
  threadgroup uint sharedDestTokenIdx[1];
  bool tokenValid = (tokenIdx < nt);
  if (tokenValid) {
    int expertIdx = expertIndices[tokenIdx];
    uint expertIdxU32 = ((uint)(expertIdx));
    bool expertValid = (expertIdxU32 < ne);
    if (expertValid) {
      if ((lid == 0u)) {
        uint localPosition = atomic_fetch_add_explicit(((device atomic_uint*)(&(expertCounters[expertIdxU32]))), 1u, memory_order_relaxed);
        uint expertOffset = expertOffsets[expertIdxU32];
        uint destTokenIdx = (expertOffset + localPosition);
        sharedDestTokenIdx[0u] = destTokenIdx;
        int tokenIdxI32 = ((int)(tokenIdx));
        tokenIndices[destTokenIdx] = tokenIdxI32;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint destTokenIdx = sharedDestTokenIdx[0u];
      for (int hIter = 0; hIter < (int)(hs); ++hIter) {
        uint hIdx = (lid + (256u * hIter));
        if ((hIdx < hs)) {
          uint srcIdx = ((tokenIdx * hs) + hIdx);
          uint dstIdx = ((destTokenIdx * hs) + hIdx);
          float val = hiddenStates[srcIdx];
          permutedStates[dstIdx] = val;
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_transformer_permute_tokens_f16 = R"kernel(
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

kernel void navatala_transformer_permute_tokens_f16(device const half* hiddenStates [[buffer(0)]], device const int* expertIndices [[buffer(1)]], device const uint* expertOffsets [[buffer(2)]], device atomic_uint* expertCounters [[buffer(3)]], device const uint* numTokens [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device const uint* numExperts [[buffer(6)]], device half* permutedStates [[buffer(7)]], device int* tokenIndices [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint tokenIdx = ((uint)(int(__tgid.x)));
  uint nt = numTokens[0u];
  uint hs = hiddenSize[0u];
  uint ne = numExperts[0u];
  threadgroup uint sharedDestTokenIdx[1];
  bool tokenValid = (tokenIdx < nt);
  if (tokenValid) {
    int expertIdx = expertIndices[tokenIdx];
    uint expertIdxU32 = ((uint)(expertIdx));
    bool expertValid = (expertIdxU32 < ne);
    if (expertValid) {
      if ((lid == 0u)) {
        uint localPosition = atomic_fetch_add_explicit(((device atomic_uint*)(&(expertCounters[expertIdxU32]))), 1u, memory_order_relaxed);
        uint expertOffset = expertOffsets[expertIdxU32];
        uint destTokenIdx = (expertOffset + localPosition);
        sharedDestTokenIdx[0u] = destTokenIdx;
        int tokenIdxI32 = ((int)(tokenIdx));
        tokenIndices[destTokenIdx] = tokenIdxI32;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint destTokenIdx = sharedDestTokenIdx[0u];
      for (int hIter = 0; hIter < (int)(hs); ++hIter) {
        uint hIdx = (lid + (256u * hIter));
        if ((hIdx < hs)) {
          uint srcIdx = ((tokenIdx * hs) + hIdx);
          uint dstIdx = ((destTokenIdx * hs) + hIdx);
          half val = hiddenStates[srcIdx];
          permutedStates[dstIdx] = val;
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_transformer_unpermute_tokens_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_unpermute_tokens_f32(device const float* permutedStates [[buffer(0)]], device const int* tokenIndices [[buffer(1)]], device const uint* numTokens [[buffer(2)]], device const uint* hiddenSize [[buffer(3)]], device float* hiddenStates [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint permutedIdx = ((uint)(int(__tgid.x)));
  uint nt = numTokens[0u];
  uint hs = hiddenSize[0u];
  bool valid = (permutedIdx < nt);
  if (valid) {
    int origIdxI32 = tokenIndices[permutedIdx];
    uint origIdx = ((uint)(origIdxI32));
    bool origValid = (origIdx < nt);
    if (origValid) {
      for (int hIter = 0; hIter < (int)(hs); ++hIter) {
        uint hIdx = (lid + (256u * hIter));
        if ((hIdx < hs)) {
          uint srcIdx = ((permutedIdx * hs) + hIdx);
          uint dstIdx = ((origIdx * hs) + hIdx);
          float val = permutedStates[srcIdx];
          hiddenStates[dstIdx] = val;
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_transformer_unpermute_tokens_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_unpermute_tokens_f16(device const half* permutedStates [[buffer(0)]], device const int* tokenIndices [[buffer(1)]], device const uint* numTokens [[buffer(2)]], device const uint* hiddenSize [[buffer(3)]], device half* hiddenStates [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint permutedIdx = ((uint)(int(__tgid.x)));
  uint nt = numTokens[0u];
  uint hs = hiddenSize[0u];
  bool valid = (permutedIdx < nt);
  if (valid) {
    int origIdxI32 = tokenIndices[permutedIdx];
    uint origIdx = ((uint)(origIdxI32));
    bool origValid = (origIdx < nt);
    if (origValid) {
      for (int hIter = 0; hIter < (int)(hs); ++hIter) {
        uint hIdx = (lid + (256u * hIter));
        if ((hIdx < hs)) {
          uint srcIdx = ((permutedIdx * hs) + hIdx);
          uint dstIdx = ((origIdx * hs) + hIdx);
          half val = permutedStates[srcIdx];
          hiddenStates[dstIdx] = val;
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_transformer_pad_sequence_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_pad_sequence_f32(device const float* _input [[buffer(0)]], device const uint* seqLengths [[buffer(1)]], device const uint* seqOffsets [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* maxSeqLen [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device float* _output [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint msl = maxSeqLen[0u];
  uint hs = hiddenSize[0u];
  uint totalOut = (bs * (msl * hs));
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint seqHidden = (msl * hs);
    uint batchIdx = (gid / seqHidden);
    uint remainder = (gid % seqHidden);
    uint seqIdx = (remainder / hs);
    uint hiddenIdx = (remainder % hs);
    uint seqLen = seqLengths[batchIdx];
    bool isValid = (seqIdx < seqLen);
    if (isValid) {
      uint seqOffset = seqOffsets[batchIdx];
      uint inputIdx = ((seqOffset * hs) + ((seqIdx * hs) + hiddenIdx));
      float val = _input[inputIdx];
      _output[gid] = val;
    } else {
      _output[gid] = as_type<float>(0x00000000u);
    }
  }
}

)kernel";
const char* k_metal_navatala_transformer_pad_sequence_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_pad_sequence_f16(device const half* _input [[buffer(0)]], device const uint* seqLengths [[buffer(1)]], device const uint* seqOffsets [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* maxSeqLen [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device half* _output [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint msl = maxSeqLen[0u];
  uint hs = hiddenSize[0u];
  uint totalOut = (bs * (msl * hs));
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint seqHidden = (msl * hs);
    uint batchIdx = (gid / seqHidden);
    uint remainder = (gid % seqHidden);
    uint seqIdx = (remainder / hs);
    uint hiddenIdx = (remainder % hs);
    uint seqLen = seqLengths[batchIdx];
    bool isValid = (seqIdx < seqLen);
    if (isValid) {
      uint seqOffset = seqOffsets[batchIdx];
      uint inputIdx = ((seqOffset * hs) + ((seqIdx * hs) + hiddenIdx));
      half val = _input[inputIdx];
      _output[gid] = val;
    } else {
      _output[gid] = half(0.000000);
    }
  }
}

)kernel";
const char* k_metal_navatala_transformer_unpad_sequence_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_unpad_sequence_f32(device const float* _input [[buffer(0)]], device const uint* seqLengths [[buffer(1)]], device const uint* seqOffsets [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* maxSeqLen [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device const uint* totalTokens [[buffer(6)]], device float* _output [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint msl = maxSeqLen[0u];
  uint hs = hiddenSize[0u];
  uint tt = totalTokens[0u];
  uint totalOut = (tt * hs);
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint tokenIdx = (gid / hs);
    uint hiddenIdx = (gid % hs);
    uint foundBatchIdx = 0u;
    for (int b = 0; b < (int)(bs); ++b) {
      uint currOffset = seqOffsets[b];
      uint nextB = (b + 1u);
      uint nextOffset = (((nextB < bs)) ? (seqOffsets[nextB]) : (tt));
      bool inRange = ((tokenIdx >= currOffset) && (tokenIdx < nextOffset));
      if (inRange) {
        foundBatchIdx = b;
      }
    }
    uint batchIdx = foundBatchIdx;
    uint batchOffset = seqOffsets[batchIdx];
    uint localTokenIdx = (tokenIdx - batchOffset);
    uint batchStride = (msl * hs);
    uint inputIdx = ((batchIdx * batchStride) + ((localTokenIdx * hs) + hiddenIdx));
    float val = _input[inputIdx];
    _output[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_transformer_unpad_sequence_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_unpad_sequence_f16(device const half* _input [[buffer(0)]], device const uint* seqLengths [[buffer(1)]], device const uint* seqOffsets [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* maxSeqLen [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device const uint* totalTokens [[buffer(6)]], device half* _output [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint msl = maxSeqLen[0u];
  uint hs = hiddenSize[0u];
  uint tt = totalTokens[0u];
  uint totalOut = (tt * hs);
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint tokenIdx = (gid / hs);
    uint hiddenIdx = (gid % hs);
    uint foundBatchIdx = 0u;
    for (int b = 0; b < (int)(bs); ++b) {
      uint currOffset = seqOffsets[b];
      uint nextB = (b + 1u);
      uint nextOffset = (((nextB < bs)) ? (seqOffsets[nextB]) : (tt));
      bool inRange = ((tokenIdx >= currOffset) && (tokenIdx < nextOffset));
      if (inRange) {
        foundBatchIdx = b;
      }
    }
    uint batchIdx = foundBatchIdx;
    uint batchOffset = seqOffsets[batchIdx];
    uint localTokenIdx = (tokenIdx - batchOffset);
    uint batchStride = (msl * hs);
    uint inputIdx = ((batchIdx * batchStride) + ((localTokenIdx * hs) + hiddenIdx));
    half val = _input[inputIdx];
    _output[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_transformer_swizzle_layout_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_swizzle_layout_f32(device const float* _input [[buffer(0)]], device const uint* rows [[buffer(1)]], device const uint* cols [[buffer(2)]], device const uint* tileSize [[buffer(3)]], device float* _output [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint r = rows[0u];
  uint c = cols[0u];
  uint ts = tileSize[0u];
  uint total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / c);
    uint col = (gid % c);
    float val = _input[gid];
    uint tileRow = (row / ts);
    uint tileCol = (col / ts);
    uint localRow = (row % ts);
    uint localCol = (col % ts);
    uint tilesPerRow = ((c + (ts - 1u)) / ts);
    uint tileIdx = ((tileRow * tilesPerRow) + tileCol);
    uint tileElements = (ts * ts);
    uint localIdx = ((localRow * ts) + localCol);
    uint outIdx = ((tileIdx * tileElements) + localIdx);
    _output[outIdx] = val;
  }
}

)kernel";
const char* k_metal_navatala_transformer_swizzle_layout_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_swizzle_layout_f16(device const half* _input [[buffer(0)]], device const uint* rows [[buffer(1)]], device const uint* cols [[buffer(2)]], device const uint* tileSize [[buffer(3)]], device half* _output [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint r = rows[0u];
  uint c = cols[0u];
  uint ts = tileSize[0u];
  uint total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / c);
    uint col = (gid % c);
    half val = _input[gid];
    uint tileRow = (row / ts);
    uint tileCol = (col / ts);
    uint localRow = (row % ts);
    uint localCol = (col % ts);
    uint tilesPerRow = ((c + (ts - 1u)) / ts);
    uint tileIdx = ((tileRow * tilesPerRow) + tileCol);
    uint tileElements = (ts * ts);
    uint localIdx = ((localRow * ts) + localCol);
    uint outIdx = ((tileIdx * tileElements) + localIdx);
    _output[outIdx] = val;
  }
}

)kernel";
const char* k_metal_navatala_transformer_hadamard_transform_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_hadamard_transform_f32(device const float* _input [[buffer(0)]], device const uint* batchSize [[buffer(1)]], device const uint* vectorSize [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint vs = vectorSize[0u];
  threadgroup float sharedBuf[256];
  bool batchValid = (batchIdx < bs);
  bool threadValid = (lid < vs);
  uint globalIdx = ((batchIdx * vs) + lid);
  if ((batchValid && threadValid)) {
    float val = _input[globalIdx];
    sharedBuf[lid] = val;
  } else {
    sharedBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride1 = 1u;
  uint partner1 = (lid ^ stride1);
  float myVal1 = sharedBuf[lid];
  float partnerVal1 = sharedBuf[partner1];
  bool lowBit1 = ((lid & stride1) == 0u);
  float newVal1 = ((lowBit1) ? ((myVal1 + partnerVal1)) : ((partnerVal1 - myVal1)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sharedBuf[lid] = newVal1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride2 = 2u;
  uint partner2 = (lid ^ stride2);
  float myVal2 = sharedBuf[lid];
  float partnerVal2 = sharedBuf[partner2];
  bool lowBit2 = ((lid & stride2) == 0u);
  float newVal2 = ((lowBit2) ? ((myVal2 + partnerVal2)) : ((partnerVal2 - myVal2)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sharedBuf[lid] = newVal2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride3 = 4u;
  uint partner3 = (lid ^ stride3);
  float myVal3 = sharedBuf[lid];
  float partnerVal3 = sharedBuf[partner3];
  bool lowBit3 = ((lid & stride3) == 0u);
  float newVal3 = ((lowBit3) ? ((myVal3 + partnerVal3)) : ((partnerVal3 - myVal3)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sharedBuf[lid] = newVal3;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride4 = 8u;
  uint partner4 = (lid ^ stride4);
  float myVal4 = sharedBuf[lid];
  float partnerVal4 = sharedBuf[partner4];
  bool lowBit4 = ((lid & stride4) == 0u);
  float newVal4 = ((lowBit4) ? ((myVal4 + partnerVal4)) : ((partnerVal4 - myVal4)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sharedBuf[lid] = newVal4;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride5 = 16u;
  uint partner5 = (lid ^ stride5);
  float myVal5 = sharedBuf[lid];
  float partnerVal5 = sharedBuf[partner5];
  bool lowBit5 = ((lid & stride5) == 0u);
  float newVal5 = ((lowBit5) ? ((myVal5 + partnerVal5)) : ((partnerVal5 - myVal5)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sharedBuf[lid] = newVal5;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride6 = 32u;
  uint partner6 = (lid ^ stride6);
  float myVal6 = sharedBuf[lid];
  float partnerVal6 = sharedBuf[partner6];
  bool lowBit6 = ((lid & stride6) == 0u);
  float newVal6 = ((lowBit6) ? ((myVal6 + partnerVal6)) : ((partnerVal6 - myVal6)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sharedBuf[lid] = newVal6;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride7 = 64u;
  uint partner7 = (lid ^ stride7);
  float myVal7 = sharedBuf[lid];
  float partnerVal7 = sharedBuf[partner7];
  bool lowBit7 = ((lid & stride7) == 0u);
  float newVal7 = ((lowBit7) ? ((myVal7 + partnerVal7)) : ((partnerVal7 - myVal7)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sharedBuf[lid] = newVal7;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride8 = 128u;
  uint partner8 = (lid ^ stride8);
  float myVal8 = sharedBuf[lid];
  float partnerVal8 = sharedBuf[partner8];
  bool lowBit8 = ((lid & stride8) == 0u);
  float newVal8 = ((lowBit8) ? ((myVal8 + partnerVal8)) : ((partnerVal8 - myVal8)));
  float vsF32 = ((float)(vs));
  float scale = sqrt(vsF32);
  float normalized = (newVal8 / scale);
  if ((batchValid && threadValid)) {
    _output[globalIdx] = normalized;
  }
}

)kernel";
const char* k_metal_navatala_transformer_hadamard_transform_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_hadamard_transform_f16(device const half* _input [[buffer(0)]], device const uint* batchSize [[buffer(1)]], device const uint* vectorSize [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint vs = vectorSize[0u];
  threadgroup float sharedBuf[256];
  bool batchValid = (batchIdx < bs);
  bool threadValid = (lid < vs);
  uint globalIdx = ((batchIdx * vs) + lid);
  if ((batchValid && threadValid)) {
    half valF16 = _input[globalIdx];
    float val = ((float)(valF16));
    sharedBuf[lid] = val;
  } else {
    sharedBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride1 = 1u;
  uint partner1 = (lid ^ stride1);
  float myVal1 = sharedBuf[lid];
  float partnerVal1 = sharedBuf[partner1];
  bool lowBit1 = ((lid & stride1) == 0u);
  float newVal1 = ((lowBit1) ? ((myVal1 + partnerVal1)) : ((partnerVal1 - myVal1)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sharedBuf[lid] = newVal1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride2 = 2u;
  uint partner2 = (lid ^ stride2);
  float myVal2 = sharedBuf[lid];
  float partnerVal2 = sharedBuf[partner2];
  bool lowBit2 = ((lid & stride2) == 0u);
  float newVal2 = ((lowBit2) ? ((myVal2 + partnerVal2)) : ((partnerVal2 - myVal2)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sharedBuf[lid] = newVal2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride3 = 4u;
  uint partner3 = (lid ^ stride3);
  float myVal3 = sharedBuf[lid];
  float partnerVal3 = sharedBuf[partner3];
  bool lowBit3 = ((lid & stride3) == 0u);
  float newVal3 = ((lowBit3) ? ((myVal3 + partnerVal3)) : ((partnerVal3 - myVal3)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sharedBuf[lid] = newVal3;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride4 = 8u;
  uint partner4 = (lid ^ stride4);
  float myVal4 = sharedBuf[lid];
  float partnerVal4 = sharedBuf[partner4];
  bool lowBit4 = ((lid & stride4) == 0u);
  float newVal4 = ((lowBit4) ? ((myVal4 + partnerVal4)) : ((partnerVal4 - myVal4)));
  float vsF32 = ((float)(vs));
  float scale = sqrt(vsF32);
  float normalized = (newVal4 / scale);
  half resultF16 = ((half)(normalized));
  if ((batchValid && threadValid)) {
    _output[globalIdx] = resultF16;
  }
}

)kernel";
const char* k_metal_navatala_transformer_permute_axes_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_permute_axes_f32(device const float* _input [[buffer(0)]], device const uint* dim0 [[buffer(1)]], device const uint* dim1 [[buffer(2)]], device const uint* dim2 [[buffer(3)]], device const uint* dim3 [[buffer(4)]], device const uint* perm0 [[buffer(5)]], device const uint* perm1 [[buffer(6)]], device const uint* perm2 [[buffer(7)]], device const uint* perm3 [[buffer(8)]], device float* _output [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint d0 = dim0[0u];
  uint d1 = dim1[0u];
  uint d2 = dim2[0u];
  uint d3 = dim3[0u];
  uint p0 = perm0[0u];
  uint p1 = perm1[0u];
  uint p2 = perm2[0u];
  uint p3 = perm3[0u];
  uint total = (d0 * (d1 * (d2 * d3)));
  bool inBounds = (gid < total);
  if (inBounds) {
    uint stride0 = (d1 * (d2 * d3));
    uint stride1 = (d2 * d3);
    uint stride2 = d3;
    uint stride3 = 1u;
    uint i0 = (gid / stride0);
    uint rem0 = (gid % stride0);
    uint i1 = (rem0 / stride1);
    uint rem1 = (rem0 % stride1);
    uint i2 = (rem1 / stride2);
    uint i3 = (rem1 % stride2);
    float val = _input[gid];
    uint od0 = (((p0 == 0u)) ? (d0) : ((((p0 == 1u)) ? (d1) : ((((p0 == 2u)) ? (d2) : (d3))))));
    uint od1 = (((p1 == 0u)) ? (d0) : ((((p1 == 1u)) ? (d1) : ((((p1 == 2u)) ? (d2) : (d3))))));
    uint od2 = (((p2 == 0u)) ? (d0) : ((((p2 == 1u)) ? (d1) : ((((p2 == 2u)) ? (d2) : (d3))))));
    uint od3 = (((p3 == 0u)) ? (d0) : ((((p3 == 1u)) ? (d1) : ((((p3 == 2u)) ? (d2) : (d3))))));
    uint oi0 = (((p0 == 0u)) ? (i0) : ((((p0 == 1u)) ? (i1) : ((((p0 == 2u)) ? (i2) : (i3))))));
    uint oi1 = (((p1 == 0u)) ? (i0) : ((((p1 == 1u)) ? (i1) : ((((p1 == 2u)) ? (i2) : (i3))))));
    uint oi2 = (((p2 == 0u)) ? (i0) : ((((p2 == 1u)) ? (i1) : ((((p2 == 2u)) ? (i2) : (i3))))));
    uint oi3 = (((p3 == 0u)) ? (i0) : ((((p3 == 1u)) ? (i1) : ((((p3 == 2u)) ? (i2) : (i3))))));
    uint ostride0 = (od1 * (od2 * od3));
    uint ostride1 = (od2 * od3);
    uint ostride2 = od3;
    uint outIdx = ((oi0 * ostride0) + ((oi1 * ostride1) + ((oi2 * ostride2) + oi3)));
    _output[outIdx] = val;
  }
}

)kernel";
const char* k_metal_navatala_transformer_permute_axes_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_permute_axes_f16(device const half* _input [[buffer(0)]], device const uint* dim0 [[buffer(1)]], device const uint* dim1 [[buffer(2)]], device const uint* dim2 [[buffer(3)]], device const uint* dim3 [[buffer(4)]], device const uint* perm0 [[buffer(5)]], device const uint* perm1 [[buffer(6)]], device const uint* perm2 [[buffer(7)]], device const uint* perm3 [[buffer(8)]], device half* _output [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint d0 = dim0[0u];
  uint d1 = dim1[0u];
  uint d2 = dim2[0u];
  uint d3 = dim3[0u];
  uint p0 = perm0[0u];
  uint p1 = perm1[0u];
  uint p2 = perm2[0u];
  uint p3 = perm3[0u];
  uint total = (d0 * (d1 * (d2 * d3)));
  bool inBounds = (gid < total);
  if (inBounds) {
    uint stride0 = (d1 * (d2 * d3));
    uint stride1 = (d2 * d3);
    uint stride2 = d3;
    uint i0 = (gid / stride0);
    uint rem0 = (gid % stride0);
    uint i1 = (rem0 / stride1);
    uint rem1 = (rem0 % stride1);
    uint i2 = (rem1 / stride2);
    uint i3 = (rem1 % stride2);
    half val = _input[gid];
    uint od0 = (((p0 == 0u)) ? (d0) : ((((p0 == 1u)) ? (d1) : ((((p0 == 2u)) ? (d2) : (d3))))));
    uint od1 = (((p1 == 0u)) ? (d0) : ((((p1 == 1u)) ? (d1) : ((((p1 == 2u)) ? (d2) : (d3))))));
    uint od2 = (((p2 == 0u)) ? (d0) : ((((p2 == 1u)) ? (d1) : ((((p2 == 2u)) ? (d2) : (d3))))));
    uint od3 = (((p3 == 0u)) ? (d0) : ((((p3 == 1u)) ? (d1) : ((((p3 == 2u)) ? (d2) : (d3))))));
    uint oi0 = (((p0 == 0u)) ? (i0) : ((((p0 == 1u)) ? (i1) : ((((p0 == 2u)) ? (i2) : (i3))))));
    uint oi1 = (((p1 == 0u)) ? (i0) : ((((p1 == 1u)) ? (i1) : ((((p1 == 2u)) ? (i2) : (i3))))));
    uint oi2 = (((p2 == 0u)) ? (i0) : ((((p2 == 1u)) ? (i1) : ((((p2 == 2u)) ? (i2) : (i3))))));
    uint oi3 = (((p3 == 0u)) ? (i0) : ((((p3 == 1u)) ? (i1) : ((((p3 == 2u)) ? (i2) : (i3))))));
    uint ostride0 = (od1 * (od2 * od3));
    uint ostride1 = (od2 * od3);
    uint ostride2 = od3;
    uint outIdx = ((oi0 * ostride0) + ((oi1 * ostride1) + ((oi2 * ostride2) + oi3)));
    _output[outIdx] = val;
  }
}

)kernel";
const char* k_metal_navatala_transformer_pack_sequences_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_pack_sequences_f32(device const float* _input [[buffer(0)]], device const uint* seqLengths [[buffer(1)]], device const uint* seqOffsets [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* maxSeqLen [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device const uint* totalTokens [[buffer(6)]], device float* _output [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint msl = maxSeqLen[0u];
  uint hs = hiddenSize[0u];
  uint tt = totalTokens[0u];
  uint totalOut = (tt * hs);
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint tokenIdx = (gid / hs);
    uint hiddenIdx = (gid % hs);
    uint foundBatchIdx = 0u;
    for (int b = 0; b < (int)(bs); ++b) {
      uint currOffset = seqOffsets[b];
      uint nextB = (b + 1u);
      uint nextOffset = (((nextB < bs)) ? (seqOffsets[nextB]) : (tt));
      bool inRange = ((tokenIdx >= currOffset) && (tokenIdx < nextOffset));
      if (inRange) {
        foundBatchIdx = b;
      }
    }
    uint batchIdx = foundBatchIdx;
    uint batchOffset = seqOffsets[batchIdx];
    uint localTokenIdx = (tokenIdx - batchOffset);
    uint batchStride = (msl * hs);
    uint inputIdx = ((batchIdx * batchStride) + ((localTokenIdx * hs) + hiddenIdx));
    float val = _input[inputIdx];
    _output[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_transformer_pack_sequences_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_pack_sequences_f16(device const half* _input [[buffer(0)]], device const uint* seqLengths [[buffer(1)]], device const uint* seqOffsets [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* maxSeqLen [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device const uint* totalTokens [[buffer(6)]], device half* _output [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint msl = maxSeqLen[0u];
  uint hs = hiddenSize[0u];
  uint tt = totalTokens[0u];
  uint totalOut = (tt * hs);
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint tokenIdx = (gid / hs);
    uint hiddenIdx = (gid % hs);
    uint foundBatchIdx = 0u;
    for (int b = 0; b < (int)(bs); ++b) {
      uint currOffset = seqOffsets[b];
      uint nextB = (b + 1u);
      uint nextOffset = (((nextB < bs)) ? (seqOffsets[nextB]) : (tt));
      bool inRange = ((tokenIdx >= currOffset) && (tokenIdx < nextOffset));
      if (inRange) {
        foundBatchIdx = b;
      }
    }
    uint batchIdx = foundBatchIdx;
    uint batchOffset = seqOffsets[batchIdx];
    uint localTokenIdx = (tokenIdx - batchOffset);
    uint batchStride = (msl * hs);
    uint inputIdx = ((batchIdx * batchStride) + ((localTokenIdx * hs) + hiddenIdx));
    half val = _input[inputIdx];
    _output[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_transformer_unpack_sequences_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_unpack_sequences_f32(device const float* _input [[buffer(0)]], device const uint* seqLengths [[buffer(1)]], device const uint* seqOffsets [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* maxSeqLen [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device float* _output [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint msl = maxSeqLen[0u];
  uint hs = hiddenSize[0u];
  uint totalOut = (bs * (msl * hs));
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint seqHidden = (msl * hs);
    uint batchIdx = (gid / seqHidden);
    uint remainder = (gid % seqHidden);
    uint seqIdx = (remainder / hs);
    uint hiddenIdx = (remainder % hs);
    uint seqLen = seqLengths[batchIdx];
    bool isValid = (seqIdx < seqLen);
    if (isValid) {
      uint seqOffset = seqOffsets[batchIdx];
      uint inputIdx = ((seqOffset * hs) + ((seqIdx * hs) + hiddenIdx));
      float val = _input[inputIdx];
      _output[gid] = val;
    } else {
      _output[gid] = as_type<float>(0x00000000u);
    }
  }
}

)kernel";
const char* k_metal_navatala_transformer_unpack_sequences_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_unpack_sequences_f16(device const half* _input [[buffer(0)]], device const uint* seqLengths [[buffer(1)]], device const uint* seqOffsets [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* maxSeqLen [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device half* _output [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint msl = maxSeqLen[0u];
  uint hs = hiddenSize[0u];
  uint totalOut = (bs * (msl * hs));
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint seqHidden = (msl * hs);
    uint batchIdx = (gid / seqHidden);
    uint remainder = (gid % seqHidden);
    uint seqIdx = (remainder / hs);
    uint hiddenIdx = (remainder % hs);
    uint seqLen = seqLengths[batchIdx];
    bool isValid = (seqIdx < seqLen);
    if (isValid) {
      uint seqOffset = seqOffsets[batchIdx];
      uint inputIdx = ((seqOffset * hs) + ((seqIdx * hs) + hiddenIdx));
      half val = _input[inputIdx];
      _output[gid] = val;
    } else {
      _output[gid] = half(0.000000);
    }
  }
}

)kernel";
const char* k_metal_navatala_transformer_generate_position_ids_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_generate_position_ids_f32(device const uint* batchSize [[buffer(0)]], device const uint* seqLen [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  uint total = (bs * sl);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint pos = (gid % sl);
    float posF32 = ((float)(pos));
    _output[gid] = posF32;
  }
}

)kernel";
const char* k_metal_navatala_transformer_generate_position_ids_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_generate_position_ids_u32(device const uint* batchSize [[buffer(0)]], device const uint* seqLen [[buffer(1)]], device uint* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  uint total = (bs * sl);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint pos = (gid % sl);
    _output[gid] = pos;
  }
}

)kernel";
const char* k_metal_navatala_transformer_generate_causal_mask_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_generate_causal_mask_f32(device const uint* seqLen [[buffer(0)]], device float* _output [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint sl = seqLen[0u];
  uint total = (sl * sl);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / sl);
    uint col = (gid % sl);
    bool canAttend = (col <= row);
    if (canAttend) {
      _output[gid] = as_type<float>(0x00000000u);
    } else {
      _output[gid] = as_type<float>(0xce6e6b28u);
    }
  }
}

)kernel";
const char* k_metal_navatala_transformer_generate_causal_mask_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_generate_causal_mask_f16(device const uint* seqLen [[buffer(0)]], device half* _output [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint sl = seqLen[0u];
  uint total = (sl * sl);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / sl);
    uint col = (gid % sl);
    bool canAttend = (col <= row);
    if (canAttend) {
      _output[gid] = half(0.000000);
    } else {
      _output[gid] = half(-65000.000000);
    }
  }
}

)kernel";
const char* k_metal_navatala_transformer_precompute_rotary_emb_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_precompute_rotary_emb_f32(device const uint* maxSeqLen [[buffer(0)]], device const uint* headDim [[buffer(1)]], device const float* base [[buffer(2)]], device float* cosOutput [[buffer(3)]], device float* sinOutput [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint msl = maxSeqLen[0u];
  uint hd = headDim[0u];
  float ropeBase = base[0u];
  uint halfHd = (hd / 2u);
  uint total = (msl * halfHd);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint pos = (gid / halfHd);
    uint dimPair = (gid % halfHd);
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (as_type<float>(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosVal = cos(theta);
    float sinVal = sin(theta);
    cosOutput[gid] = cosVal;
    sinOutput[gid] = sinVal;
  }
}

)kernel";
const char* k_metal_navatala_transformer_precompute_rotary_emb_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_precompute_rotary_emb_f16(device const uint* maxSeqLen [[buffer(0)]], device const uint* headDim [[buffer(1)]], device const float* base [[buffer(2)]], device half* cosOutput [[buffer(3)]], device half* sinOutput [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint msl = maxSeqLen[0u];
  uint hd = headDim[0u];
  float ropeBase = base[0u];
  uint halfHd = (hd / 2u);
  uint total = (msl * halfHd);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint pos = (gid / halfHd);
    uint dimPair = (gid % halfHd);
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (as_type<float>(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosValF32 = cos(theta);
    float sinValF32 = sin(theta);
    half cosVal = ((half)(cosValF32));
    half sinVal = ((half)(sinValF32));
    cosOutput[gid] = cosVal;
    sinOutput[gid] = sinVal;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_gelu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_gelu_f32 = {
  1,
  "navatala_transformer_gelu_f32",
  "metal",
  "navatala_transformer_gelu_f32",
  "kernel:metal:navatala_transformer_gelu_f32",
  "abi-r1:metal:navatala_transformer_gelu_f32",
  "abi-r1:metal:navatala_transformer_gelu_f32",
  3,
  kAbiArgs_metal_navatala_transformer_gelu_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_gelu_tanh_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_gelu_tanh_f32 = {
  1,
  "navatala_transformer_gelu_tanh_f32",
  "metal",
  "navatala_transformer_gelu_tanh_f32",
  "kernel:metal:navatala_transformer_gelu_tanh_f32",
  "abi-r1:metal:navatala_transformer_gelu_tanh_f32",
  "abi-r1:metal:navatala_transformer_gelu_tanh_f32",
  3,
  kAbiArgs_metal_navatala_transformer_gelu_tanh_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_silu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_silu_f32 = {
  1,
  "navatala_transformer_silu_f32",
  "metal",
  "navatala_transformer_silu_f32",
  "kernel:metal:navatala_transformer_silu_f32",
  "abi-r1:metal:navatala_transformer_silu_f32",
  "abi-r1:metal:navatala_transformer_silu_f32",
  3,
  kAbiArgs_metal_navatala_transformer_silu_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_relu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_relu_f32 = {
  1,
  "navatala_transformer_relu_f32",
  "metal",
  "navatala_transformer_relu_f32",
  "kernel:metal:navatala_transformer_relu_f32",
  "abi-r1:metal:navatala_transformer_relu_f32",
  "abi-r1:metal:navatala_transformer_relu_f32",
  3,
  kAbiArgs_metal_navatala_transformer_relu_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_quick_gelu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_quick_gelu_f32 = {
  1,
  "navatala_transformer_quick_gelu_f32",
  "metal",
  "navatala_transformer_quick_gelu_f32",
  "kernel:metal:navatala_transformer_quick_gelu_f32",
  "abi-r1:metal:navatala_transformer_quick_gelu_f32",
  "abi-r1:metal:navatala_transformer_quick_gelu_f32",
  3,
  kAbiArgs_metal_navatala_transformer_quick_gelu_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_squared_relu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_squared_relu_f32 = {
  1,
  "navatala_transformer_squared_relu_f32",
  "metal",
  "navatala_transformer_squared_relu_f32",
  "kernel:metal:navatala_transformer_squared_relu_f32",
  "abi-r1:metal:navatala_transformer_squared_relu_f32",
  "abi-r1:metal:navatala_transformer_squared_relu_f32",
  3,
  kAbiArgs_metal_navatala_transformer_squared_relu_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_bias_gelu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bias", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "biasSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_bias_gelu_f32 = {
  1,
  "navatala_transformer_bias_gelu_f32",
  "metal",
  "navatala_transformer_bias_gelu_f32",
  "kernel:metal:navatala_transformer_bias_gelu_f32",
  "abi-r1:metal:navatala_transformer_bias_gelu_f32",
  "abi-r1:metal:navatala_transformer_bias_gelu_f32",
  5,
  kAbiArgs_metal_navatala_transformer_bias_gelu_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_gelu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_gelu_f16 = {
  1,
  "navatala_transformer_gelu_f16",
  "metal",
  "navatala_transformer_gelu_f16",
  "kernel:metal:navatala_transformer_gelu_f16",
  "abi-r1:metal:navatala_transformer_gelu_f16",
  "abi-r1:metal:navatala_transformer_gelu_f16",
  3,
  kAbiArgs_metal_navatala_transformer_gelu_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_silu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_silu_f16 = {
  1,
  "navatala_transformer_silu_f16",
  "metal",
  "navatala_transformer_silu_f16",
  "kernel:metal:navatala_transformer_silu_f16",
  "abi-r1:metal:navatala_transformer_silu_f16",
  "abi-r1:metal:navatala_transformer_silu_f16",
  3,
  kAbiArgs_metal_navatala_transformer_silu_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_relu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_relu_f16 = {
  1,
  "navatala_transformer_relu_f16",
  "metal",
  "navatala_transformer_relu_f16",
  "kernel:metal:navatala_transformer_relu_f16",
  "abi-r1:metal:navatala_transformer_relu_f16",
  "abi-r1:metal:navatala_transformer_relu_f16",
  3,
  kAbiArgs_metal_navatala_transformer_relu_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_quick_gelu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_quick_gelu_f16 = {
  1,
  "navatala_transformer_quick_gelu_f16",
  "metal",
  "navatala_transformer_quick_gelu_f16",
  "kernel:metal:navatala_transformer_quick_gelu_f16",
  "abi-r1:metal:navatala_transformer_quick_gelu_f16",
  "abi-r1:metal:navatala_transformer_quick_gelu_f16",
  3,
  kAbiArgs_metal_navatala_transformer_quick_gelu_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_squared_relu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_squared_relu_f16 = {
  1,
  "navatala_transformer_squared_relu_f16",
  "metal",
  "navatala_transformer_squared_relu_f16",
  "kernel:metal:navatala_transformer_squared_relu_f16",
  "abi-r1:metal:navatala_transformer_squared_relu_f16",
  "abi-r1:metal:navatala_transformer_squared_relu_f16",
  3,
  kAbiArgs_metal_navatala_transformer_squared_relu_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_geglu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_geglu_f32 = {
  1,
  "navatala_transformer_geglu_f32",
  "metal",
  "navatala_transformer_geglu_f32",
  "kernel:metal:navatala_transformer_geglu_f32",
  "abi-r1:metal:navatala_transformer_geglu_f32",
  "abi-r1:metal:navatala_transformer_geglu_f32",
  3,
  kAbiArgs_metal_navatala_transformer_geglu_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_swiglu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_swiglu_f32 = {
  1,
  "navatala_transformer_swiglu_f32",
  "metal",
  "navatala_transformer_swiglu_f32",
  "kernel:metal:navatala_transformer_swiglu_f32",
  "abi-r1:metal:navatala_transformer_swiglu_f32",
  "abi-r1:metal:navatala_transformer_swiglu_f32",
  3,
  kAbiArgs_metal_navatala_transformer_swiglu_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_reglu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_reglu_f32 = {
  1,
  "navatala_transformer_reglu_f32",
  "metal",
  "navatala_transformer_reglu_f32",
  "kernel:metal:navatala_transformer_reglu_f32",
  "abi-r1:metal:navatala_transformer_reglu_f32",
  "abi-r1:metal:navatala_transformer_reglu_f32",
  3,
  kAbiArgs_metal_navatala_transformer_reglu_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_geglu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_geglu_f16 = {
  1,
  "navatala_transformer_geglu_f16",
  "metal",
  "navatala_transformer_geglu_f16",
  "kernel:metal:navatala_transformer_geglu_f16",
  "abi-r1:metal:navatala_transformer_geglu_f16",
  "abi-r1:metal:navatala_transformer_geglu_f16",
  3,
  kAbiArgs_metal_navatala_transformer_geglu_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_swiglu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_swiglu_f16 = {
  1,
  "navatala_transformer_swiglu_f16",
  "metal",
  "navatala_transformer_swiglu_f16",
  "kernel:metal:navatala_transformer_swiglu_f16",
  "abi-r1:metal:navatala_transformer_swiglu_f16",
  "abi-r1:metal:navatala_transformer_swiglu_f16",
  3,
  kAbiArgs_metal_navatala_transformer_swiglu_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_reglu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_reglu_f16 = {
  1,
  "navatala_transformer_reglu_f16",
  "metal",
  "navatala_transformer_reglu_f16",
  "kernel:metal:navatala_transformer_reglu_f16",
  "abi-r1:metal:navatala_transformer_reglu_f16",
  "abi-r1:metal:navatala_transformer_reglu_f16",
  3,
  kAbiArgs_metal_navatala_transformer_reglu_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_gelu_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_gelu_backward_f32 = {
  1,
  "navatala_transformer_gelu_backward_f32",
  "metal",
  "navatala_transformer_gelu_backward_f32",
  "kernel:metal:navatala_transformer_gelu_backward_f32",
  "abi-r1:metal:navatala_transformer_gelu_backward_f32",
  "abi-r1:metal:navatala_transformer_gelu_backward_f32",
  4,
  kAbiArgs_metal_navatala_transformer_gelu_backward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_silu_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_silu_backward_f32 = {
  1,
  "navatala_transformer_silu_backward_f32",
  "metal",
  "navatala_transformer_silu_backward_f32",
  "kernel:metal:navatala_transformer_silu_backward_f32",
  "abi-r1:metal:navatala_transformer_silu_backward_f32",
  "abi-r1:metal:navatala_transformer_silu_backward_f32",
  4,
  kAbiArgs_metal_navatala_transformer_silu_backward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_relu_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_relu_backward_f32 = {
  1,
  "navatala_transformer_relu_backward_f32",
  "metal",
  "navatala_transformer_relu_backward_f32",
  "kernel:metal:navatala_transformer_relu_backward_f32",
  "abi-r1:metal:navatala_transformer_relu_backward_f32",
  "abi-r1:metal:navatala_transformer_relu_backward_f32",
  4,
  kAbiArgs_metal_navatala_transformer_relu_backward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_quick_gelu_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_quick_gelu_backward_f32 = {
  1,
  "navatala_transformer_quick_gelu_backward_f32",
  "metal",
  "navatala_transformer_quick_gelu_backward_f32",
  "kernel:metal:navatala_transformer_quick_gelu_backward_f32",
  "abi-r1:metal:navatala_transformer_quick_gelu_backward_f32",
  "abi-r1:metal:navatala_transformer_quick_gelu_backward_f32",
  4,
  kAbiArgs_metal_navatala_transformer_quick_gelu_backward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_gelu_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_gelu_backward_f16 = {
  1,
  "navatala_transformer_gelu_backward_f16",
  "metal",
  "navatala_transformer_gelu_backward_f16",
  "kernel:metal:navatala_transformer_gelu_backward_f16",
  "abi-r1:metal:navatala_transformer_gelu_backward_f16",
  "abi-r1:metal:navatala_transformer_gelu_backward_f16",
  4,
  kAbiArgs_metal_navatala_transformer_gelu_backward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_silu_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_silu_backward_f16 = {
  1,
  "navatala_transformer_silu_backward_f16",
  "metal",
  "navatala_transformer_silu_backward_f16",
  "kernel:metal:navatala_transformer_silu_backward_f16",
  "abi-r1:metal:navatala_transformer_silu_backward_f16",
  "abi-r1:metal:navatala_transformer_silu_backward_f16",
  4,
  kAbiArgs_metal_navatala_transformer_silu_backward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_relu_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_relu_backward_f16 = {
  1,
  "navatala_transformer_relu_backward_f16",
  "metal",
  "navatala_transformer_relu_backward_f16",
  "kernel:metal:navatala_transformer_relu_backward_f16",
  "abi-r1:metal:navatala_transformer_relu_backward_f16",
  "abi-r1:metal:navatala_transformer_relu_backward_f16",
  4,
  kAbiArgs_metal_navatala_transformer_relu_backward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_quick_gelu_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_quick_gelu_backward_f16 = {
  1,
  "navatala_transformer_quick_gelu_backward_f16",
  "metal",
  "navatala_transformer_quick_gelu_backward_f16",
  "kernel:metal:navatala_transformer_quick_gelu_backward_f16",
  "abi-r1:metal:navatala_transformer_quick_gelu_backward_f16",
  "abi-r1:metal:navatala_transformer_quick_gelu_backward_f16",
  4,
  kAbiArgs_metal_navatala_transformer_quick_gelu_backward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_rms_norm_forward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_rms_norm_forward_f32 = {
  1,
  "navatala_transformer_rms_norm_forward_f32",
  "metal",
  "navatala_transformer_rms_norm_forward_f32",
  "kernel:metal:navatala_transformer_rms_norm_forward_f32",
  "abi-r1:metal:navatala_transformer_rms_norm_forward_f32",
  "abi-r1:metal:navatala_transformer_rms_norm_forward_f32",
  6,
  kAbiArgs_metal_navatala_transformer_rms_norm_forward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_rms_norm_forward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_rms_norm_forward_f16 = {
  1,
  "navatala_transformer_rms_norm_forward_f16",
  "metal",
  "navatala_transformer_rms_norm_forward_f16",
  "kernel:metal:navatala_transformer_rms_norm_forward_f16",
  "abi-r1:metal:navatala_transformer_rms_norm_forward_f16",
  "abi-r1:metal:navatala_transformer_rms_norm_forward_f16",
  6,
  kAbiArgs_metal_navatala_transformer_rms_norm_forward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_layer_norm_forward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_layer_norm_forward_f32 = {
  1,
  "navatala_transformer_layer_norm_forward_f32",
  "metal",
  "navatala_transformer_layer_norm_forward_f32",
  "kernel:metal:navatala_transformer_layer_norm_forward_f32",
  "abi-r1:metal:navatala_transformer_layer_norm_forward_f32",
  "abi-r1:metal:navatala_transformer_layer_norm_forward_f32",
  7,
  kAbiArgs_metal_navatala_transformer_layer_norm_forward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_layer_norm_forward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_layer_norm_forward_f16 = {
  1,
  "navatala_transformer_layer_norm_forward_f16",
  "metal",
  "navatala_transformer_layer_norm_forward_f16",
  "kernel:metal:navatala_transformer_layer_norm_forward_f16",
  "abi-r1:metal:navatala_transformer_layer_norm_forward_f16",
  "abi-r1:metal:navatala_transformer_layer_norm_forward_f16",
  7,
  kAbiArgs_metal_navatala_transformer_layer_norm_forward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_layer_norm_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mean", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rsigma", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradGamma", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradBeta", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_layer_norm_backward_f32 = {
  1,
  "navatala_transformer_layer_norm_backward_f32",
  "metal",
  "navatala_transformer_layer_norm_backward_f32",
  "kernel:metal:navatala_transformer_layer_norm_backward_f32",
  "abi-r1:metal:navatala_transformer_layer_norm_backward_f32",
  "abi-r1:metal:navatala_transformer_layer_norm_backward_f32",
  10,
  kAbiArgs_metal_navatala_transformer_layer_norm_backward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_layer_norm_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mean", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rsigma", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradGamma", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradBeta", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_layer_norm_backward_f16 = {
  1,
  "navatala_transformer_layer_norm_backward_f16",
  "metal",
  "navatala_transformer_layer_norm_backward_f16",
  "kernel:metal:navatala_transformer_layer_norm_backward_f16",
  "abi-r1:metal:navatala_transformer_layer_norm_backward_f16",
  "abi-r1:metal:navatala_transformer_layer_norm_backward_f16",
  10,
  kAbiArgs_metal_navatala_transformer_layer_norm_backward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_rms_norm_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rsigma", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradGamma", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_rms_norm_backward_f32 = {
  1,
  "navatala_transformer_rms_norm_backward_f32",
  "metal",
  "navatala_transformer_rms_norm_backward_f32",
  "kernel:metal:navatala_transformer_rms_norm_backward_f32",
  "abi-r1:metal:navatala_transformer_rms_norm_backward_f32",
  "abi-r1:metal:navatala_transformer_rms_norm_backward_f32",
  8,
  kAbiArgs_metal_navatala_transformer_rms_norm_backward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_rms_norm_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rsigma", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradGamma", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_rms_norm_backward_f16 = {
  1,
  "navatala_transformer_rms_norm_backward_f16",
  "metal",
  "navatala_transformer_rms_norm_backward_f16",
  "kernel:metal:navatala_transformer_rms_norm_backward_f16",
  "abi-r1:metal:navatala_transformer_rms_norm_backward_f16",
  "abi-r1:metal:navatala_transformer_rms_norm_backward_f16",
  8,
  kAbiArgs_metal_navatala_transformer_rms_norm_backward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_layer_norm_multi_pass_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_layer_norm_multi_pass_f32 = {
  1,
  "navatala_transformer_layer_norm_multi_pass_f32",
  "metal",
  "navatala_transformer_layer_norm_multi_pass_f32",
  "kernel:metal:navatala_transformer_layer_norm_multi_pass_f32",
  "abi-r1:metal:navatala_transformer_layer_norm_multi_pass_f32",
  "abi-r1:metal:navatala_transformer_layer_norm_multi_pass_f32",
  7,
  kAbiArgs_metal_navatala_transformer_layer_norm_multi_pass_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_layer_norm_multi_pass_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_layer_norm_multi_pass_f16 = {
  1,
  "navatala_transformer_layer_norm_multi_pass_f16",
  "metal",
  "navatala_transformer_layer_norm_multi_pass_f16",
  "kernel:metal:navatala_transformer_layer_norm_multi_pass_f16",
  "abi-r1:metal:navatala_transformer_layer_norm_multi_pass_f16",
  "abi-r1:metal:navatala_transformer_layer_norm_multi_pass_f16",
  7,
  kAbiArgs_metal_navatala_transformer_layer_norm_multi_pass_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_rms_norm_multi_pass_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_rms_norm_multi_pass_f32 = {
  1,
  "navatala_transformer_rms_norm_multi_pass_f32",
  "metal",
  "navatala_transformer_rms_norm_multi_pass_f32",
  "kernel:metal:navatala_transformer_rms_norm_multi_pass_f32",
  "abi-r1:metal:navatala_transformer_rms_norm_multi_pass_f32",
  "abi-r1:metal:navatala_transformer_rms_norm_multi_pass_f32",
  6,
  kAbiArgs_metal_navatala_transformer_rms_norm_multi_pass_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_rms_norm_multi_pass_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_rms_norm_multi_pass_f16 = {
  1,
  "navatala_transformer_rms_norm_multi_pass_f16",
  "metal",
  "navatala_transformer_rms_norm_multi_pass_f16",
  "kernel:metal:navatala_transformer_rms_norm_multi_pass_f16",
  "abi-r1:metal:navatala_transformer_rms_norm_multi_pass_f16",
  "abi-r1:metal:navatala_transformer_rms_norm_multi_pass_f16",
  6,
  kAbiArgs_metal_navatala_transformer_rms_norm_multi_pass_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_softmax_forward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_softmax_forward_f32 = {
  1,
  "navatala_transformer_softmax_forward_f32",
  "metal",
  "navatala_transformer_softmax_forward_f32",
  "kernel:metal:navatala_transformer_softmax_forward_f32",
  "abi-r1:metal:navatala_transformer_softmax_forward_f32",
  "abi-r1:metal:navatala_transformer_softmax_forward_f32",
  4,
  kAbiArgs_metal_navatala_transformer_softmax_forward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_softmax_forward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_softmax_forward_f16 = {
  1,
  "navatala_transformer_softmax_forward_f16",
  "metal",
  "navatala_transformer_softmax_forward_f16",
  "kernel:metal:navatala_transformer_softmax_forward_f16",
  "abi-r1:metal:navatala_transformer_softmax_forward_f16",
  "abi-r1:metal:navatala_transformer_softmax_forward_f16",
  4,
  kAbiArgs_metal_navatala_transformer_softmax_forward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_softmax_backward_f32[] = {
  { "output", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_softmax_backward_f32 = {
  1,
  "navatala_transformer_softmax_backward_f32",
  "metal",
  "navatala_transformer_softmax_backward_f32",
  "kernel:metal:navatala_transformer_softmax_backward_f32",
  "abi-r1:metal:navatala_transformer_softmax_backward_f32",
  "abi-r1:metal:navatala_transformer_softmax_backward_f32",
  5,
  kAbiArgs_metal_navatala_transformer_softmax_backward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_softmax_backward_f16[] = {
  { "output", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_softmax_backward_f16 = {
  1,
  "navatala_transformer_softmax_backward_f16",
  "metal",
  "navatala_transformer_softmax_backward_f16",
  "kernel:metal:navatala_transformer_softmax_backward_f16",
  "abi-r1:metal:navatala_transformer_softmax_backward_f16",
  "abi-r1:metal:navatala_transformer_softmax_backward_f16",
  5,
  kAbiArgs_metal_navatala_transformer_softmax_backward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_softmax_with_mask_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "isCausal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_softmax_with_mask_f32 = {
  1,
  "navatala_transformer_softmax_with_mask_f32",
  "metal",
  "navatala_transformer_softmax_with_mask_f32",
  "kernel:metal:navatala_transformer_softmax_with_mask_f32",
  "abi-r1:metal:navatala_transformer_softmax_with_mask_f32",
  "abi-r1:metal:navatala_transformer_softmax_with_mask_f32",
  5,
  kAbiArgs_metal_navatala_transformer_softmax_with_mask_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_softmax_with_mask_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "isCausal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_softmax_with_mask_f16 = {
  1,
  "navatala_transformer_softmax_with_mask_f16",
  "metal",
  "navatala_transformer_softmax_with_mask_f16",
  "kernel:metal:navatala_transformer_softmax_with_mask_f16",
  "abi-r1:metal:navatala_transformer_softmax_with_mask_f16",
  "abi-r1:metal:navatala_transformer_softmax_with_mask_f16",
  5,
  kAbiArgs_metal_navatala_transformer_softmax_with_mask_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_softmax_with_mask_backward_f32[] = {
  { "output", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "isCausal", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_softmax_with_mask_backward_f32 = {
  1,
  "navatala_transformer_softmax_with_mask_backward_f32",
  "metal",
  "navatala_transformer_softmax_with_mask_backward_f32",
  "kernel:metal:navatala_transformer_softmax_with_mask_backward_f32",
  "abi-r1:metal:navatala_transformer_softmax_with_mask_backward_f32",
  "abi-r1:metal:navatala_transformer_softmax_with_mask_backward_f32",
  6,
  kAbiArgs_metal_navatala_transformer_softmax_with_mask_backward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_softmax_with_mask_backward_f16[] = {
  { "output", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "isCausal", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_softmax_with_mask_backward_f16 = {
  1,
  "navatala_transformer_softmax_with_mask_backward_f16",
  "metal",
  "navatala_transformer_softmax_with_mask_backward_f16",
  "kernel:metal:navatala_transformer_softmax_with_mask_backward_f16",
  "abi-r1:metal:navatala_transformer_softmax_with_mask_backward_f16",
  "abi-r1:metal:navatala_transformer_softmax_with_mask_backward_f16",
  6,
  kAbiArgs_metal_navatala_transformer_softmax_with_mask_backward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_softmax_multi_pass_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_softmax_multi_pass_f32 = {
  1,
  "navatala_transformer_softmax_multi_pass_f32",
  "metal",
  "navatala_transformer_softmax_multi_pass_f32",
  "kernel:metal:navatala_transformer_softmax_multi_pass_f32",
  "abi-r1:metal:navatala_transformer_softmax_multi_pass_f32",
  "abi-r1:metal:navatala_transformer_softmax_multi_pass_f32",
  4,
  kAbiArgs_metal_navatala_transformer_softmax_multi_pass_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_softmax_multi_pass_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_softmax_multi_pass_f16 = {
  1,
  "navatala_transformer_softmax_multi_pass_f16",
  "metal",
  "navatala_transformer_softmax_multi_pass_f16",
  "kernel:metal:navatala_transformer_softmax_multi_pass_f16",
  "abi-r1:metal:navatala_transformer_softmax_multi_pass_f16",
  "abi-r1:metal:navatala_transformer_softmax_multi_pass_f16",
  4,
  kAbiArgs_metal_navatala_transformer_softmax_multi_pass_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_cast_f32_to_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_cast_f32_to_f16 = {
  1,
  "navatala_transformer_cast_f32_to_f16",
  "metal",
  "navatala_transformer_cast_f32_to_f16",
  "kernel:metal:navatala_transformer_cast_f32_to_f16",
  "abi-r1:metal:navatala_transformer_cast_f32_to_f16",
  "abi-r1:metal:navatala_transformer_cast_f32_to_f16",
  3,
  kAbiArgs_metal_navatala_transformer_cast_f32_to_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_cast_f16_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_cast_f16_to_f32 = {
  1,
  "navatala_transformer_cast_f16_to_f32",
  "metal",
  "navatala_transformer_cast_f16_to_f32",
  "kernel:metal:navatala_transformer_cast_f16_to_f32",
  "abi-r1:metal:navatala_transformer_cast_f16_to_f32",
  "abi-r1:metal:navatala_transformer_cast_f16_to_f32",
  3,
  kAbiArgs_metal_navatala_transformer_cast_f16_to_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_transpose_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_transpose_f32 = {
  1,
  "navatala_transformer_transpose_f32",
  "metal",
  "navatala_transformer_transpose_f32",
  "kernel:metal:navatala_transformer_transpose_f32",
  "abi-r1:metal:navatala_transformer_transpose_f32",
  "abi-r1:metal:navatala_transformer_transpose_f32",
  4,
  kAbiArgs_metal_navatala_transformer_transpose_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_transpose_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_transpose_f16 = {
  1,
  "navatala_transformer_transpose_f16",
  "metal",
  "navatala_transformer_transpose_f16",
  "kernel:metal:navatala_transformer_transpose_f16",
  "abi-r1:metal:navatala_transformer_transpose_f16",
  "abi-r1:metal:navatala_transformer_transpose_f16",
  4,
  kAbiArgs_metal_navatala_transformer_transpose_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_fused_cast_transpose_f32_to_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_fused_cast_transpose_f32_to_f16 = {
  1,
  "navatala_transformer_fused_cast_transpose_f32_to_f16",
  "metal",
  "navatala_transformer_fused_cast_transpose_f32_to_f16",
  "kernel:metal:navatala_transformer_fused_cast_transpose_f32_to_f16",
  "abi-r1:metal:navatala_transformer_fused_cast_transpose_f32_to_f16",
  "abi-r1:metal:navatala_transformer_fused_cast_transpose_f32_to_f16",
  4,
  kAbiArgs_metal_navatala_transformer_fused_cast_transpose_f32_to_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_cast_f32_to_f_p8_e4_m3[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_cast_f32_to_f_p8_e4_m3 = {
  1,
  "navatala_transformer_cast_f32_to_f_p8_e4_m3",
  "metal",
  "navatala_transformer_cast_f32_to_f_p8_e4_m3",
  "kernel:metal:navatala_transformer_cast_f32_to_f_p8_e4_m3",
  "abi-r1:metal:navatala_transformer_cast_f32_to_f_p8_e4_m3",
  "abi-r1:metal:navatala_transformer_cast_f32_to_f_p8_e4_m3",
  4,
  kAbiArgs_metal_navatala_transformer_cast_f32_to_f_p8_e4_m3
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_cast_f_p8_e4_m3_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_cast_f_p8_e4_m3_to_f32 = {
  1,
  "navatala_transformer_cast_f_p8_e4_m3_to_f32",
  "metal",
  "navatala_transformer_cast_f_p8_e4_m3_to_f32",
  "kernel:metal:navatala_transformer_cast_f_p8_e4_m3_to_f32",
  "abi-r1:metal:navatala_transformer_cast_f_p8_e4_m3_to_f32",
  "abi-r1:metal:navatala_transformer_cast_f_p8_e4_m3_to_f32",
  4,
  kAbiArgs_metal_navatala_transformer_cast_f_p8_e4_m3_to_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_cast_f32_to_f_p8_e5_m2[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_cast_f32_to_f_p8_e5_m2 = {
  1,
  "navatala_transformer_cast_f32_to_f_p8_e5_m2",
  "metal",
  "navatala_transformer_cast_f32_to_f_p8_e5_m2",
  "kernel:metal:navatala_transformer_cast_f32_to_f_p8_e5_m2",
  "abi-r1:metal:navatala_transformer_cast_f32_to_f_p8_e5_m2",
  "abi-r1:metal:navatala_transformer_cast_f32_to_f_p8_e5_m2",
  4,
  kAbiArgs_metal_navatala_transformer_cast_f32_to_f_p8_e5_m2
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_cast_f_p8_e5_m2_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_cast_f_p8_e5_m2_to_f32 = {
  1,
  "navatala_transformer_cast_f_p8_e5_m2_to_f32",
  "metal",
  "navatala_transformer_cast_f_p8_e5_m2_to_f32",
  "kernel:metal:navatala_transformer_cast_f_p8_e5_m2_to_f32",
  "abi-r1:metal:navatala_transformer_cast_f_p8_e5_m2_to_f32",
  "abi-r1:metal:navatala_transformer_cast_f_p8_e5_m2_to_f32",
  4,
  kAbiArgs_metal_navatala_transformer_cast_f_p8_e5_m2_to_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_fused_cast_transpose_f_p8[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_fused_cast_transpose_f_p8 = {
  1,
  "navatala_transformer_fused_cast_transpose_f_p8",
  "metal",
  "navatala_transformer_fused_cast_transpose_f_p8",
  "kernel:metal:navatala_transformer_fused_cast_transpose_f_p8",
  "abi-r1:metal:navatala_transformer_fused_cast_transpose_f_p8",
  "abi-r1:metal:navatala_transformer_fused_cast_transpose_f_p8",
  5,
  kAbiArgs_metal_navatala_transformer_fused_cast_transpose_f_p8
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_transpose4_d_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dim0", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim1", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim2", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim3", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm0", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm1", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm2", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm3", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_transpose4_d_f32 = {
  1,
  "navatala_transformer_transpose4_d_f32",
  "metal",
  "navatala_transformer_transpose4_d_f32",
  "kernel:metal:navatala_transformer_transpose4_d_f32",
  "abi-r1:metal:navatala_transformer_transpose4_d_f32",
  "abi-r1:metal:navatala_transformer_transpose4_d_f32",
  10,
  kAbiArgs_metal_navatala_transformer_transpose4_d_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_transpose4_d_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dim0", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim1", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim2", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim3", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm0", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm1", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm2", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm3", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_transpose4_d_f16 = {
  1,
  "navatala_transformer_transpose4_d_f16",
  "metal",
  "navatala_transformer_transpose4_d_f16",
  "kernel:metal:navatala_transformer_transpose4_d_f16",
  "abi-r1:metal:navatala_transformer_transpose4_d_f16",
  "abi-r1:metal:navatala_transformer_transpose4_d_f16",
  10,
  kAbiArgs_metal_navatala_transformer_transpose4_d_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_dropout_forward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "prob", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_dropout_forward_f32 = {
  1,
  "navatala_transformer_dropout_forward_f32",
  "metal",
  "navatala_transformer_dropout_forward_f32",
  "kernel:metal:navatala_transformer_dropout_forward_f32",
  "abi-r1:metal:navatala_transformer_dropout_forward_f32",
  "abi-r1:metal:navatala_transformer_dropout_forward_f32",
  5,
  kAbiArgs_metal_navatala_transformer_dropout_forward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_dropout_forward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "prob", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_dropout_forward_f16 = {
  1,
  "navatala_transformer_dropout_forward_f16",
  "metal",
  "navatala_transformer_dropout_forward_f16",
  "kernel:metal:navatala_transformer_dropout_forward_f16",
  "abi-r1:metal:navatala_transformer_dropout_forward_f16",
  "abi-r1:metal:navatala_transformer_dropout_forward_f16",
  5,
  kAbiArgs_metal_navatala_transformer_dropout_forward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_dropout_backward_f32[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "prob", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_dropout_backward_f32 = {
  1,
  "navatala_transformer_dropout_backward_f32",
  "metal",
  "navatala_transformer_dropout_backward_f32",
  "kernel:metal:navatala_transformer_dropout_backward_f32",
  "abi-r1:metal:navatala_transformer_dropout_backward_f32",
  "abi-r1:metal:navatala_transformer_dropout_backward_f32",
  5,
  kAbiArgs_metal_navatala_transformer_dropout_backward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_dropout_backward_f16[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "prob", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_dropout_backward_f16 = {
  1,
  "navatala_transformer_dropout_backward_f16",
  "metal",
  "navatala_transformer_dropout_backward_f16",
  "kernel:metal:navatala_transformer_dropout_backward_f16",
  "abi-r1:metal:navatala_transformer_dropout_backward_f16",
  "abi-r1:metal:navatala_transformer_dropout_backward_f16",
  5,
  kAbiArgs_metal_navatala_transformer_dropout_backward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_rope_forward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_rope_forward_f32 = {
  1,
  "navatala_transformer_rope_forward_f32",
  "metal",
  "navatala_transformer_rope_forward_f32",
  "kernel:metal:navatala_transformer_rope_forward_f32",
  "abi-r1:metal:navatala_transformer_rope_forward_f32",
  "abi-r1:metal:navatala_transformer_rope_forward_f32",
  7,
  kAbiArgs_metal_navatala_transformer_rope_forward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_rope_forward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_rope_forward_f16 = {
  1,
  "navatala_transformer_rope_forward_f16",
  "metal",
  "navatala_transformer_rope_forward_f16",
  "kernel:metal:navatala_transformer_rope_forward_f16",
  "abi-r1:metal:navatala_transformer_rope_forward_f16",
  "abi-r1:metal:navatala_transformer_rope_forward_f16",
  7,
  kAbiArgs_metal_navatala_transformer_rope_forward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_rope_with_cache_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cosCache", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sinCache", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_rope_with_cache_f32 = {
  1,
  "navatala_transformer_rope_with_cache_f32",
  "metal",
  "navatala_transformer_rope_with_cache_f32",
  "kernel:metal:navatala_transformer_rope_with_cache_f32",
  "abi-r1:metal:navatala_transformer_rope_with_cache_f32",
  "abi-r1:metal:navatala_transformer_rope_with_cache_f32",
  8,
  kAbiArgs_metal_navatala_transformer_rope_with_cache_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_rope_backward_f32[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_rope_backward_f32 = {
  1,
  "navatala_transformer_rope_backward_f32",
  "metal",
  "navatala_transformer_rope_backward_f32",
  "kernel:metal:navatala_transformer_rope_backward_f32",
  "abi-r1:metal:navatala_transformer_rope_backward_f32",
  "abi-r1:metal:navatala_transformer_rope_backward_f32",
  7,
  kAbiArgs_metal_navatala_transformer_rope_backward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_rope_backward_f16[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_rope_backward_f16 = {
  1,
  "navatala_transformer_rope_backward_f16",
  "metal",
  "navatala_transformer_rope_backward_f16",
  "kernel:metal:navatala_transformer_rope_backward_f16",
  "abi-r1:metal:navatala_transformer_rope_backward_f16",
  "abi-r1:metal:navatala_transformer_rope_backward_f16",
  7,
  kAbiArgs_metal_navatala_transformer_rope_backward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_rope_interleaved_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_rope_interleaved_f32 = {
  1,
  "navatala_transformer_rope_interleaved_f32",
  "metal",
  "navatala_transformer_rope_interleaved_f32",
  "kernel:metal:navatala_transformer_rope_interleaved_f32",
  "abi-r1:metal:navatala_transformer_rope_interleaved_f32",
  "abi-r1:metal:navatala_transformer_rope_interleaved_f32",
  7,
  kAbiArgs_metal_navatala_transformer_rope_interleaved_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_rope_interleaved_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_rope_interleaved_f16 = {
  1,
  "navatala_transformer_rope_interleaved_f16",
  "metal",
  "navatala_transformer_rope_interleaved_f16",
  "kernel:metal:navatala_transformer_rope_interleaved_f16",
  "abi-r1:metal:navatala_transformer_rope_interleaved_f16",
  "abi-r1:metal:navatala_transformer_rope_interleaved_f16",
  7,
  kAbiArgs_metal_navatala_transformer_rope_interleaved_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_simple_attention_f16[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_simple_attention_f16 = {
  1,
  "navatala_transformer_simple_attention_f16",
  "metal",
  "navatala_transformer_simple_attention_f16",
  "kernel:metal:navatala_transformer_simple_attention_f16",
  "abi-r1:metal:navatala_transformer_simple_attention_f16",
  "abi-r1:metal:navatala_transformer_simple_attention_f16",
  8,
  kAbiArgs_metal_navatala_transformer_simple_attention_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_simple_attention_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_simple_attention_f32 = {
  1,
  "navatala_transformer_simple_attention_f32",
  "metal",
  "navatala_transformer_simple_attention_f32",
  "kernel:metal:navatala_transformer_simple_attention_f32",
  "abi-r1:metal:navatala_transformer_simple_attention_f32",
  "abi-r1:metal:navatala_transformer_simple_attention_f32",
  8,
  kAbiArgs_metal_navatala_transformer_simple_attention_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_paged_attention_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kvCache", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "blockTables", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "contextLens", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numHeads", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "pageSize", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxBlocks", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_paged_attention_f32 = {
  1,
  "navatala_transformer_paged_attention_f32",
  "metal",
  "navatala_transformer_paged_attention_f32",
  "kernel:metal:navatala_transformer_paged_attention_f32",
  "abi-r1:metal:navatala_transformer_paged_attention_f32",
  "abi-r1:metal:navatala_transformer_paged_attention_f32",
  9,
  kAbiArgs_metal_navatala_transformer_paged_attention_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_simple_attention_with_mask_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_simple_attention_with_mask_f32 = {
  1,
  "navatala_transformer_simple_attention_with_mask_f32",
  "metal",
  "navatala_transformer_simple_attention_with_mask_f32",
  "kernel:metal:navatala_transformer_simple_attention_with_mask_f32",
  "abi-r1:metal:navatala_transformer_simple_attention_with_mask_f32",
  "abi-r1:metal:navatala_transformer_simple_attention_with_mask_f32",
  8,
  kAbiArgs_metal_navatala_transformer_simple_attention_with_mask_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_simple_attention_with_mask_f16[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_simple_attention_with_mask_f16 = {
  1,
  "navatala_transformer_simple_attention_with_mask_f16",
  "metal",
  "navatala_transformer_simple_attention_with_mask_f16",
  "kernel:metal:navatala_transformer_simple_attention_with_mask_f16",
  "abi-r1:metal:navatala_transformer_simple_attention_with_mask_f16",
  "abi-r1:metal:navatala_transformer_simple_attention_with_mask_f16",
  8,
  kAbiArgs_metal_navatala_transformer_simple_attention_with_mask_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_simple_attention_with_padding_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_simple_attention_with_padding_f32 = {
  1,
  "navatala_transformer_simple_attention_with_padding_f32",
  "metal",
  "navatala_transformer_simple_attention_with_padding_f32",
  "kernel:metal:navatala_transformer_simple_attention_with_padding_f32",
  "abi-r1:metal:navatala_transformer_simple_attention_with_padding_f32",
  "abi-r1:metal:navatala_transformer_simple_attention_with_padding_f32",
  9,
  kAbiArgs_metal_navatala_transformer_simple_attention_with_padding_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_simple_attention_with_padding_f16[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_simple_attention_with_padding_f16 = {
  1,
  "navatala_transformer_simple_attention_with_padding_f16",
  "metal",
  "navatala_transformer_simple_attention_with_padding_f16",
  "kernel:metal:navatala_transformer_simple_attention_with_padding_f16",
  "abi-r1:metal:navatala_transformer_simple_attention_with_padding_f16",
  "abi-r1:metal:navatala_transformer_simple_attention_with_padding_f16",
  9,
  kAbiArgs_metal_navatala_transformer_simple_attention_with_padding_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_simple_attention_backward_f32[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "query", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradQuery", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradKey", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradValue", 10, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_simple_attention_backward_f32 = {
  1,
  "navatala_transformer_simple_attention_backward_f32",
  "metal",
  "navatala_transformer_simple_attention_backward_f32",
  "kernel:metal:navatala_transformer_simple_attention_backward_f32",
  "abi-r1:metal:navatala_transformer_simple_attention_backward_f32",
  "abi-r1:metal:navatala_transformer_simple_attention_backward_f32",
  11,
  kAbiArgs_metal_navatala_transformer_simple_attention_backward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_simple_attention_backward_f16[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "query", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradQuery", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradKey", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradValue", 10, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_simple_attention_backward_f16 = {
  1,
  "navatala_transformer_simple_attention_backward_f16",
  "metal",
  "navatala_transformer_simple_attention_backward_f16",
  "kernel:metal:navatala_transformer_simple_attention_backward_f16",
  "abi-r1:metal:navatala_transformer_simple_attention_backward_f16",
  "abi-r1:metal:navatala_transformer_simple_attention_backward_f16",
  11,
  kAbiArgs_metal_navatala_transformer_simple_attention_backward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_kv_cache_append_f16[] = {
  { "newK", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "newV", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "blockTables", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqPositions", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numHeads", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "pageSize", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxBlocks", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kvCache", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_kv_cache_append_f16 = {
  1,
  "navatala_transformer_kv_cache_append_f16",
  "metal",
  "navatala_transformer_kv_cache_append_f16",
  "kernel:metal:navatala_transformer_kv_cache_append_f16",
  "abi-r1:metal:navatala_transformer_kv_cache_append_f16",
  "abi-r1:metal:navatala_transformer_kv_cache_append_f16",
  9,
  kAbiArgs_metal_navatala_transformer_kv_cache_append_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_kv_cache_append_f32[] = {
  { "newK", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "newV", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "blockTables", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqPositions", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numHeads", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "pageSize", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxBlocks", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kvCache", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_kv_cache_append_f32 = {
  1,
  "navatala_transformer_kv_cache_append_f32",
  "metal",
  "navatala_transformer_kv_cache_append_f32",
  "kernel:metal:navatala_transformer_kv_cache_append_f32",
  "abi-r1:metal:navatala_transformer_kv_cache_append_f32",
  "abi-r1:metal:navatala_transformer_kv_cache_append_f32",
  9,
  kAbiArgs_metal_navatala_transformer_kv_cache_append_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_kv_cache_rotate_f16[] = {
  { "blockTables", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "contextLens", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "shiftAmount", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "pageSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxBlocks", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kvCache", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_kv_cache_rotate_f16 = {
  1,
  "navatala_transformer_kv_cache_rotate_f16",
  "metal",
  "navatala_transformer_kv_cache_rotate_f16",
  "kernel:metal:navatala_transformer_kv_cache_rotate_f16",
  "abi-r1:metal:navatala_transformer_kv_cache_rotate_f16",
  "abi-r1:metal:navatala_transformer_kv_cache_rotate_f16",
  8,
  kAbiArgs_metal_navatala_transformer_kv_cache_rotate_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_tiled_gemm_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_tiled_gemm_f16 = {
  1,
  "navatala_transformer_tiled_gemm_f16",
  "metal",
  "navatala_transformer_tiled_gemm_f16",
  "kernel:metal:navatala_transformer_tiled_gemm_f16",
  "abi-r1:metal:navatala_transformer_tiled_gemm_f16",
  "abi-r1:metal:navatala_transformer_tiled_gemm_f16",
  8,
  kAbiArgs_metal_navatala_transformer_tiled_gemm_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_tiled_gemm_f16_f32_out[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_tiled_gemm_f16_f32_out = {
  1,
  "navatala_transformer_tiled_gemm_f16_f32_out",
  "metal",
  "navatala_transformer_tiled_gemm_f16_f32_out",
  "kernel:metal:navatala_transformer_tiled_gemm_f16_f32_out",
  "abi-r1:metal:navatala_transformer_tiled_gemm_f16_f32_out",
  "abi-r1:metal:navatala_transformer_tiled_gemm_f16_f32_out",
  8,
  kAbiArgs_metal_navatala_transformer_tiled_gemm_f16_f32_out
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_tiled_gemm_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_tiled_gemm_f32 = {
  1,
  "navatala_transformer_tiled_gemm_f32",
  "metal",
  "navatala_transformer_tiled_gemm_f32",
  "kernel:metal:navatala_transformer_tiled_gemm_f32",
  "abi-r1:metal:navatala_transformer_tiled_gemm_f32",
  "abi-r1:metal:navatala_transformer_tiled_gemm_f32",
  8,
  kAbiArgs_metal_navatala_transformer_tiled_gemm_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_tiled_gemm_backward_f32[] = {
  { "gradC", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradA", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_tiled_gemm_backward_f32 = {
  1,
  "navatala_transformer_tiled_gemm_backward_f32",
  "metal",
  "navatala_transformer_tiled_gemm_backward_f32",
  "kernel:metal:navatala_transformer_tiled_gemm_backward_f32",
  "abi-r1:metal:navatala_transformer_tiled_gemm_backward_f32",
  "abi-r1:metal:navatala_transformer_tiled_gemm_backward_f32",
  6,
  kAbiArgs_metal_navatala_transformer_tiled_gemm_backward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_tiled_gemm_backward_f16[] = {
  { "gradC", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradA", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_tiled_gemm_backward_f16 = {
  1,
  "navatala_transformer_tiled_gemm_backward_f16",
  "metal",
  "navatala_transformer_tiled_gemm_backward_f16",
  "kernel:metal:navatala_transformer_tiled_gemm_backward_f16",
  "abi-r1:metal:navatala_transformer_tiled_gemm_backward_f16",
  "abi-r1:metal:navatala_transformer_tiled_gemm_backward_f16",
  6,
  kAbiArgs_metal_navatala_transformer_tiled_gemm_backward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_multi_tensor_scale_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_multi_tensor_scale_f32 = {
  1,
  "navatala_transformer_multi_tensor_scale_f32",
  "metal",
  "navatala_transformer_multi_tensor_scale_f32",
  "kernel:metal:navatala_transformer_multi_tensor_scale_f32",
  "abi-r1:metal:navatala_transformer_multi_tensor_scale_f32",
  "abi-r1:metal:navatala_transformer_multi_tensor_scale_f32",
  4,
  kAbiArgs_metal_navatala_transformer_multi_tensor_scale_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_multi_tensor_scale_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_multi_tensor_scale_f16 = {
  1,
  "navatala_transformer_multi_tensor_scale_f16",
  "metal",
  "navatala_transformer_multi_tensor_scale_f16",
  "kernel:metal:navatala_transformer_multi_tensor_scale_f16",
  "abi-r1:metal:navatala_transformer_multi_tensor_scale_f16",
  "abi-r1:metal:navatala_transformer_multi_tensor_scale_f16",
  4,
  kAbiArgs_metal_navatala_transformer_multi_tensor_scale_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_multi_tensor_add_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_multi_tensor_add_f32 = {
  1,
  "navatala_transformer_multi_tensor_add_f32",
  "metal",
  "navatala_transformer_multi_tensor_add_f32",
  "kernel:metal:navatala_transformer_multi_tensor_add_f32",
  "abi-r1:metal:navatala_transformer_multi_tensor_add_f32",
  "abi-r1:metal:navatala_transformer_multi_tensor_add_f32",
  4,
  kAbiArgs_metal_navatala_transformer_multi_tensor_add_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_multi_tensor_add_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_multi_tensor_add_f16 = {
  1,
  "navatala_transformer_multi_tensor_add_f16",
  "metal",
  "navatala_transformer_multi_tensor_add_f16",
  "kernel:metal:navatala_transformer_multi_tensor_add_f16",
  "abi-r1:metal:navatala_transformer_multi_tensor_add_f16",
  "abi-r1:metal:navatala_transformer_multi_tensor_add_f16",
  4,
  kAbiArgs_metal_navatala_transformer_multi_tensor_add_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_multi_tensor_copy_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "totalElements", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_multi_tensor_copy_f32 = {
  1,
  "navatala_transformer_multi_tensor_copy_f32",
  "metal",
  "navatala_transformer_multi_tensor_copy_f32",
  "kernel:metal:navatala_transformer_multi_tensor_copy_f32",
  "abi-r1:metal:navatala_transformer_multi_tensor_copy_f32",
  "abi-r1:metal:navatala_transformer_multi_tensor_copy_f32",
  3,
  kAbiArgs_metal_navatala_transformer_multi_tensor_copy_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_multi_tensor_copy_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "totalElements", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_multi_tensor_copy_f16 = {
  1,
  "navatala_transformer_multi_tensor_copy_f16",
  "metal",
  "navatala_transformer_multi_tensor_copy_f16",
  "kernel:metal:navatala_transformer_multi_tensor_copy_f16",
  "abi-r1:metal:navatala_transformer_multi_tensor_copy_f16",
  "abi-r1:metal:navatala_transformer_multi_tensor_copy_f16",
  3,
  kAbiArgs_metal_navatala_transformer_multi_tensor_copy_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_multi_tensor_l2_norm_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "totalElements", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partialSums", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_multi_tensor_l2_norm_f32 = {
  1,
  "navatala_transformer_multi_tensor_l2_norm_f32",
  "metal",
  "navatala_transformer_multi_tensor_l2_norm_f32",
  "kernel:metal:navatala_transformer_multi_tensor_l2_norm_f32",
  "abi-r1:metal:navatala_transformer_multi_tensor_l2_norm_f32",
  "abi-r1:metal:navatala_transformer_multi_tensor_l2_norm_f32",
  3,
  kAbiArgs_metal_navatala_transformer_multi_tensor_l2_norm_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_multi_tensor_l2_norm_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "totalElements", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partialSums", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_multi_tensor_l2_norm_f16 = {
  1,
  "navatala_transformer_multi_tensor_l2_norm_f16",
  "metal",
  "navatala_transformer_multi_tensor_l2_norm_f16",
  "kernel:metal:navatala_transformer_multi_tensor_l2_norm_f16",
  "abi-r1:metal:navatala_transformer_multi_tensor_l2_norm_f16",
  "abi-r1:metal:navatala_transformer_multi_tensor_l2_norm_f16",
  3,
  kAbiArgs_metal_navatala_transformer_multi_tensor_l2_norm_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_multi_tensor_clip_grad_f32[] = {
  { "totalNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxNorm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "grad", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_multi_tensor_clip_grad_f32 = {
  1,
  "navatala_transformer_multi_tensor_clip_grad_f32",
  "metal",
  "navatala_transformer_multi_tensor_clip_grad_f32",
  "kernel:metal:navatala_transformer_multi_tensor_clip_grad_f32",
  "abi-r1:metal:navatala_transformer_multi_tensor_clip_grad_f32",
  "abi-r1:metal:navatala_transformer_multi_tensor_clip_grad_f32",
  4,
  kAbiArgs_metal_navatala_transformer_multi_tensor_clip_grad_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_multi_tensor_clip_grad_f16[] = {
  { "totalNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxNorm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "grad", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_multi_tensor_clip_grad_f16 = {
  1,
  "navatala_transformer_multi_tensor_clip_grad_f16",
  "metal",
  "navatala_transformer_multi_tensor_clip_grad_f16",
  "kernel:metal:navatala_transformer_multi_tensor_clip_grad_f16",
  "abi-r1:metal:navatala_transformer_multi_tensor_clip_grad_f16",
  "abi-r1:metal:navatala_transformer_multi_tensor_clip_grad_f16",
  4,
  kAbiArgs_metal_navatala_transformer_multi_tensor_clip_grad_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_top_k_gating_f32[] = {
  { "routerLogits", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numExperts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "topK", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "expertIndices", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertWeights", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_top_k_gating_f32 = {
  1,
  "navatala_transformer_top_k_gating_f32",
  "metal",
  "navatala_transformer_top_k_gating_f32",
  "kernel:metal:navatala_transformer_top_k_gating_f32",
  "abi-r1:metal:navatala_transformer_top_k_gating_f32",
  "abi-r1:metal:navatala_transformer_top_k_gating_f32",
  6,
  kAbiArgs_metal_navatala_transformer_top_k_gating_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_top_k_gating_f16[] = {
  { "routerLogits", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numExperts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "topK", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "expertIndices", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertWeights", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_top_k_gating_f16 = {
  1,
  "navatala_transformer_top_k_gating_f16",
  "metal",
  "navatala_transformer_top_k_gating_f16",
  "kernel:metal:navatala_transformer_top_k_gating_f16",
  "abi-r1:metal:navatala_transformer_top_k_gating_f16",
  "abi-r1:metal:navatala_transformer_top_k_gating_f16",
  6,
  kAbiArgs_metal_navatala_transformer_top_k_gating_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_expert_capacity_mask_f32[] = {
  { "expertIndices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numExperts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "topK", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "capacity", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "mask", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_expert_capacity_mask_f32 = {
  1,
  "navatala_transformer_expert_capacity_mask_f32",
  "metal",
  "navatala_transformer_expert_capacity_mask_f32",
  "kernel:metal:navatala_transformer_expert_capacity_mask_f32",
  "abi-r1:metal:navatala_transformer_expert_capacity_mask_f32",
  "abi-r1:metal:navatala_transformer_expert_capacity_mask_f32",
  6,
  kAbiArgs_metal_navatala_transformer_expert_capacity_mask_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_expert_capacity_mask_f16[] = {
  { "expertIndices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numExperts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "topK", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "capacity", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "mask", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_expert_capacity_mask_f16 = {
  1,
  "navatala_transformer_expert_capacity_mask_f16",
  "metal",
  "navatala_transformer_expert_capacity_mask_f16",
  "kernel:metal:navatala_transformer_expert_capacity_mask_f16",
  "abi-r1:metal:navatala_transformer_expert_capacity_mask_f16",
  "abi-r1:metal:navatala_transformer_expert_capacity_mask_f16",
  6,
  kAbiArgs_metal_navatala_transformer_expert_capacity_mask_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_permute_tokens_f32[] = {
  { "hiddenStates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertCounters", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numExperts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "permutedStates", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tokenIndices", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_permute_tokens_f32 = {
  1,
  "navatala_transformer_permute_tokens_f32",
  "metal",
  "navatala_transformer_permute_tokens_f32",
  "kernel:metal:navatala_transformer_permute_tokens_f32",
  "abi-r1:metal:navatala_transformer_permute_tokens_f32",
  "abi-r1:metal:navatala_transformer_permute_tokens_f32",
  9,
  kAbiArgs_metal_navatala_transformer_permute_tokens_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_permute_tokens_f16[] = {
  { "hiddenStates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertCounters", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numExperts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "permutedStates", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tokenIndices", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_permute_tokens_f16 = {
  1,
  "navatala_transformer_permute_tokens_f16",
  "metal",
  "navatala_transformer_permute_tokens_f16",
  "kernel:metal:navatala_transformer_permute_tokens_f16",
  "abi-r1:metal:navatala_transformer_permute_tokens_f16",
  "abi-r1:metal:navatala_transformer_permute_tokens_f16",
  9,
  kAbiArgs_metal_navatala_transformer_permute_tokens_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_unpermute_tokens_f32[] = {
  { "permutedStates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tokenIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenStates", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_unpermute_tokens_f32 = {
  1,
  "navatala_transformer_unpermute_tokens_f32",
  "metal",
  "navatala_transformer_unpermute_tokens_f32",
  "kernel:metal:navatala_transformer_unpermute_tokens_f32",
  "abi-r1:metal:navatala_transformer_unpermute_tokens_f32",
  "abi-r1:metal:navatala_transformer_unpermute_tokens_f32",
  5,
  kAbiArgs_metal_navatala_transformer_unpermute_tokens_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_unpermute_tokens_f16[] = {
  { "permutedStates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tokenIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenStates", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_unpermute_tokens_f16 = {
  1,
  "navatala_transformer_unpermute_tokens_f16",
  "metal",
  "navatala_transformer_unpermute_tokens_f16",
  "kernel:metal:navatala_transformer_unpermute_tokens_f16",
  "abi-r1:metal:navatala_transformer_unpermute_tokens_f16",
  "abi-r1:metal:navatala_transformer_unpermute_tokens_f16",
  5,
  kAbiArgs_metal_navatala_transformer_unpermute_tokens_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_pad_sequence_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_pad_sequence_f32 = {
  1,
  "navatala_transformer_pad_sequence_f32",
  "metal",
  "navatala_transformer_pad_sequence_f32",
  "kernel:metal:navatala_transformer_pad_sequence_f32",
  "abi-r1:metal:navatala_transformer_pad_sequence_f32",
  "abi-r1:metal:navatala_transformer_pad_sequence_f32",
  7,
  kAbiArgs_metal_navatala_transformer_pad_sequence_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_pad_sequence_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_pad_sequence_f16 = {
  1,
  "navatala_transformer_pad_sequence_f16",
  "metal",
  "navatala_transformer_pad_sequence_f16",
  "kernel:metal:navatala_transformer_pad_sequence_f16",
  "abi-r1:metal:navatala_transformer_pad_sequence_f16",
  "abi-r1:metal:navatala_transformer_pad_sequence_f16",
  7,
  kAbiArgs_metal_navatala_transformer_pad_sequence_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_unpad_sequence_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalTokens", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_unpad_sequence_f32 = {
  1,
  "navatala_transformer_unpad_sequence_f32",
  "metal",
  "navatala_transformer_unpad_sequence_f32",
  "kernel:metal:navatala_transformer_unpad_sequence_f32",
  "abi-r1:metal:navatala_transformer_unpad_sequence_f32",
  "abi-r1:metal:navatala_transformer_unpad_sequence_f32",
  8,
  kAbiArgs_metal_navatala_transformer_unpad_sequence_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_unpad_sequence_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalTokens", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_unpad_sequence_f16 = {
  1,
  "navatala_transformer_unpad_sequence_f16",
  "metal",
  "navatala_transformer_unpad_sequence_f16",
  "kernel:metal:navatala_transformer_unpad_sequence_f16",
  "abi-r1:metal:navatala_transformer_unpad_sequence_f16",
  "abi-r1:metal:navatala_transformer_unpad_sequence_f16",
  8,
  kAbiArgs_metal_navatala_transformer_unpad_sequence_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_swizzle_layout_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "tileSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_swizzle_layout_f32 = {
  1,
  "navatala_transformer_swizzle_layout_f32",
  "metal",
  "navatala_transformer_swizzle_layout_f32",
  "kernel:metal:navatala_transformer_swizzle_layout_f32",
  "abi-r1:metal:navatala_transformer_swizzle_layout_f32",
  "abi-r1:metal:navatala_transformer_swizzle_layout_f32",
  5,
  kAbiArgs_metal_navatala_transformer_swizzle_layout_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_swizzle_layout_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "tileSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_swizzle_layout_f16 = {
  1,
  "navatala_transformer_swizzle_layout_f16",
  "metal",
  "navatala_transformer_swizzle_layout_f16",
  "kernel:metal:navatala_transformer_swizzle_layout_f16",
  "abi-r1:metal:navatala_transformer_swizzle_layout_f16",
  "abi-r1:metal:navatala_transformer_swizzle_layout_f16",
  5,
  kAbiArgs_metal_navatala_transformer_swizzle_layout_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_hadamard_transform_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "vectorSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_hadamard_transform_f32 = {
  1,
  "navatala_transformer_hadamard_transform_f32",
  "metal",
  "navatala_transformer_hadamard_transform_f32",
  "kernel:metal:navatala_transformer_hadamard_transform_f32",
  "abi-r1:metal:navatala_transformer_hadamard_transform_f32",
  "abi-r1:metal:navatala_transformer_hadamard_transform_f32",
  4,
  kAbiArgs_metal_navatala_transformer_hadamard_transform_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_hadamard_transform_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "vectorSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_hadamard_transform_f16 = {
  1,
  "navatala_transformer_hadamard_transform_f16",
  "metal",
  "navatala_transformer_hadamard_transform_f16",
  "kernel:metal:navatala_transformer_hadamard_transform_f16",
  "abi-r1:metal:navatala_transformer_hadamard_transform_f16",
  "abi-r1:metal:navatala_transformer_hadamard_transform_f16",
  4,
  kAbiArgs_metal_navatala_transformer_hadamard_transform_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_permute_axes_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dim0", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim1", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim2", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim3", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm0", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm1", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm2", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm3", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_permute_axes_f32 = {
  1,
  "navatala_transformer_permute_axes_f32",
  "metal",
  "navatala_transformer_permute_axes_f32",
  "kernel:metal:navatala_transformer_permute_axes_f32",
  "abi-r1:metal:navatala_transformer_permute_axes_f32",
  "abi-r1:metal:navatala_transformer_permute_axes_f32",
  10,
  kAbiArgs_metal_navatala_transformer_permute_axes_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_permute_axes_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dim0", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim1", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim2", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim3", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm0", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm1", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm2", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm3", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_permute_axes_f16 = {
  1,
  "navatala_transformer_permute_axes_f16",
  "metal",
  "navatala_transformer_permute_axes_f16",
  "kernel:metal:navatala_transformer_permute_axes_f16",
  "abi-r1:metal:navatala_transformer_permute_axes_f16",
  "abi-r1:metal:navatala_transformer_permute_axes_f16",
  10,
  kAbiArgs_metal_navatala_transformer_permute_axes_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_pack_sequences_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalTokens", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_pack_sequences_f32 = {
  1,
  "navatala_transformer_pack_sequences_f32",
  "metal",
  "navatala_transformer_pack_sequences_f32",
  "kernel:metal:navatala_transformer_pack_sequences_f32",
  "abi-r1:metal:navatala_transformer_pack_sequences_f32",
  "abi-r1:metal:navatala_transformer_pack_sequences_f32",
  8,
  kAbiArgs_metal_navatala_transformer_pack_sequences_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_pack_sequences_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalTokens", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_pack_sequences_f16 = {
  1,
  "navatala_transformer_pack_sequences_f16",
  "metal",
  "navatala_transformer_pack_sequences_f16",
  "kernel:metal:navatala_transformer_pack_sequences_f16",
  "abi-r1:metal:navatala_transformer_pack_sequences_f16",
  "abi-r1:metal:navatala_transformer_pack_sequences_f16",
  8,
  kAbiArgs_metal_navatala_transformer_pack_sequences_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_unpack_sequences_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_unpack_sequences_f32 = {
  1,
  "navatala_transformer_unpack_sequences_f32",
  "metal",
  "navatala_transformer_unpack_sequences_f32",
  "kernel:metal:navatala_transformer_unpack_sequences_f32",
  "abi-r1:metal:navatala_transformer_unpack_sequences_f32",
  "abi-r1:metal:navatala_transformer_unpack_sequences_f32",
  7,
  kAbiArgs_metal_navatala_transformer_unpack_sequences_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_unpack_sequences_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_unpack_sequences_f16 = {
  1,
  "navatala_transformer_unpack_sequences_f16",
  "metal",
  "navatala_transformer_unpack_sequences_f16",
  "kernel:metal:navatala_transformer_unpack_sequences_f16",
  "abi-r1:metal:navatala_transformer_unpack_sequences_f16",
  "abi-r1:metal:navatala_transformer_unpack_sequences_f16",
  7,
  kAbiArgs_metal_navatala_transformer_unpack_sequences_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_generate_position_ids_f32[] = {
  { "batchSize", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_generate_position_ids_f32 = {
  1,
  "navatala_transformer_generate_position_ids_f32",
  "metal",
  "navatala_transformer_generate_position_ids_f32",
  "kernel:metal:navatala_transformer_generate_position_ids_f32",
  "abi-r1:metal:navatala_transformer_generate_position_ids_f32",
  "abi-r1:metal:navatala_transformer_generate_position_ids_f32",
  3,
  kAbiArgs_metal_navatala_transformer_generate_position_ids_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_generate_position_ids_u32[] = {
  { "batchSize", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_generate_position_ids_u32 = {
  1,
  "navatala_transformer_generate_position_ids_u32",
  "metal",
  "navatala_transformer_generate_position_ids_u32",
  "kernel:metal:navatala_transformer_generate_position_ids_u32",
  "abi-r1:metal:navatala_transformer_generate_position_ids_u32",
  "abi-r1:metal:navatala_transformer_generate_position_ids_u32",
  3,
  kAbiArgs_metal_navatala_transformer_generate_position_ids_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_generate_causal_mask_f32[] = {
  { "seqLen", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_generate_causal_mask_f32 = {
  1,
  "navatala_transformer_generate_causal_mask_f32",
  "metal",
  "navatala_transformer_generate_causal_mask_f32",
  "kernel:metal:navatala_transformer_generate_causal_mask_f32",
  "abi-r1:metal:navatala_transformer_generate_causal_mask_f32",
  "abi-r1:metal:navatala_transformer_generate_causal_mask_f32",
  2,
  kAbiArgs_metal_navatala_transformer_generate_causal_mask_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_generate_causal_mask_f16[] = {
  { "seqLen", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_generate_causal_mask_f16 = {
  1,
  "navatala_transformer_generate_causal_mask_f16",
  "metal",
  "navatala_transformer_generate_causal_mask_f16",
  "kernel:metal:navatala_transformer_generate_causal_mask_f16",
  "abi-r1:metal:navatala_transformer_generate_causal_mask_f16",
  "abi-r1:metal:navatala_transformer_generate_causal_mask_f16",
  2,
  kAbiArgs_metal_navatala_transformer_generate_causal_mask_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_precompute_rotary_emb_f32[] = {
  { "maxSeqLen", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cosOutput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sinOutput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_precompute_rotary_emb_f32 = {
  1,
  "navatala_transformer_precompute_rotary_emb_f32",
  "metal",
  "navatala_transformer_precompute_rotary_emb_f32",
  "kernel:metal:navatala_transformer_precompute_rotary_emb_f32",
  "abi-r1:metal:navatala_transformer_precompute_rotary_emb_f32",
  "abi-r1:metal:navatala_transformer_precompute_rotary_emb_f32",
  5,
  kAbiArgs_metal_navatala_transformer_precompute_rotary_emb_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_transformer_precompute_rotary_emb_f16[] = {
  { "maxSeqLen", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cosOutput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sinOutput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_transformer_precompute_rotary_emb_f16 = {
  1,
  "navatala_transformer_precompute_rotary_emb_f16",
  "metal",
  "navatala_transformer_precompute_rotary_emb_f16",
  "kernel:metal:navatala_transformer_precompute_rotary_emb_f16",
  "abi-r1:metal:navatala_transformer_precompute_rotary_emb_f16",
  "abi-r1:metal:navatala_transformer_precompute_rotary_emb_f16",
  5,
  kAbiArgs_metal_navatala_transformer_precompute_rotary_emb_f16
};

bool tryGetKernelAbiManifest_metal_transformer(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_transformer_gelu_f32") {
    out = &kAbiManifest_metal_navatala_transformer_gelu_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_gelu_tanh_f32") {
    out = &kAbiManifest_metal_navatala_transformer_gelu_tanh_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_silu_f32") {
    out = &kAbiManifest_metal_navatala_transformer_silu_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_relu_f32") {
    out = &kAbiManifest_metal_navatala_transformer_relu_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_quick_gelu_f32") {
    out = &kAbiManifest_metal_navatala_transformer_quick_gelu_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_squared_relu_f32") {
    out = &kAbiManifest_metal_navatala_transformer_squared_relu_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_bias_gelu_f32") {
    out = &kAbiManifest_metal_navatala_transformer_bias_gelu_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_gelu_f16") {
    out = &kAbiManifest_metal_navatala_transformer_gelu_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_silu_f16") {
    out = &kAbiManifest_metal_navatala_transformer_silu_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_relu_f16") {
    out = &kAbiManifest_metal_navatala_transformer_relu_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_quick_gelu_f16") {
    out = &kAbiManifest_metal_navatala_transformer_quick_gelu_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_squared_relu_f16") {
    out = &kAbiManifest_metal_navatala_transformer_squared_relu_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_geglu_f32") {
    out = &kAbiManifest_metal_navatala_transformer_geglu_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_swiglu_f32") {
    out = &kAbiManifest_metal_navatala_transformer_swiglu_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_reglu_f32") {
    out = &kAbiManifest_metal_navatala_transformer_reglu_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_geglu_f16") {
    out = &kAbiManifest_metal_navatala_transformer_geglu_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_swiglu_f16") {
    out = &kAbiManifest_metal_navatala_transformer_swiglu_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_reglu_f16") {
    out = &kAbiManifest_metal_navatala_transformer_reglu_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_gelu_backward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_gelu_backward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_silu_backward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_silu_backward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_relu_backward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_relu_backward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_quick_gelu_backward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_quick_gelu_backward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_gelu_backward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_gelu_backward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_silu_backward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_silu_backward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_relu_backward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_relu_backward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_quick_gelu_backward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_quick_gelu_backward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rms_norm_forward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_rms_norm_forward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rms_norm_forward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_rms_norm_forward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_layer_norm_forward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_layer_norm_forward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_layer_norm_forward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_layer_norm_forward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_layer_norm_backward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_layer_norm_backward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_layer_norm_backward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_layer_norm_backward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rms_norm_backward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_rms_norm_backward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rms_norm_backward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_rms_norm_backward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_layer_norm_multi_pass_f32") {
    out = &kAbiManifest_metal_navatala_transformer_layer_norm_multi_pass_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_layer_norm_multi_pass_f16") {
    out = &kAbiManifest_metal_navatala_transformer_layer_norm_multi_pass_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rms_norm_multi_pass_f32") {
    out = &kAbiManifest_metal_navatala_transformer_rms_norm_multi_pass_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rms_norm_multi_pass_f16") {
    out = &kAbiManifest_metal_navatala_transformer_rms_norm_multi_pass_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_forward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_softmax_forward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_forward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_softmax_forward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_backward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_softmax_backward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_backward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_softmax_backward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_with_mask_f32") {
    out = &kAbiManifest_metal_navatala_transformer_softmax_with_mask_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_with_mask_f16") {
    out = &kAbiManifest_metal_navatala_transformer_softmax_with_mask_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_with_mask_backward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_softmax_with_mask_backward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_with_mask_backward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_softmax_with_mask_backward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_multi_pass_f32") {
    out = &kAbiManifest_metal_navatala_transformer_softmax_multi_pass_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_multi_pass_f16") {
    out = &kAbiManifest_metal_navatala_transformer_softmax_multi_pass_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_cast_f32_to_f16") {
    out = &kAbiManifest_metal_navatala_transformer_cast_f32_to_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_cast_f16_to_f32") {
    out = &kAbiManifest_metal_navatala_transformer_cast_f16_to_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_transpose_f32") {
    out = &kAbiManifest_metal_navatala_transformer_transpose_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_transpose_f16") {
    out = &kAbiManifest_metal_navatala_transformer_transpose_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_fused_cast_transpose_f32_to_f16") {
    out = &kAbiManifest_metal_navatala_transformer_fused_cast_transpose_f32_to_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_cast_f32_to_f_p8_e4_m3") {
    out = &kAbiManifest_metal_navatala_transformer_cast_f32_to_f_p8_e4_m3;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_cast_f_p8_e4_m3_to_f32") {
    out = &kAbiManifest_metal_navatala_transformer_cast_f_p8_e4_m3_to_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_cast_f32_to_f_p8_e5_m2") {
    out = &kAbiManifest_metal_navatala_transformer_cast_f32_to_f_p8_e5_m2;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_cast_f_p8_e5_m2_to_f32") {
    out = &kAbiManifest_metal_navatala_transformer_cast_f_p8_e5_m2_to_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_fused_cast_transpose_f_p8") {
    out = &kAbiManifest_metal_navatala_transformer_fused_cast_transpose_f_p8;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_transpose4_d_f32") {
    out = &kAbiManifest_metal_navatala_transformer_transpose4_d_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_transpose4_d_f16") {
    out = &kAbiManifest_metal_navatala_transformer_transpose4_d_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_dropout_forward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_dropout_forward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_dropout_forward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_dropout_forward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_dropout_backward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_dropout_backward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_dropout_backward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_dropout_backward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rope_forward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_rope_forward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rope_forward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_rope_forward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rope_with_cache_f32") {
    out = &kAbiManifest_metal_navatala_transformer_rope_with_cache_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rope_backward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_rope_backward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rope_backward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_rope_backward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rope_interleaved_f32") {
    out = &kAbiManifest_metal_navatala_transformer_rope_interleaved_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rope_interleaved_f16") {
    out = &kAbiManifest_metal_navatala_transformer_rope_interleaved_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_simple_attention_f16") {
    out = &kAbiManifest_metal_navatala_transformer_simple_attention_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_simple_attention_f32") {
    out = &kAbiManifest_metal_navatala_transformer_simple_attention_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_paged_attention_f32") {
    out = &kAbiManifest_metal_navatala_transformer_paged_attention_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_simple_attention_with_mask_f32") {
    out = &kAbiManifest_metal_navatala_transformer_simple_attention_with_mask_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_simple_attention_with_mask_f16") {
    out = &kAbiManifest_metal_navatala_transformer_simple_attention_with_mask_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_simple_attention_with_padding_f32") {
    out = &kAbiManifest_metal_navatala_transformer_simple_attention_with_padding_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_simple_attention_with_padding_f16") {
    out = &kAbiManifest_metal_navatala_transformer_simple_attention_with_padding_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_simple_attention_backward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_simple_attention_backward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_simple_attention_backward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_simple_attention_backward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_kv_cache_append_f16") {
    out = &kAbiManifest_metal_navatala_transformer_kv_cache_append_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_kv_cache_append_f32") {
    out = &kAbiManifest_metal_navatala_transformer_kv_cache_append_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_kv_cache_rotate_f16") {
    out = &kAbiManifest_metal_navatala_transformer_kv_cache_rotate_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_f16") {
    out = &kAbiManifest_metal_navatala_transformer_tiled_gemm_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out = &kAbiManifest_metal_navatala_transformer_tiled_gemm_f16_f32_out;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_f32") {
    out = &kAbiManifest_metal_navatala_transformer_tiled_gemm_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_backward_f32") {
    out = &kAbiManifest_metal_navatala_transformer_tiled_gemm_backward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_backward_f16") {
    out = &kAbiManifest_metal_navatala_transformer_tiled_gemm_backward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_scale_f32") {
    out = &kAbiManifest_metal_navatala_transformer_multi_tensor_scale_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_scale_f16") {
    out = &kAbiManifest_metal_navatala_transformer_multi_tensor_scale_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_add_f32") {
    out = &kAbiManifest_metal_navatala_transformer_multi_tensor_add_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_add_f16") {
    out = &kAbiManifest_metal_navatala_transformer_multi_tensor_add_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_copy_f32") {
    out = &kAbiManifest_metal_navatala_transformer_multi_tensor_copy_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_copy_f16") {
    out = &kAbiManifest_metal_navatala_transformer_multi_tensor_copy_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_l2_norm_f32") {
    out = &kAbiManifest_metal_navatala_transformer_multi_tensor_l2_norm_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_l2_norm_f16") {
    out = &kAbiManifest_metal_navatala_transformer_multi_tensor_l2_norm_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_clip_grad_f32") {
    out = &kAbiManifest_metal_navatala_transformer_multi_tensor_clip_grad_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_clip_grad_f16") {
    out = &kAbiManifest_metal_navatala_transformer_multi_tensor_clip_grad_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_top_k_gating_f32") {
    out = &kAbiManifest_metal_navatala_transformer_top_k_gating_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_top_k_gating_f16") {
    out = &kAbiManifest_metal_navatala_transformer_top_k_gating_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_expert_capacity_mask_f32") {
    out = &kAbiManifest_metal_navatala_transformer_expert_capacity_mask_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_expert_capacity_mask_f16") {
    out = &kAbiManifest_metal_navatala_transformer_expert_capacity_mask_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_permute_tokens_f32") {
    out = &kAbiManifest_metal_navatala_transformer_permute_tokens_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_permute_tokens_f16") {
    out = &kAbiManifest_metal_navatala_transformer_permute_tokens_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_unpermute_tokens_f32") {
    out = &kAbiManifest_metal_navatala_transformer_unpermute_tokens_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_unpermute_tokens_f16") {
    out = &kAbiManifest_metal_navatala_transformer_unpermute_tokens_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_pad_sequence_f32") {
    out = &kAbiManifest_metal_navatala_transformer_pad_sequence_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_pad_sequence_f16") {
    out = &kAbiManifest_metal_navatala_transformer_pad_sequence_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_unpad_sequence_f32") {
    out = &kAbiManifest_metal_navatala_transformer_unpad_sequence_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_unpad_sequence_f16") {
    out = &kAbiManifest_metal_navatala_transformer_unpad_sequence_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_swizzle_layout_f32") {
    out = &kAbiManifest_metal_navatala_transformer_swizzle_layout_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_swizzle_layout_f16") {
    out = &kAbiManifest_metal_navatala_transformer_swizzle_layout_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_hadamard_transform_f32") {
    out = &kAbiManifest_metal_navatala_transformer_hadamard_transform_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_hadamard_transform_f16") {
    out = &kAbiManifest_metal_navatala_transformer_hadamard_transform_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_permute_axes_f32") {
    out = &kAbiManifest_metal_navatala_transformer_permute_axes_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_permute_axes_f16") {
    out = &kAbiManifest_metal_navatala_transformer_permute_axes_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_pack_sequences_f32") {
    out = &kAbiManifest_metal_navatala_transformer_pack_sequences_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_pack_sequences_f16") {
    out = &kAbiManifest_metal_navatala_transformer_pack_sequences_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_unpack_sequences_f32") {
    out = &kAbiManifest_metal_navatala_transformer_unpack_sequences_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_unpack_sequences_f16") {
    out = &kAbiManifest_metal_navatala_transformer_unpack_sequences_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_generate_position_ids_f32") {
    out = &kAbiManifest_metal_navatala_transformer_generate_position_ids_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_generate_position_ids_u32") {
    out = &kAbiManifest_metal_navatala_transformer_generate_position_ids_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_generate_causal_mask_f32") {
    out = &kAbiManifest_metal_navatala_transformer_generate_causal_mask_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_generate_causal_mask_f16") {
    out = &kAbiManifest_metal_navatala_transformer_generate_causal_mask_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_precompute_rotary_emb_f32") {
    out = &kAbiManifest_metal_navatala_transformer_precompute_rotary_emb_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_precompute_rotary_emb_f16") {
    out = &kAbiManifest_metal_navatala_transformer_precompute_rotary_emb_f16;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_transformer(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_transformer_gelu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_gelu_f32";
    std::string_view sv(k_metal_navatala_transformer_gelu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_gelu_tanh_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_gelu_tanh_f32";
    std::string_view sv(k_metal_navatala_transformer_gelu_tanh_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_silu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_silu_f32";
    std::string_view sv(k_metal_navatala_transformer_silu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_relu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_relu_f32";
    std::string_view sv(k_metal_navatala_transformer_relu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_quick_gelu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_quick_gelu_f32";
    std::string_view sv(k_metal_navatala_transformer_quick_gelu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_squared_relu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_squared_relu_f32";
    std::string_view sv(k_metal_navatala_transformer_squared_relu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_bias_gelu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_bias_gelu_f32";
    std::string_view sv(k_metal_navatala_transformer_bias_gelu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_gelu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_gelu_f16";
    std::string_view sv(k_metal_navatala_transformer_gelu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_silu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_silu_f16";
    std::string_view sv(k_metal_navatala_transformer_silu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_relu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_relu_f16";
    std::string_view sv(k_metal_navatala_transformer_relu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_quick_gelu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_quick_gelu_f16";
    std::string_view sv(k_metal_navatala_transformer_quick_gelu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_squared_relu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_squared_relu_f16";
    std::string_view sv(k_metal_navatala_transformer_squared_relu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_geglu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_geglu_f32";
    std::string_view sv(k_metal_navatala_transformer_geglu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_swiglu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_swiglu_f32";
    std::string_view sv(k_metal_navatala_transformer_swiglu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_reglu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_reglu_f32";
    std::string_view sv(k_metal_navatala_transformer_reglu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_geglu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_geglu_f16";
    std::string_view sv(k_metal_navatala_transformer_geglu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_swiglu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_swiglu_f16";
    std::string_view sv(k_metal_navatala_transformer_swiglu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_reglu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_reglu_f16";
    std::string_view sv(k_metal_navatala_transformer_reglu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_gelu_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_gelu_backward_f32";
    std::string_view sv(k_metal_navatala_transformer_gelu_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_silu_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_silu_backward_f32";
    std::string_view sv(k_metal_navatala_transformer_silu_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_relu_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_relu_backward_f32";
    std::string_view sv(k_metal_navatala_transformer_relu_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_quick_gelu_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_quick_gelu_backward_f32";
    std::string_view sv(k_metal_navatala_transformer_quick_gelu_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_gelu_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_gelu_backward_f16";
    std::string_view sv(k_metal_navatala_transformer_gelu_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_silu_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_silu_backward_f16";
    std::string_view sv(k_metal_navatala_transformer_silu_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_relu_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_relu_backward_f16";
    std::string_view sv(k_metal_navatala_transformer_relu_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_quick_gelu_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_quick_gelu_backward_f16";
    std::string_view sv(k_metal_navatala_transformer_quick_gelu_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rms_norm_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_rms_norm_forward_f32";
    std::string_view sv(k_metal_navatala_transformer_rms_norm_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rms_norm_forward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_rms_norm_forward_f16";
    std::string_view sv(k_metal_navatala_transformer_rms_norm_forward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_layer_norm_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_layer_norm_forward_f32";
    std::string_view sv(k_metal_navatala_transformer_layer_norm_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_layer_norm_forward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_layer_norm_forward_f16";
    std::string_view sv(k_metal_navatala_transformer_layer_norm_forward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_layer_norm_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_layer_norm_backward_f32";
    std::string_view sv(k_metal_navatala_transformer_layer_norm_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_layer_norm_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_layer_norm_backward_f16";
    std::string_view sv(k_metal_navatala_transformer_layer_norm_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rms_norm_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_rms_norm_backward_f32";
    std::string_view sv(k_metal_navatala_transformer_rms_norm_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rms_norm_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_rms_norm_backward_f16";
    std::string_view sv(k_metal_navatala_transformer_rms_norm_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_layer_norm_multi_pass_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_layer_norm_multi_pass_f32";
    std::string_view sv(k_metal_navatala_transformer_layer_norm_multi_pass_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_layer_norm_multi_pass_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_layer_norm_multi_pass_f16";
    std::string_view sv(k_metal_navatala_transformer_layer_norm_multi_pass_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rms_norm_multi_pass_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_rms_norm_multi_pass_f32";
    std::string_view sv(k_metal_navatala_transformer_rms_norm_multi_pass_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rms_norm_multi_pass_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_rms_norm_multi_pass_f16";
    std::string_view sv(k_metal_navatala_transformer_rms_norm_multi_pass_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_softmax_forward_f32";
    std::string_view sv(k_metal_navatala_transformer_softmax_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_forward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_softmax_forward_f16";
    std::string_view sv(k_metal_navatala_transformer_softmax_forward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_softmax_backward_f32";
    std::string_view sv(k_metal_navatala_transformer_softmax_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_softmax_backward_f16";
    std::string_view sv(k_metal_navatala_transformer_softmax_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_with_mask_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_softmax_with_mask_f32";
    std::string_view sv(k_metal_navatala_transformer_softmax_with_mask_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_with_mask_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_softmax_with_mask_f16";
    std::string_view sv(k_metal_navatala_transformer_softmax_with_mask_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_with_mask_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_softmax_with_mask_backward_f32";
    std::string_view sv(k_metal_navatala_transformer_softmax_with_mask_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_with_mask_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_softmax_with_mask_backward_f16";
    std::string_view sv(k_metal_navatala_transformer_softmax_with_mask_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_multi_pass_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_softmax_multi_pass_f32";
    std::string_view sv(k_metal_navatala_transformer_softmax_multi_pass_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_softmax_multi_pass_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_softmax_multi_pass_f16";
    std::string_view sv(k_metal_navatala_transformer_softmax_multi_pass_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_cast_f32_to_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_cast_f32_to_f16";
    std::string_view sv(k_metal_navatala_transformer_cast_f32_to_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_cast_f16_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_cast_f16_to_f32";
    std::string_view sv(k_metal_navatala_transformer_cast_f16_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_transpose_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_transpose_f32";
    std::string_view sv(k_metal_navatala_transformer_transpose_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_transpose_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_transpose_f16";
    std::string_view sv(k_metal_navatala_transformer_transpose_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_fused_cast_transpose_f32_to_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_fused_cast_transpose_f32_to_f16";
    std::string_view sv(k_metal_navatala_transformer_fused_cast_transpose_f32_to_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_cast_f32_to_f_p8_e4_m3") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_cast_f32_to_f_p8_e4_m3";
    std::string_view sv(k_metal_navatala_transformer_cast_f32_to_f_p8_e4_m3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_cast_f_p8_e4_m3_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_cast_f_p8_e4_m3_to_f32";
    std::string_view sv(k_metal_navatala_transformer_cast_f_p8_e4_m3_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_cast_f32_to_f_p8_e5_m2") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_cast_f32_to_f_p8_e5_m2";
    std::string_view sv(k_metal_navatala_transformer_cast_f32_to_f_p8_e5_m2);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_cast_f_p8_e5_m2_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_cast_f_p8_e5_m2_to_f32";
    std::string_view sv(k_metal_navatala_transformer_cast_f_p8_e5_m2_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_fused_cast_transpose_f_p8") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_fused_cast_transpose_f_p8";
    std::string_view sv(k_metal_navatala_transformer_fused_cast_transpose_f_p8);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_transpose4_d_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_transpose4_d_f32";
    std::string_view sv(k_metal_navatala_transformer_transpose4_d_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_transpose4_d_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_transpose4_d_f16";
    std::string_view sv(k_metal_navatala_transformer_transpose4_d_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_dropout_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_dropout_forward_f32";
    std::string_view sv(k_metal_navatala_transformer_dropout_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_dropout_forward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_dropout_forward_f16";
    std::string_view sv(k_metal_navatala_transformer_dropout_forward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_dropout_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_dropout_backward_f32";
    std::string_view sv(k_metal_navatala_transformer_dropout_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_dropout_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_dropout_backward_f16";
    std::string_view sv(k_metal_navatala_transformer_dropout_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rope_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_rope_forward_f32";
    std::string_view sv(k_metal_navatala_transformer_rope_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rope_forward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_rope_forward_f16";
    std::string_view sv(k_metal_navatala_transformer_rope_forward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rope_with_cache_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_rope_with_cache_f32";
    std::string_view sv(k_metal_navatala_transformer_rope_with_cache_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rope_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_rope_backward_f32";
    std::string_view sv(k_metal_navatala_transformer_rope_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rope_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_rope_backward_f16";
    std::string_view sv(k_metal_navatala_transformer_rope_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rope_interleaved_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_rope_interleaved_f32";
    std::string_view sv(k_metal_navatala_transformer_rope_interleaved_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_rope_interleaved_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_rope_interleaved_f16";
    std::string_view sv(k_metal_navatala_transformer_rope_interleaved_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_simple_attention_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_simple_attention_f16";
    std::string_view sv(k_metal_navatala_transformer_simple_attention_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_simple_attention_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_simple_attention_f32";
    std::string_view sv(k_metal_navatala_transformer_simple_attention_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_paged_attention_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_paged_attention_f32";
    std::string_view sv(k_metal_navatala_transformer_paged_attention_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_simple_attention_with_mask_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_simple_attention_with_mask_f32";
    std::string_view sv(k_metal_navatala_transformer_simple_attention_with_mask_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_simple_attention_with_mask_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_simple_attention_with_mask_f16";
    std::string_view sv(k_metal_navatala_transformer_simple_attention_with_mask_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_simple_attention_with_padding_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_simple_attention_with_padding_f32";
    std::string_view sv(k_metal_navatala_transformer_simple_attention_with_padding_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_simple_attention_with_padding_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_simple_attention_with_padding_f16";
    std::string_view sv(k_metal_navatala_transformer_simple_attention_with_padding_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_simple_attention_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_simple_attention_backward_f32";
    std::string_view sv(k_metal_navatala_transformer_simple_attention_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_simple_attention_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_simple_attention_backward_f16";
    std::string_view sv(k_metal_navatala_transformer_simple_attention_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_kv_cache_append_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_kv_cache_append_f16";
    std::string_view sv(k_metal_navatala_transformer_kv_cache_append_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_kv_cache_append_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_kv_cache_append_f32";
    std::string_view sv(k_metal_navatala_transformer_kv_cache_append_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_kv_cache_rotate_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_kv_cache_rotate_f16";
    std::string_view sv(k_metal_navatala_transformer_kv_cache_rotate_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16";
    std::string_view sv(k_metal_navatala_transformer_tiled_gemm_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_metal_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_tiled_gemm_f32";
    std::string_view sv(k_metal_navatala_transformer_tiled_gemm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_tiled_gemm_backward_f32";
    std::string_view sv(k_metal_navatala_transformer_tiled_gemm_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_tiled_gemm_backward_f16";
    std::string_view sv(k_metal_navatala_transformer_tiled_gemm_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_scale_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_multi_tensor_scale_f32";
    std::string_view sv(k_metal_navatala_transformer_multi_tensor_scale_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_scale_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_multi_tensor_scale_f16";
    std::string_view sv(k_metal_navatala_transformer_multi_tensor_scale_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_add_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_multi_tensor_add_f32";
    std::string_view sv(k_metal_navatala_transformer_multi_tensor_add_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_add_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_multi_tensor_add_f16";
    std::string_view sv(k_metal_navatala_transformer_multi_tensor_add_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_copy_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_multi_tensor_copy_f32";
    std::string_view sv(k_metal_navatala_transformer_multi_tensor_copy_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_copy_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_multi_tensor_copy_f16";
    std::string_view sv(k_metal_navatala_transformer_multi_tensor_copy_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_l2_norm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_multi_tensor_l2_norm_f32";
    std::string_view sv(k_metal_navatala_transformer_multi_tensor_l2_norm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_l2_norm_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_multi_tensor_l2_norm_f16";
    std::string_view sv(k_metal_navatala_transformer_multi_tensor_l2_norm_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_clip_grad_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_multi_tensor_clip_grad_f32";
    std::string_view sv(k_metal_navatala_transformer_multi_tensor_clip_grad_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_multi_tensor_clip_grad_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_multi_tensor_clip_grad_f16";
    std::string_view sv(k_metal_navatala_transformer_multi_tensor_clip_grad_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_top_k_gating_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_top_k_gating_f32";
    std::string_view sv(k_metal_navatala_transformer_top_k_gating_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_top_k_gating_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_top_k_gating_f16";
    std::string_view sv(k_metal_navatala_transformer_top_k_gating_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_expert_capacity_mask_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_expert_capacity_mask_f32";
    std::string_view sv(k_metal_navatala_transformer_expert_capacity_mask_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_expert_capacity_mask_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_expert_capacity_mask_f16";
    std::string_view sv(k_metal_navatala_transformer_expert_capacity_mask_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_permute_tokens_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_permute_tokens_f32";
    std::string_view sv(k_metal_navatala_transformer_permute_tokens_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_permute_tokens_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_permute_tokens_f16";
    std::string_view sv(k_metal_navatala_transformer_permute_tokens_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_unpermute_tokens_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_unpermute_tokens_f32";
    std::string_view sv(k_metal_navatala_transformer_unpermute_tokens_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_unpermute_tokens_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_unpermute_tokens_f16";
    std::string_view sv(k_metal_navatala_transformer_unpermute_tokens_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_pad_sequence_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_pad_sequence_f32";
    std::string_view sv(k_metal_navatala_transformer_pad_sequence_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_pad_sequence_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_pad_sequence_f16";
    std::string_view sv(k_metal_navatala_transformer_pad_sequence_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_unpad_sequence_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_unpad_sequence_f32";
    std::string_view sv(k_metal_navatala_transformer_unpad_sequence_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_unpad_sequence_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_unpad_sequence_f16";
    std::string_view sv(k_metal_navatala_transformer_unpad_sequence_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_swizzle_layout_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_swizzle_layout_f32";
    std::string_view sv(k_metal_navatala_transformer_swizzle_layout_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_swizzle_layout_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_swizzle_layout_f16";
    std::string_view sv(k_metal_navatala_transformer_swizzle_layout_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_hadamard_transform_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_hadamard_transform_f32";
    std::string_view sv(k_metal_navatala_transformer_hadamard_transform_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_hadamard_transform_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_hadamard_transform_f16";
    std::string_view sv(k_metal_navatala_transformer_hadamard_transform_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_permute_axes_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_permute_axes_f32";
    std::string_view sv(k_metal_navatala_transformer_permute_axes_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_permute_axes_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_permute_axes_f16";
    std::string_view sv(k_metal_navatala_transformer_permute_axes_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_pack_sequences_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_pack_sequences_f32";
    std::string_view sv(k_metal_navatala_transformer_pack_sequences_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_pack_sequences_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_pack_sequences_f16";
    std::string_view sv(k_metal_navatala_transformer_pack_sequences_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_unpack_sequences_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_unpack_sequences_f32";
    std::string_view sv(k_metal_navatala_transformer_unpack_sequences_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_unpack_sequences_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_unpack_sequences_f16";
    std::string_view sv(k_metal_navatala_transformer_unpack_sequences_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_generate_position_ids_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_generate_position_ids_f32";
    std::string_view sv(k_metal_navatala_transformer_generate_position_ids_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_generate_position_ids_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_generate_position_ids_u32";
    std::string_view sv(k_metal_navatala_transformer_generate_position_ids_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_generate_causal_mask_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_generate_causal_mask_f32";
    std::string_view sv(k_metal_navatala_transformer_generate_causal_mask_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_generate_causal_mask_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_generate_causal_mask_f16";
    std::string_view sv(k_metal_navatala_transformer_generate_causal_mask_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_precompute_rotary_emb_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_precompute_rotary_emb_f32";
    std::string_view sv(k_metal_navatala_transformer_precompute_rotary_emb_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_precompute_rotary_emb_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_precompute_rotary_emb_f16";
    std::string_view sv(k_metal_navatala_transformer_precompute_rotary_emb_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16";
    std::string_view sv(k_metal_navatala_transformer_tiled_gemm_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16";
    std::string_view sv(k_metal_navatala_transformer_tiled_gemm_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_metal_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_metal_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_metal_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_metal_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_metal_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_metal_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_metal_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_metal_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

