// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_ml` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_ml_init_ema_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_init_ema_f32(device const float* alphaIn [[buffer(0)]], device float* ema [[buffer(1)]], device float* alpha [[buffer(2)]], device uint* count [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    float a = alphaIn[0];
    ema[0] = as_type<float>(0x00000000u);
    alpha[0] = a;
    count[0] = 0u;
  }
}

)kernel";
const char* k_metal_navatala_ml_init_ema_from_span_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_init_ema_from_span_f32(device const float* span [[buffer(0)]], device float* ema [[buffer(1)]], device float* alpha [[buffer(2)]], device uint* count [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    float s = span[0];
    float spanPlus1 = (s + as_type<float>(0x3f800000u));
    float a = (as_type<float>(0x40000000u) / spanPlus1);
    ema[0] = as_type<float>(0x00000000u);
    alpha[0] = a;
    count[0] = 0u;
  }
}

)kernel";
const char* k_metal_navatala_ml_ema_update_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_ema_update_f32(device const float* x [[buffer(0)]], device const float* alpha [[buffer(1)]], device float* ema [[buffer(2)]], device uint* count [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    float xVal = x[0];
    float a = alpha[0];
    float oldEma = ema[0];
    uint n = count[0];
    float oneMinusA = (as_type<float>(0x3f800000u) - a);
    float alphaX = (a * xVal);
    float omAEma = (oneMinusA * oldEma);
    float emaUpd = (alphaX + omAEma);
    float newEma = (((n == 0u)) ? (xVal) : (emaUpd));
    ema[0] = newEma;
    uint newCnt = (n + 1u);
    count[0] = newCnt;
  }
}

)kernel";
const char* k_metal_navatala_ml_ema_extract_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_ema_extract_f32(device const float* ema [[buffer(0)]], device float* result [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    float v = ema[0];
    result[0] = v;
  }
}

)kernel";
const char* k_metal_navatala_ml_ema_reset_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_ema_reset_f32(device float* ema [[buffer(0)]], device uint* count [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    ema[0] = as_type<float>(0x00000000u);
    count[0] = 0u;
  }
}

)kernel";
const char* k_metal_navatala_ml_init_dema_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_init_dema_f32(device const float* alphaIn [[buffer(0)]], device float* ema1 [[buffer(1)]], device float* ema2 [[buffer(2)]], device float* alpha [[buffer(3)]], device uint* count [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    float a = alphaIn[0];
    ema1[0] = as_type<float>(0x00000000u);
    ema2[0] = as_type<float>(0x00000000u);
    alpha[0] = a;
    count[0] = 0u;
  }
}

)kernel";
const char* k_metal_navatala_ml_dema_update_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_dema_update_f32(device const float* x [[buffer(0)]], device const float* alpha [[buffer(1)]], device float* ema1 [[buffer(2)]], device float* ema2 [[buffer(3)]], device uint* count [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    float xVal = x[0];
    float a = alpha[0];
    float old1 = ema1[0];
    float old2 = ema2[0];
    uint n = count[0];
    float omA = (as_type<float>(0x3f800000u) - a);
    float aX = (a * xVal);
    float omA1 = (omA * old1);
    float e1u = (aX + omA1);
    float new1 = (((n == 0u)) ? (xVal) : (e1u));
    float aE1 = (a * new1);
    float omA2 = (omA * old2);
    float e2u = (aE1 + omA2);
    float new2 = (((n == 0u)) ? (new1) : (e2u));
    ema1[0] = new1;
    ema2[0] = new2;
    uint nc = (n + 1u);
    count[0] = nc;
  }
}

)kernel";
const char* k_metal_navatala_ml_dema_extract_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_dema_extract_f32(device const float* ema1 [[buffer(0)]], device const float* ema2 [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    float e1 = ema1[0];
    float e2 = ema2[0];
    float t2 = (as_type<float>(0x40000000u) * e1);
    float d = (t2 - e2);
    result[0] = d;
  }
}

)kernel";
const char* k_metal_navatala_ml_maxpool_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_maxpool_f32(device const float* _input [[buffer(0)]], device const uint* window [[buffer(1)]], device const uint* stride [[buffer(2)]], device const uint* inCount [[buffer(3)]], device const uint* outCount [[buffer(4)]], device float* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint outCountVal = outCount[0];
  if (gid < outCountVal) {
    uint windowVal = window[0];
    uint strideVal = stride[0];
    uint inCountVal = inCount[0];
    uint base = (gid * strideVal);
    float acc = as_type<float>(0xff7fc99eu);
    for (int w = 0; w < (int)(windowVal); ++w) {
      uint idx = (base + ((uint)(w)));
      if (idx < inCountVal) {
        float v = _input[idx];
        acc = (((acc > v)) ? (acc) : (v));
      }
    }
    _output[gid] = acc;
  }
}

)kernel";
const char* k_metal_navatala_ml_avgpool_exclude_pad_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_avgpool_exclude_pad_f32(device const float* _input [[buffer(0)]], device const uint* window [[buffer(1)]], device const uint* stride [[buffer(2)]], device const uint* inCount [[buffer(3)]], device const uint* outCount [[buffer(4)]], device float* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint outCountVal = outCount[0];
  if (gid < outCountVal) {
    uint windowVal = window[0];
    uint strideVal = stride[0];
    uint inCountVal = inCount[0];
    uint base = (gid * strideVal);
    float sum = as_type<float>(0x00000000u);
    float cnt = as_type<float>(0x00000000u);
    for (int w = 0; w < (int)(windowVal); ++w) {
      uint idx = (base + ((uint)(w)));
      if (idx < inCountVal) {
        float v = _input[idx];
        sum = (sum + v);
        cnt = (cnt + as_type<float>(0x3f800000u));
      }
    }
    _output[gid] = (sum / (((cnt > as_type<float>(0x00000000u))) ? (cnt) : (as_type<float>(0x3f800000u))));
  }
}

)kernel";
const char* k_metal_navatala_ml_avgpool_include_pad_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_avgpool_include_pad_f32(device const float* _input [[buffer(0)]], device const uint* window [[buffer(1)]], device const uint* stride [[buffer(2)]], device const uint* inCount [[buffer(3)]], device const uint* outCount [[buffer(4)]], device float* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint outCountVal = outCount[0];
  if (gid < outCountVal) {
    uint windowVal = window[0];
    uint strideVal = stride[0];
    uint inCountVal = inCount[0];
    uint base = (gid * strideVal);
    float sum = as_type<float>(0x00000000u);
    float cnt = as_type<float>(0x00000000u);
    for (int w = 0; w < (int)(windowVal); ++w) {
      uint idx = (base + ((uint)(w)));
      if (idx < inCountVal) {
        float v = _input[idx];
        sum = (sum + v);
        cnt = (cnt + as_type<float>(0x3f800000u));
      }
    }
    _output[gid] = (sum / ((float)(windowVal)));
  }
}

)kernel";
const char* k_metal_navatala_ml_resize_nearest_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_resize_nearest_f32(device const float* _input [[buffer(0)]], device const uint* inSize [[buffer(1)]], device const uint* outSize [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint outSizeVal = outSize[0];
  if (gid < outSizeVal) {
    uint inSizeVal = inSize[0];
    uint idx = ((gid * inSizeVal) / outSizeVal);
    float v = _input[idx];
    _output[gid] = v;
  }
}

)kernel";
const char* k_metal_navatala_ml_resize_bilinear_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_resize_bilinear_f32(device const float* _input [[buffer(0)]], device const uint* inSize [[buffer(1)]], device const uint* outSize [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint outSizeVal = outSize[0];
  if (gid < outSizeVal) {
    uint inSizeVal = inSize[0];
    uint inM1 = (inSizeVal - 1u);
    uint outM1 = (outSizeVal - 1u);
    float pos = ((((float)(gid)) * ((float)(inM1))) / ((float)(outM1)));
    float i0f = floor(pos);
    uint i0 = ((uint)(i0f));
    float frac = (pos - i0f);
    uint i0p1 = (i0 + 1u);
    uint i1 = (((i0p1 < inM1)) ? (i0p1) : (inM1));
    float v0 = _input[i0];
    float v1 = _input[i1];
    float outF = ((v0 * (as_type<float>(0x3f800000u) - frac)) + (v1 * frac));
    _output[gid] = outF;
  }
}

)kernel";
const char* k_metal_navatala_ml_maxpool_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_maxpool_f16(device const half* _input [[buffer(0)]], device const uint* window [[buffer(1)]], device const uint* stride [[buffer(2)]], device const uint* inCount [[buffer(3)]], device const uint* outCount [[buffer(4)]], device half* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint outCountVal = outCount[0];
  if (gid < outCountVal) {
    uint windowVal = window[0];
    uint strideVal = stride[0];
    uint inCountVal = inCount[0];
    uint base = (gid * strideVal);
    float acc = as_type<float>(0xff7fc99eu);
    for (int w = 0; w < (int)(windowVal); ++w) {
      uint idx = (base + ((uint)(w)));
      if (idx < inCountVal) {
        float v = ((float)(_input[idx]));
        acc = (((acc > v)) ? (acc) : (v));
      }
    }
    _output[gid] = ((half)(acc));
  }
}

)kernel";
const char* k_metal_navatala_ml_avgpool_exclude_pad_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_avgpool_exclude_pad_f16(device const half* _input [[buffer(0)]], device const uint* window [[buffer(1)]], device const uint* stride [[buffer(2)]], device const uint* inCount [[buffer(3)]], device const uint* outCount [[buffer(4)]], device half* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint outCountVal = outCount[0];
  if (gid < outCountVal) {
    uint windowVal = window[0];
    uint strideVal = stride[0];
    uint inCountVal = inCount[0];
    uint base = (gid * strideVal);
    float sum = as_type<float>(0x00000000u);
    float cnt = as_type<float>(0x00000000u);
    for (int w = 0; w < (int)(windowVal); ++w) {
      uint idx = (base + ((uint)(w)));
      if (idx < inCountVal) {
        float v = ((float)(_input[idx]));
        sum = (sum + v);
        cnt = (cnt + as_type<float>(0x3f800000u));
      }
    }
    _output[gid] = ((half)((sum / (((cnt > as_type<float>(0x00000000u))) ? (cnt) : (as_type<float>(0x3f800000u))))));
  }
}

)kernel";
const char* k_metal_navatala_ml_avgpool_include_pad_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_avgpool_include_pad_f16(device const half* _input [[buffer(0)]], device const uint* window [[buffer(1)]], device const uint* stride [[buffer(2)]], device const uint* inCount [[buffer(3)]], device const uint* outCount [[buffer(4)]], device half* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint outCountVal = outCount[0];
  if (gid < outCountVal) {
    uint windowVal = window[0];
    uint strideVal = stride[0];
    uint inCountVal = inCount[0];
    uint base = (gid * strideVal);
    float sum = as_type<float>(0x00000000u);
    float cnt = as_type<float>(0x00000000u);
    for (int w = 0; w < (int)(windowVal); ++w) {
      uint idx = (base + ((uint)(w)));
      if (idx < inCountVal) {
        float v = ((float)(_input[idx]));
        sum = (sum + v);
        cnt = (cnt + as_type<float>(0x3f800000u));
      }
    }
    _output[gid] = ((half)((sum / ((float)(windowVal)))));
  }
}

)kernel";
const char* k_metal_navatala_ml_resize_nearest_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_resize_nearest_f16(device const half* _input [[buffer(0)]], device const uint* inSize [[buffer(1)]], device const uint* outSize [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint outSizeVal = outSize[0];
  if (gid < outSizeVal) {
    uint inSizeVal = inSize[0];
    uint idx = ((gid * inSizeVal) / outSizeVal);
    half v = _input[idx];
    _output[gid] = v;
  }
}

)kernel";
const char* k_metal_navatala_ml_resize_bilinear_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_resize_bilinear_f16(device const half* _input [[buffer(0)]], device const uint* inSize [[buffer(1)]], device const uint* outSize [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint outSizeVal = outSize[0];
  if (gid < outSizeVal) {
    uint inSizeVal = inSize[0];
    uint inM1 = (inSizeVal - 1u);
    uint outM1 = (outSizeVal - 1u);
    float pos = ((((float)(gid)) * ((float)(inM1))) / ((float)(outM1)));
    float i0f = floor(pos);
    uint i0 = ((uint)(i0f));
    float frac = (pos - i0f);
    uint i0p1 = (i0 + 1u);
    uint i1 = (((i0p1 < inM1)) ? (i0p1) : (inM1));
    float v0 = ((float)(_input[i0]));
    float v1 = ((float)(_input[i1]));
    float outF = ((v0 * (as_type<float>(0x3f800000u) - frac)) + (v1 * frac));
    _output[gid] = ((half)(outF));
  }
}

)kernel";
const char* k_metal_navatala_ml_reshape_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reshape_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float v = _input[gid];
    _output[gid] = v;
  }
}

)kernel";
const char* k_metal_navatala_ml_slice_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_slice_f32(device const float* _input [[buffer(0)]], device const uint* offset [[buffer(1)]], device const uint* outCount [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint outCountVal = outCount[0];
  if (gid < outCountVal) {
    uint offsetVal = offset[0];
    uint srcIdx = (offsetVal + gid);
    float v = _input[srcIdx];
    _output[gid] = v;
  }
}

)kernel";
const char* k_metal_navatala_ml_transpose_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_transpose_f32(device const float* _input [[buffer(0)]], device const uint* nrows [[buffer(1)]], device const uint* ncols [[buffer(2)]], device const uint* count [[buffer(3)]], device float* _output [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    uint ncolsVal = ncols[0];
    uint nrowsVal = nrows[0];
    uint r = (gid / ncolsVal);
    uint c = (gid % ncolsVal);
    uint outIdx = ((c * nrowsVal) + r);
    float v = _input[gid];
    _output[outIdx] = v;
  }
}

)kernel";
const char* k_metal_navatala_ml_concatenate2_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_concatenate2_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* nA [[buffer(2)]], device const uint* nB [[buffer(3)]], device const uint* count [[buffer(4)]], device float* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    uint nAVal = nA[0];
    if (gid < nAVal) {
      float v = a[gid];
      _output[gid] = v;
    } else {
      uint srcIdx = (gid - nAVal);
      uint nBVal = nB[0];
      if (srcIdx < nBVal) {
        float v = b[srcIdx];
        _output[gid] = v;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_reshape_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reshape_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half v = _input[gid];
    _output[gid] = v;
  }
}

)kernel";
const char* k_metal_navatala_ml_slice_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_slice_f16(device const half* _input [[buffer(0)]], device const uint* offset [[buffer(1)]], device const uint* outCount [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint outCountVal = outCount[0];
  if (gid < outCountVal) {
    uint offsetVal = offset[0];
    uint srcIdx = (offsetVal + gid);
    half v = _input[srcIdx];
    _output[gid] = v;
  }
}

)kernel";
const char* k_metal_navatala_ml_transpose_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_transpose_f16(device const half* _input [[buffer(0)]], device const uint* nrows [[buffer(1)]], device const uint* ncols [[buffer(2)]], device const uint* count [[buffer(3)]], device half* _output [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    uint ncolsVal = ncols[0];
    uint nrowsVal = nrows[0];
    uint r = (gid / ncolsVal);
    uint c = (gid % ncolsVal);
    uint outIdx = ((c * nrowsVal) + r);
    half v = _input[gid];
    _output[outIdx] = v;
  }
}

)kernel";
const char* k_metal_navatala_ml_concatenate2_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_concatenate2_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* nA [[buffer(2)]], device const uint* nB [[buffer(3)]], device const uint* count [[buffer(4)]], device half* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    uint nAVal = nA[0];
    if (gid < nAVal) {
      half v = a[gid];
      _output[gid] = v;
    } else {
      uint srcIdx = (gid - nAVal);
      uint nBVal = nB[0];
      if (srcIdx < nBVal) {
        half v = b[srcIdx];
        _output[gid] = v;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_softmax_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_softmax_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsMax = as_type<float>(0xff7fc99eu);
  for (int itA = 0; itA < (int)(numIters); ++itA) {
    uint idxA = (lid + (((uint)(itA)) * 256u));
    if (idxA < countVal) {
      float xA = _input[idxA];
      gsMax = (((gsMax > xA)) ? (gsMax) : (xA));
    }
  }
  sdata[lid] = gsMax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint maxStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = maxStride;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = maxStride;
    uint nextStride = (strideToHalve >> 1u);
    maxStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float maxVal = sdata[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gsSum = as_type<float>(0x00000000u);
  for (int itB = 0; itB < (int)(numIters); ++itB) {
    uint idxB = (lid + (((uint)(itB)) * 256u));
    if (idxB < countVal) {
      float eB = exp((_input[idxB] - maxVal));
      gsSum = (gsSum + eB);
    }
  }
  sdata[lid] = gsSum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint sumStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = sumStride;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = sumStride;
    uint nextStride = (strideToHalve >> 1u);
    sumStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float sumVal = sdata[0];
  for (int itC = 0; itC < (int)(numIters); ++itC) {
    uint idxC = (lid + (((uint)(itC)) * 256u));
    if (idxC < countVal) {
      float eC = exp((_input[idxC] - maxVal));
      _output[idxC] = (eC / sumVal);
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_softmax_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_softmax_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsMax = as_type<float>(0xff7fc99eu);
  for (int itA = 0; itA < (int)(numIters); ++itA) {
    uint idxA = (lid + (((uint)(itA)) * 256u));
    if (idxA < countVal) {
      float xA = ((float)(_input[idxA]));
      gsMax = (((gsMax > xA)) ? (gsMax) : (xA));
    }
  }
  sdata[lid] = gsMax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint maxStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = maxStride;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = maxStride;
    uint nextStride = (strideToHalve >> 1u);
    maxStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float maxVal = sdata[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gsSum = as_type<float>(0x00000000u);
  for (int itB = 0; itB < (int)(numIters); ++itB) {
    uint idxB = (lid + (((uint)(itB)) * 256u));
    if (idxB < countVal) {
      float eB = exp((((float)(_input[idxB])) - maxVal));
      gsSum = (gsSum + eB);
    }
  }
  sdata[lid] = gsSum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint sumStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = sumStride;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = sumStride;
    uint nextStride = (strideToHalve >> 1u);
    sumStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float sumVal = sdata[0];
  for (int itC = 0; itC < (int)(numIters); ++itC) {
    uint idxC = (lid + (((uint)(itC)) * 256u));
    if (idxC < countVal) {
      float eC = exp((((float)(_input[idxC])) - maxVal));
      _output[idxC] = ((half)((eC / sumVal)));
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_layernorm_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_layernorm_f32(device const float* x [[buffer(0)]], device const float* gamma [[buffer(1)]], device const float* beta [[buffer(2)]], device const uint* count [[buffer(3)]], device const float* eps [[buffer(4)]], device float* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  float n = ((float)(countVal));
  uint numIters = ((countVal + 255u) / 256u);
  float epsVal = eps[0];
  float gsM = as_type<float>(0x00000000u);
  for (int itM = 0; itM < (int)(numIters); ++itM) {
    if ((lid + (((uint)(itM)) * 256u)) < countVal) {
      gsM = (gsM + x[(lid + (((uint)(itM)) * 256u))]);
    }
  }
  sdata[lid] = gsM;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint meanStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = meanStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = meanStr;
    uint nextStride = (strideToHalve >> 1u);
    meanStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float mean = (sdata[0] / n);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gsV = as_type<float>(0x00000000u);
  for (int itV = 0; itV < (int)(numIters); ++itV) {
    if ((lid + (((uint)(itV)) * 256u)) < countVal) {
      gsV = (gsV + ((x[(lid + (((uint)(itV)) * 256u))] - mean) * (x[(lid + (((uint)(itV)) * 256u))] - mean)));
    }
  }
  sdata[lid] = gsV;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint varStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = varStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = varStr;
    uint nextStride = (strideToHalve >> 1u);
    varStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float var = (sdata[0] / n);
  float denom = sqrt((var + epsVal));
  for (int itW = 0; itW < (int)(numIters); ++itW) {
    uint idxW = (lid + (((uint)(itW)) * 256u));
    if (idxW < countVal) {
      float xnW = ((x[idxW] - mean) / denom);
      float outF = ((gamma[idxW] * xnW) + beta[idxW]);
      _output[idxW] = outF;
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_spatial_batchnorm_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_spatial_batchnorm_f32(device const float* x [[buffer(0)]], device const float* gamma [[buffer(1)]], device const float* beta [[buffer(2)]], device const uint* N [[buffer(3)]], device const uint* C [[buffer(4)]], device const uint* HW [[buffer(5)]], device const float* eps [[buffer(6)]], device float* _output [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint grp = ((uint)(int(__tgid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint Nv = N[0];
  uint Cv = C[0];
  uint HWv = HW[0];
  uint CHW = (Cv * HWv);
  uint chBase = (grp * HWv);
  uint countVal = (Nv * HWv);
  float n = ((float)(countVal));
  uint numIters = ((countVal + 255u) / 256u);
  float epsVal = eps[0];
  float gsM = as_type<float>(0x00000000u);
  for (int itM = 0; itM < (int)(numIters); ++itM) {
    if ((lid + (((uint)(itM)) * 256u)) < countVal) {
      gsM = (gsM + x[(((((lid + (((uint)(itM)) * 256u)) / HWv) * CHW) + chBase) + ((lid + (((uint)(itM)) * 256u)) % HWv))]);
    }
  }
  sdata[lid] = gsM;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint meanStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = meanStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = meanStr;
    uint nextStride = (strideToHalve >> 1u);
    meanStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float mean = (sdata[0] / n);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gsV = as_type<float>(0x00000000u);
  for (int itV = 0; itV < (int)(numIters); ++itV) {
    if ((lid + (((uint)(itV)) * 256u)) < countVal) {
      gsV = (gsV + ((x[(((((lid + (((uint)(itV)) * 256u)) / HWv) * CHW) + chBase) + ((lid + (((uint)(itV)) * 256u)) % HWv))] - mean) * (x[(((((lid + (((uint)(itV)) * 256u)) / HWv) * CHW) + chBase) + ((lid + (((uint)(itV)) * 256u)) % HWv))] - mean)));
    }
  }
  sdata[lid] = gsV;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint varStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = varStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = varStr;
    uint nextStride = (strideToHalve >> 1u);
    varStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float var = (sdata[0] / n);
  float denom = sqrt((var + epsVal));
  for (int itW = 0; itW < (int)(numIters); ++itW) {
    if ((lid + (((uint)(itW)) * 256u)) < countVal) {
      float gv = gamma[grp];
      float bv = beta[grp];
      float xv = x[(((((lid + (((uint)(itW)) * 256u)) / HWv) * CHW) + chBase) + ((lid + (((uint)(itW)) * 256u)) % HWv))];
      float xn = ((xv - mean) / denom);
      _output[(((((lid + (((uint)(itW)) * 256u)) / HWv) * CHW) + chBase) + ((lid + (((uint)(itW)) * 256u)) % HWv))] = ((gv * xn) + bv);
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_instancenorm_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_instancenorm_f32(device const float* x [[buffer(0)]], device const float* gamma [[buffer(1)]], device const float* beta [[buffer(2)]], device const uint* C [[buffer(3)]], device const uint* HW [[buffer(4)]], device const float* eps [[buffer(5)]], device float* _output [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint grp = ((uint)(int(__tgid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint Cv = C[0];
  uint HWv = HW[0];
  uint chan = (grp % Cv);
  uint base = (grp * HWv);
  uint countVal = HWv;
  float n = ((float)(countVal));
  uint numIters = ((countVal + 255u) / 256u);
  float epsVal = eps[0];
  float gsM = as_type<float>(0x00000000u);
  for (int itM = 0; itM < (int)(numIters); ++itM) {
    if ((lid + (((uint)(itM)) * 256u)) < countVal) {
      gsM = (gsM + x[(base + (lid + (((uint)(itM)) * 256u)))]);
    }
  }
  sdata[lid] = gsM;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint meanStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = meanStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = meanStr;
    uint nextStride = (strideToHalve >> 1u);
    meanStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float mean = (sdata[0] / n);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gsV = as_type<float>(0x00000000u);
  for (int itV = 0; itV < (int)(numIters); ++itV) {
    if ((lid + (((uint)(itV)) * 256u)) < countVal) {
      gsV = (gsV + ((x[(base + (lid + (((uint)(itV)) * 256u)))] - mean) * (x[(base + (lid + (((uint)(itV)) * 256u)))] - mean)));
    }
  }
  sdata[lid] = gsV;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint varStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = varStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = varStr;
    uint nextStride = (strideToHalve >> 1u);
    varStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float var = (sdata[0] / n);
  float denom = sqrt((var + epsVal));
  for (int itW = 0; itW < (int)(numIters); ++itW) {
    if ((lid + (((uint)(itW)) * 256u)) < countVal) {
      float gv = gamma[chan];
      float bv = beta[chan];
      float xv = x[(base + (lid + (((uint)(itW)) * 256u)))];
      float xn = ((xv - mean) / denom);
      _output[(base + (lid + (((uint)(itW)) * 256u)))] = ((gv * xn) + bv);
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_batchnorm_inference_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_batchnorm_inference_f32(device const float* x [[buffer(0)]], device const float* mean [[buffer(1)]], device const float* var [[buffer(2)]], device const float* gamma [[buffer(3)]], device const float* beta [[buffer(4)]], device const uint* count [[buffer(5)]], device const float* eps [[buffer(6)]], device float* _output [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float epsVal = eps[0];
    float d = (x[gid] - mean[gid]);
    float denom = sqrt((var[gid] + epsVal));
    float xn = (d / denom);
    float outF = ((gamma[gid] * xn) + beta[gid]);
    _output[gid] = outF;
  }
}

)kernel";
const char* k_metal_navatala_ml_layernorm_backward_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_layernorm_backward_f32(device const float* x [[buffer(0)]], device const float* dy [[buffer(1)]], device const float* gamma [[buffer(2)]], device const uint* count [[buffer(3)]], device const float* eps [[buffer(4)]], device float* dx [[buffer(5)]], device float* dgamma [[buffer(6)]], device float* dbeta [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  float n = ((float)(countVal));
  uint numIters = ((countVal + 255u) / 256u);
  float epsVal = eps[0];
  float gM = as_type<float>(0x00000000u);
  for (int itm = 0; itm < (int)(numIters); ++itm) {
    if ((lid + (((uint)(itm)) * 256u)) < countVal) {
      gM = (gM + x[(lid + (((uint)(itm)) * 256u))]);
    }
  }
  sdata[lid] = gM;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint mStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = mStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = mStr;
    uint nextStride = (strideToHalve >> 1u);
    mStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float mean = (sdata[0] / n);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gV = as_type<float>(0x00000000u);
  for (int itv = 0; itv < (int)(numIters); ++itv) {
    if ((lid + (((uint)(itv)) * 256u)) < countVal) {
      gV = (gV + ((x[(lid + (((uint)(itv)) * 256u))] - mean) * (x[(lid + (((uint)(itv)) * 256u))] - mean)));
    }
  }
  sdata[lid] = gV;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint vStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = vStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = vStr;
    uint nextStride = (strideToHalve >> 1u);
    vStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float var = (sdata[0] / n);
  float std = sqrt((var + epsVal));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float g1 = as_type<float>(0x00000000u);
  for (int it1 = 0; it1 < (int)(numIters); ++it1) {
    if ((lid + (((uint)(it1)) * 256u)) < countVal) {
      g1 = (g1 + (dy[(lid + (((uint)(it1)) * 256u))] * gamma[(lid + (((uint)(it1)) * 256u))]));
    }
  }
  sdata[lid] = g1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint s1Str = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = s1Str;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = s1Str;
    uint nextStride = (strideToHalve >> 1u);
    s1Str = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float m1 = (sdata[0] / n);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float g2 = as_type<float>(0x00000000u);
  for (int it2 = 0; it2 < (int)(numIters); ++it2) {
    if ((lid + (((uint)(it2)) * 256u)) < countVal) {
      g2 = (g2 + ((dy[(lid + (((uint)(it2)) * 256u))] * gamma[(lid + (((uint)(it2)) * 256u))]) * ((x[(lid + (((uint)(it2)) * 256u))] - mean) / std)));
    }
  }
  sdata[lid] = g2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint s2Str = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = s2Str;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = s2Str;
    uint nextStride = (strideToHalve >> 1u);
    s2Str = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float m2 = (sdata[0] / n);
  for (int itw = 0; itw < (int)(numIters); ++itw) {
    uint idw = (lid + (((uint)(itw)) * 256u));
    if (idw < countVal) {
      float xhatw = ((x[idw] - mean) / std);
      float dyvw = dy[idw];
      float dxhw = (dyvw * gamma[idw]);
      float dxFw = (((dxhw - m1) - (xhatw * m2)) / std);
      dx[idw] = dxFw;
      dgamma[idw] = (dyvw * xhatw);
      dbeta[idw] = dyvw;
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_layernorm_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_layernorm_f16(device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]], device const half* beta [[buffer(2)]], device const uint* count [[buffer(3)]], device const float* eps [[buffer(4)]], device half* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  float n = ((float)(countVal));
  uint numIters = ((countVal + 255u) / 256u);
  float epsVal = eps[0];
  float gsM = as_type<float>(0x00000000u);
  for (int itM = 0; itM < (int)(numIters); ++itM) {
    if ((lid + (((uint)(itM)) * 256u)) < countVal) {
      gsM = (gsM + ((float)(x[(lid + (((uint)(itM)) * 256u))])));
    }
  }
  sdata[lid] = gsM;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint meanStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = meanStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = meanStr;
    uint nextStride = (strideToHalve >> 1u);
    meanStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float mean = (sdata[0] / n);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gsV = as_type<float>(0x00000000u);
  for (int itV = 0; itV < (int)(numIters); ++itV) {
    if ((lid + (((uint)(itV)) * 256u)) < countVal) {
      gsV = (gsV + ((((float)(x[(lid + (((uint)(itV)) * 256u))])) - mean) * (((float)(x[(lid + (((uint)(itV)) * 256u))])) - mean)));
    }
  }
  sdata[lid] = gsV;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint varStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = varStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = varStr;
    uint nextStride = (strideToHalve >> 1u);
    varStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float var = (sdata[0] / n);
  float denom = sqrt((var + epsVal));
  for (int itW = 0; itW < (int)(numIters); ++itW) {
    uint idxW = (lid + (((uint)(itW)) * 256u));
    if (idxW < countVal) {
      float xnW = ((((float)(x[idxW])) - mean) / denom);
      float outF = ((((float)(gamma[idxW])) * xnW) + ((float)(beta[idxW])));
      _output[idxW] = ((half)(outF));
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_spatial_batchnorm_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_spatial_batchnorm_f16(device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]], device const half* beta [[buffer(2)]], device const uint* N [[buffer(3)]], device const uint* C [[buffer(4)]], device const uint* HW [[buffer(5)]], device const float* eps [[buffer(6)]], device half* _output [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint grp = ((uint)(int(__tgid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint Nv = N[0];
  uint Cv = C[0];
  uint HWv = HW[0];
  uint CHW = (Cv * HWv);
  uint chBase = (grp * HWv);
  uint countVal = (Nv * HWv);
  float n = ((float)(countVal));
  uint numIters = ((countVal + 255u) / 256u);
  float epsVal = eps[0];
  float gsM = as_type<float>(0x00000000u);
  for (int itM = 0; itM < (int)(numIters); ++itM) {
    if ((lid + (((uint)(itM)) * 256u)) < countVal) {
      gsM = (gsM + ((float)(x[(((((lid + (((uint)(itM)) * 256u)) / HWv) * CHW) + chBase) + ((lid + (((uint)(itM)) * 256u)) % HWv))])));
    }
  }
  sdata[lid] = gsM;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint meanStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = meanStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = meanStr;
    uint nextStride = (strideToHalve >> 1u);
    meanStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float mean = (sdata[0] / n);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gsV = as_type<float>(0x00000000u);
  for (int itV = 0; itV < (int)(numIters); ++itV) {
    if ((lid + (((uint)(itV)) * 256u)) < countVal) {
      gsV = (gsV + ((((float)(x[(((((lid + (((uint)(itV)) * 256u)) / HWv) * CHW) + chBase) + ((lid + (((uint)(itV)) * 256u)) % HWv))])) - mean) * (((float)(x[(((((lid + (((uint)(itV)) * 256u)) / HWv) * CHW) + chBase) + ((lid + (((uint)(itV)) * 256u)) % HWv))])) - mean)));
    }
  }
  sdata[lid] = gsV;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint varStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = varStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = varStr;
    uint nextStride = (strideToHalve >> 1u);
    varStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float var = (sdata[0] / n);
  float denom = sqrt((var + epsVal));
  for (int itW = 0; itW < (int)(numIters); ++itW) {
    if ((lid + (((uint)(itW)) * 256u)) < countVal) {
      float gv = ((float)(gamma[grp]));
      float bv = ((float)(beta[grp]));
      float xv = ((float)(x[(((((lid + (((uint)(itW)) * 256u)) / HWv) * CHW) + chBase) + ((lid + (((uint)(itW)) * 256u)) % HWv))]));
      float xn = ((xv - mean) / denom);
      _output[(((((lid + (((uint)(itW)) * 256u)) / HWv) * CHW) + chBase) + ((lid + (((uint)(itW)) * 256u)) % HWv))] = ((half)(((gv * xn) + bv)));
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_instancenorm_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_instancenorm_f16(device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]], device const half* beta [[buffer(2)]], device const uint* C [[buffer(3)]], device const uint* HW [[buffer(4)]], device const float* eps [[buffer(5)]], device half* _output [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint grp = ((uint)(int(__tgid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint Cv = C[0];
  uint HWv = HW[0];
  uint chan = (grp % Cv);
  uint base = (grp * HWv);
  uint countVal = HWv;
  float n = ((float)(countVal));
  uint numIters = ((countVal + 255u) / 256u);
  float epsVal = eps[0];
  float gsM = as_type<float>(0x00000000u);
  for (int itM = 0; itM < (int)(numIters); ++itM) {
    if ((lid + (((uint)(itM)) * 256u)) < countVal) {
      gsM = (gsM + ((float)(x[(base + (lid + (((uint)(itM)) * 256u)))])));
    }
  }
  sdata[lid] = gsM;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint meanStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = meanStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = meanStr;
    uint nextStride = (strideToHalve >> 1u);
    meanStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float mean = (sdata[0] / n);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gsV = as_type<float>(0x00000000u);
  for (int itV = 0; itV < (int)(numIters); ++itV) {
    if ((lid + (((uint)(itV)) * 256u)) < countVal) {
      gsV = (gsV + ((((float)(x[(base + (lid + (((uint)(itV)) * 256u)))])) - mean) * (((float)(x[(base + (lid + (((uint)(itV)) * 256u)))])) - mean)));
    }
  }
  sdata[lid] = gsV;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint varStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = varStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = varStr;
    uint nextStride = (strideToHalve >> 1u);
    varStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float var = (sdata[0] / n);
  float denom = sqrt((var + epsVal));
  for (int itW = 0; itW < (int)(numIters); ++itW) {
    if ((lid + (((uint)(itW)) * 256u)) < countVal) {
      float gv = ((float)(gamma[chan]));
      float bv = ((float)(beta[chan]));
      float xv = ((float)(x[(base + (lid + (((uint)(itW)) * 256u)))]));
      float xn = ((xv - mean) / denom);
      _output[(base + (lid + (((uint)(itW)) * 256u)))] = ((half)(((gv * xn) + bv)));
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_batchnorm_inference_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_batchnorm_inference_f16(device const half* x [[buffer(0)]], device const half* mean [[buffer(1)]], device const half* var [[buffer(2)]], device const half* gamma [[buffer(3)]], device const half* beta [[buffer(4)]], device const uint* count [[buffer(5)]], device const float* eps [[buffer(6)]], device half* _output [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float epsVal = eps[0];
    float d = (((float)(x[gid])) - ((float)(mean[gid])));
    float denom = sqrt((((float)(var[gid])) + epsVal));
    float xn = (d / denom);
    float outF = ((((float)(gamma[gid])) * xn) + ((float)(beta[gid])));
    _output[gid] = ((half)(outF));
  }
}

)kernel";
const char* k_metal_navatala_ml_layernorm_backward_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_layernorm_backward_f16(device const half* x [[buffer(0)]], device const half* dy [[buffer(1)]], device const half* gamma [[buffer(2)]], device const uint* count [[buffer(3)]], device const float* eps [[buffer(4)]], device half* dx [[buffer(5)]], device half* dgamma [[buffer(6)]], device half* dbeta [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  float n = ((float)(countVal));
  uint numIters = ((countVal + 255u) / 256u);
  float epsVal = eps[0];
  float gM = as_type<float>(0x00000000u);
  for (int itm = 0; itm < (int)(numIters); ++itm) {
    if ((lid + (((uint)(itm)) * 256u)) < countVal) {
      gM = (gM + ((float)(x[(lid + (((uint)(itm)) * 256u))])));
    }
  }
  sdata[lid] = gM;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint mStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = mStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = mStr;
    uint nextStride = (strideToHalve >> 1u);
    mStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float mean = (sdata[0] / n);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gV = as_type<float>(0x00000000u);
  for (int itv = 0; itv < (int)(numIters); ++itv) {
    if ((lid + (((uint)(itv)) * 256u)) < countVal) {
      gV = (gV + ((((float)(x[(lid + (((uint)(itv)) * 256u))])) - mean) * (((float)(x[(lid + (((uint)(itv)) * 256u))])) - mean)));
    }
  }
  sdata[lid] = gV;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint vStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = vStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = vStr;
    uint nextStride = (strideToHalve >> 1u);
    vStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float var = (sdata[0] / n);
  float std = sqrt((var + epsVal));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float g1 = as_type<float>(0x00000000u);
  for (int it1 = 0; it1 < (int)(numIters); ++it1) {
    if ((lid + (((uint)(it1)) * 256u)) < countVal) {
      g1 = (g1 + (((float)(dy[(lid + (((uint)(it1)) * 256u))])) * ((float)(gamma[(lid + (((uint)(it1)) * 256u))]))));
    }
  }
  sdata[lid] = g1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint s1Str = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = s1Str;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = s1Str;
    uint nextStride = (strideToHalve >> 1u);
    s1Str = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float m1 = (sdata[0] / n);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float g2 = as_type<float>(0x00000000u);
  for (int it2 = 0; it2 < (int)(numIters); ++it2) {
    if ((lid + (((uint)(it2)) * 256u)) < countVal) {
      g2 = (g2 + ((((float)(dy[(lid + (((uint)(it2)) * 256u))])) * ((float)(gamma[(lid + (((uint)(it2)) * 256u))]))) * ((((float)(x[(lid + (((uint)(it2)) * 256u))])) - mean) / std)));
    }
  }
  sdata[lid] = g2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint s2Str = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = s2Str;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = s2Str;
    uint nextStride = (strideToHalve >> 1u);
    s2Str = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float m2 = (sdata[0] / n);
  for (int itw = 0; itw < (int)(numIters); ++itw) {
    uint idw = (lid + (((uint)(itw)) * 256u));
    if (idw < countVal) {
      float xhatw = ((((float)(x[idw])) - mean) / std);
      float dyvw = ((float)(dy[idw]));
      float dxhw = (dyvw * ((float)(gamma[idw])));
      float dxFw = (((dxhw - m1) - (xhatw * m2)) / std);
      dx[idw] = ((half)(dxFw));
      dgamma[idw] = ((half)((dyvw * xhatw)));
      dbeta[idw] = ((half)(dyvw));
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_uniform_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_uniform_f32(device const uint* count [[buffer(0)]], device const uint* seed [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  uint seedVal = seed[0];
  if (gid < countVal) {
    uint uk0 = ((seedVal + (gid * 2654435769u)) + 0u);
    uint uk1 = ((uk0 ^ (uk0 >> 16u)) * 2246822519u);
    uint uk2 = ((uk1 ^ (uk1 >> 13u)) * 3266489917u);
    uint uk3 = (uk2 ^ (uk2 >> 16u));
    float uF = (((float)(uk3)) * as_type<float>(0x2f800000u));
    float outV = uF;
    _output[gid] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_bernoulli_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_bernoulli_f32(device const uint* count [[buffer(0)]], device const uint* seed [[buffer(1)]], device const float* p [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  uint seedVal = seed[0];
  if (gid < countVal) {
    uint uk0 = ((seedVal + (gid * 2654435769u)) + 0u);
    uint uk1 = ((uk0 ^ (uk0 >> 16u)) * 2246822519u);
    uint uk2 = ((uk1 ^ (uk1 >> 13u)) * 3266489917u);
    uint uk3 = (uk2 ^ (uk2 >> 16u));
    float uF = (((float)(uk3)) * as_type<float>(0x2f800000u));
    float pVal = p[0];
    float bit = (((uF < pVal)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    float outV = bit;
    _output[gid] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_normal_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_normal_f32(device const uint* count [[buffer(0)]], device const uint* seed [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  uint seedVal = seed[0];
  if (gid < countVal) {
    uint ak0 = ((seedVal + (gid * 2654435769u)) + 0u);
    uint ak1 = ((ak0 ^ (ak0 >> 16u)) * 2246822519u);
    uint ak2 = ((ak1 ^ (ak1 >> 13u)) * 3266489917u);
    uint ak3 = (ak2 ^ (ak2 >> 16u));
    float u1 = (((float)(ak3)) * as_type<float>(0x2f800000u));
    uint bk0 = ((seedVal + (gid * 2654435769u)) + 2246822519u);
    uint bk1 = ((bk0 ^ (bk0 >> 16u)) * 2246822519u);
    uint bk2 = ((bk1 ^ (bk1 >> 13u)) * 3266489917u);
    uint bk3 = (bk2 ^ (bk2 >> 16u));
    float u2 = (((float)(bk3)) * as_type<float>(0x2f800000u));
    float u1e = (u1 + as_type<float>(0x33d6bf95u));
    float r = sqrt((as_type<float>(0xc0000000u) * log(u1e)));
    float z = (r * cos((as_type<float>(0x40c90fdbu) * u2)));
    float outV = z;
    _output[gid] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_uniform_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_uniform_f16(device const uint* count [[buffer(0)]], device const uint* seed [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  uint seedVal = seed[0];
  if (gid < countVal) {
    uint uk0 = ((seedVal + (gid * 2654435769u)) + 0u);
    uint uk1 = ((uk0 ^ (uk0 >> 16u)) * 2246822519u);
    uint uk2 = ((uk1 ^ (uk1 >> 13u)) * 3266489917u);
    uint uk3 = (uk2 ^ (uk2 >> 16u));
    float uF = (((float)(uk3)) * as_type<float>(0x2f800000u));
    half outV = ((half)(uF));
    _output[gid] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_bernoulli_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_bernoulli_f16(device const uint* count [[buffer(0)]], device const uint* seed [[buffer(1)]], device const float* p [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  uint seedVal = seed[0];
  if (gid < countVal) {
    uint uk0 = ((seedVal + (gid * 2654435769u)) + 0u);
    uint uk1 = ((uk0 ^ (uk0 >> 16u)) * 2246822519u);
    uint uk2 = ((uk1 ^ (uk1 >> 13u)) * 3266489917u);
    uint uk3 = (uk2 ^ (uk2 >> 16u));
    float uF = (((float)(uk3)) * as_type<float>(0x2f800000u));
    float pVal = p[0];
    float bit = (((uF < pVal)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    half outV = ((half)(bit));
    _output[gid] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_normal_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_normal_f16(device const uint* count [[buffer(0)]], device const uint* seed [[buffer(1)]], device half* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  uint seedVal = seed[0];
  if (gid < countVal) {
    uint ak0 = ((seedVal + (gid * 2654435769u)) + 0u);
    uint ak1 = ((ak0 ^ (ak0 >> 16u)) * 2246822519u);
    uint ak2 = ((ak1 ^ (ak1 >> 13u)) * 3266489917u);
    uint ak3 = (ak2 ^ (ak2 >> 16u));
    float u1 = (((float)(ak3)) * as_type<float>(0x2f800000u));
    uint bk0 = ((seedVal + (gid * 2654435769u)) + 2246822519u);
    uint bk1 = ((bk0 ^ (bk0 >> 16u)) * 2246822519u);
    uint bk2 = ((bk1 ^ (bk1 >> 13u)) * 3266489917u);
    uint bk3 = (bk2 ^ (bk2 >> 16u));
    float u2 = (((float)(bk3)) * as_type<float>(0x2f800000u));
    float u1e = (u1 + as_type<float>(0x33d6bf95u));
    float r = sqrt((as_type<float>(0xc0000000u) * log(u1e)));
    float z = (r * cos((as_type<float>(0x40c90fdbu) * u2)));
    half outV = ((half)(z));
    _output[gid] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_kde_bandwidth_silverman_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_kde_bandwidth_silverman_f32(device const float* stddev [[buffer(0)]], device const uint* n [[buffer(1)]], device float* bandwidth [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  if (gid == 0u) {
    float sigma = stddev[0];
    uint count = n[0];
    float countF = ((float)(count));
    float logN = log(countF);
    float negLogN = (as_type<float>(0xbe4ccccdu) * logN);
    float nPowNeg02 = exp(negLogN);
    float h = (as_type<float>(0x3f87ae14u) * (sigma * nPowNeg02));
    bandwidth[0] = h;
  }
}

)kernel";
const char* k_metal_navatala_ml_kde_bandwidth_scott_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_kde_bandwidth_scott_f32(device const float* stddev [[buffer(0)]], device const uint* n [[buffer(1)]], device float* bandwidth [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  if (gid == 0u) {
    float sigma = stddev[0];
    uint count = n[0];
    float countF = ((float)(count));
    float logN = log(countF);
    float negLogN = (as_type<float>(0xbe4ccccdu) * logN);
    float nPowNeg02 = exp(negLogN);
    float h = (sigma * nPowNeg02);
    bandwidth[0] = h;
  }
}

)kernel";
const char* k_metal_navatala_ml_kde_evaluate_gaussian_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_kde_evaluate_gaussian_f32(device const float* data [[buffer(0)]], device const float* queryPoints [[buffer(1)]], device const float* bandwidth [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* m [[buffer(4)]], device float* densities [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numQueries = m[0];
  bool inBounds = (gid < numQueries);
  if (inBounds) {
    uint numSamples = n[0];
    float h = bandwidth[0];
    float x = queryPoints[gid];
    float invSqrt2Pi = as_type<float>(0x3ecc422au);
    float sum = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(numSamples); ++i) {
      uint iU32 = ((uint)(i));
      float xi = data[iU32];
      float diff = (x - xi);
      float u = (diff / h);
      float uSq = (u * u);
      float negHalfUSq = (as_type<float>(0xbf000000u) * uSq);
      float expTerm = exp(negHalfUSq);
      float kernelVal = (invSqrt2Pi * expTerm);
      float currSum = sum;
      sum = (currSum + kernelVal);
    }
    float nFloat = ((float)(numSamples));
    float nh = (nFloat * h);
    float finalSum = sum;
    float density = (finalSum / nh);
    densities[gid] = density;
  }
}

)kernel";
const char* k_metal_navatala_ml_kde_evaluate_epanechnikov_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_kde_evaluate_epanechnikov_f32(device const float* data [[buffer(0)]], device const float* queryPoints [[buffer(1)]], device const float* bandwidth [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* m [[buffer(4)]], device float* densities [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numQueries = m[0];
  bool inBounds = (gid < numQueries);
  if (inBounds) {
    uint numSamples = n[0];
    float h = bandwidth[0];
    float x = queryPoints[gid];
    float sum = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(numSamples); ++i) {
      uint iU32 = ((uint)(i));
      float xi = data[iU32];
      float diff = (x - xi);
      float u = (diff / h);
      float absU = abs(u);
      float uSq = (u * u);
      float oneMinusUSq = (as_type<float>(0x3f800000u) - uSq);
      float kernelRaw = (as_type<float>(0x3f400000u) * oneMinusUSq);
      bool inSupport = (absU <= as_type<float>(0x3f800000u));
      float kernelVal = ((inSupport) ? (kernelRaw) : (as_type<float>(0x00000000u)));
      float currSum = sum;
      sum = (currSum + kernelVal);
    }
    float nFloat = ((float)(numSamples));
    float nh = (nFloat * h);
    float finalSum = sum;
    float density = (finalSum / nh);
    densities[gid] = density;
  }
}

)kernel";
const char* k_metal_navatala_ml_kde_find_mode_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_kde_find_mode_f32(device const float* densities [[buffer(0)]], device const float* queryPoints [[buffer(1)]], device const uint* m [[buffer(2)]], device float* modeValue [[buffer(3)]], device float* modeLocation [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  if (gid == 0u) {
    uint numPoints = m[0];
    float maxDensity = densities[0];
    float maxLocation = queryPoints[0];
    for (int i = 0; i < (int)(numPoints); ++i) {
      uint iU32 = ((uint)(i));
      float d = densities[iU32];
      float loc = queryPoints[iU32];
      float currMaxDensity = maxDensity;
      float currMaxLocation = maxLocation;
      bool isGreater = (d > currMaxDensity);
      float newMaxDensity = ((isGreater) ? (d) : (currMaxDensity));
      float newMaxLocation = ((isGreater) ? (loc) : (currMaxLocation));
      maxDensity = newMaxDensity;
      maxLocation = newMaxLocation;
    }
    float finalMaxDensity = maxDensity;
    float finalMaxLocation = maxLocation;
    modeValue[0] = finalMaxDensity;
    modeLocation[0] = finalMaxLocation;
  }
}

)kernel";
const char* k_metal_navatala_ml_pairwise_euclidean_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pairwise_euclidean_f32(device const float* X [[buffer(0)]], device const float* Y [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* m [[buffer(3)]], device const uint* d [[buffer(4)]], device float* D [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint mVal = m[0];
  uint dVal = d[0];
  uint totalElems = (nVal * mVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / mVal);
    uint j = (gid % mVal);
    uint xBase = (i * dVal);
    uint yBase = (j * dVal);
    float sumSq = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(dVal); ++k) {
      uint kU32 = ((uint)(k));
      uint xIdx = (xBase + kU32);
      uint yIdx = (yBase + kU32);
      float xVal = X[xIdx];
      float yVal = Y[yIdx];
      float diff = (xVal - yVal);
      float diffSq = (diff * diff);
      float currentSum = sumSq;
      float newSum = (currentSum + diffSq);
      sumSq = newSum;
    }
    float finalSum = sumSq;
    float dist = sqrt(finalSum);
    D[gid] = dist;
  }
}

)kernel";
const char* k_metal_navatala_ml_pairwise_squared_euclidean_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pairwise_squared_euclidean_f32(device const float* X [[buffer(0)]], device const float* Y [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* m [[buffer(3)]], device const uint* d [[buffer(4)]], device float* D [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint mVal = m[0];
  uint dVal = d[0];
  uint totalElems = (nVal * mVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / mVal);
    uint j = (gid % mVal);
    uint xBase = (i * dVal);
    uint yBase = (j * dVal);
    float sumSq = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(dVal); ++k) {
      uint kU32 = ((uint)(k));
      uint xIdx = (xBase + kU32);
      uint yIdx = (yBase + kU32);
      float xVal = X[xIdx];
      float yVal = Y[yIdx];
      float diff = (xVal - yVal);
      float diffSq = (diff * diff);
      float currentSum = sumSq;
      float newSum = (currentSum + diffSq);
      sumSq = newSum;
    }
    float finalSum = sumSq;
    D[gid] = finalSum;
  }
}

)kernel";
const char* k_metal_navatala_ml_pairwise_manhattan_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pairwise_manhattan_f32(device const float* X [[buffer(0)]], device const float* Y [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* m [[buffer(3)]], device const uint* d [[buffer(4)]], device float* D [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint mVal = m[0];
  uint dVal = d[0];
  uint totalElems = (nVal * mVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / mVal);
    uint j = (gid % mVal);
    uint xBase = (i * dVal);
    uint yBase = (j * dVal);
    float sumAbs = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(dVal); ++k) {
      uint kU32 = ((uint)(k));
      uint xIdx = (xBase + kU32);
      uint yIdx = (yBase + kU32);
      float xVal = X[xIdx];
      float yVal = Y[yIdx];
      float diff = (xVal - yVal);
      float absDiff = abs(diff);
      float currentSum = sumAbs;
      float newSum = (currentSum + absDiff);
      sumAbs = newSum;
    }
    float finalSum = sumAbs;
    D[gid] = finalSum;
  }
}

)kernel";
const char* k_metal_navatala_ml_pairwise_cosine_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pairwise_cosine_f32(device const float* X [[buffer(0)]], device const float* Y [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* m [[buffer(3)]], device const uint* d [[buffer(4)]], device float* D [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint mVal = m[0];
  uint dVal = d[0];
  uint totalElems = (nVal * mVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / mVal);
    uint j = (gid % mVal);
    uint xBase = (i * dVal);
    uint yBase = (j * dVal);
    float dotProduct = as_type<float>(0x00000000u);
    float normXSq = as_type<float>(0x00000000u);
    float normYSq = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(dVal); ++k) {
      uint kU32 = ((uint)(k));
      uint xIdx = (xBase + kU32);
      uint yIdx = (yBase + kU32);
      float xVal = X[xIdx];
      float yVal = Y[yIdx];
      float prod = (xVal * yVal);
      float currentDot = dotProduct;
      float newDot = (currentDot + prod);
      dotProduct = newDot;
      float xSq = (xVal * xVal);
      float ySq = (yVal * yVal);
      float currentNormX = normXSq;
      float newNormX = (currentNormX + xSq);
      normXSq = newNormX;
      float currentNormY = normYSq;
      float newNormY = (currentNormY + ySq);
      normYSq = newNormY;
    }
    float finalDot = dotProduct;
    float finalNormXSq = normXSq;
    float finalNormYSq = normYSq;
    float normX = sqrt(finalNormXSq);
    float normY = sqrt(finalNormYSq);
    float normProduct = (normX * normY);
    bool normIsZero = (normProduct == as_type<float>(0x00000000u));
    if (normIsZero) {
      D[gid] = as_type<float>(0x00000000u);
    } else {
      float cosineSim = (finalDot / normProduct);
      float cosineDist = (as_type<float>(0x3f800000u) - cosineSim);
      D[gid] = cosineDist;
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_pairwise_chebyshev_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_pairwise_chebyshev_f32(device const float* X [[buffer(0)]], device const float* Y [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* m [[buffer(3)]], device const uint* d [[buffer(4)]], device float* D [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint mVal = m[0];
  uint dVal = d[0];
  uint totalElems = (nVal * mVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / mVal);
    uint j = (gid % mVal);
    uint xBase = (i * dVal);
    uint yBase = (j * dVal);
    float maxDiff = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(dVal); ++k) {
      uint kU32 = ((uint)(k));
      uint xIdx = (xBase + kU32);
      uint yIdx = (yBase + kU32);
      float xVal = X[xIdx];
      float yVal = Y[yIdx];
      float diff = (xVal - yVal);
      float absDiff = abs(diff);
      float currentMax = maxDiff;
      bool isGreater = (absDiff > currentMax);
      float newMax = ((isGreater) ? (absDiff) : (currentMax));
      maxDiff = newMax;
    }
    float finalMax = maxDiff;
    D[gid] = finalMax;
  }
}

)kernel";
const char* k_metal_navatala_ml_nearest_neighbor_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_nearest_neighbor_f32(device const float* D [[buffer(0)]], device const uint* n [[buffer(1)]], device const uint* m [[buffer(2)]], device uint* indices [[buffer(3)]], device float* distances [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint mVal = m[0];
  uint rowIdx = gid;
  bool inBounds = (rowIdx < nVal);
  if (inBounds) {
    uint rowBase = (rowIdx * mVal);
    float initDist = D[rowBase];
    float minDist = initDist;
    uint minIdx = 0u;
    for (int j = 0; j < (int)(mVal); ++j) {
      uint jU32 = ((uint)(j));
      uint colIdx = (rowBase + jU32);
      float dist = D[colIdx];
      float currentMinDist = minDist;
      uint currentMinIdx = minIdx;
      bool isSmaller = (dist < currentMinDist);
      float newMinDist = ((isSmaller) ? (dist) : (currentMinDist));
      uint newMinIdx = ((isSmaller) ? (jU32) : (currentMinIdx));
      minDist = newMinDist;
      minIdx = newMinIdx;
    }
    uint finalMinIdx = minIdx;
    float finalMinDist = minDist;
    indices[rowIdx] = finalMinIdx;
    distances[rowIdx] = finalMinDist;
  }
}

)kernel";
const char* k_metal_navatala_ml_init_centroids_random_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_init_centroids_random_f32(device const float* data [[buffer(0)]], device const uint* indices [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* d [[buffer(4)]], device float* centroids [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint kVal = k[0];
  uint dVal = d[0];
  uint totalElems = (kVal * dVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint centroidIdx = (gid / dVal);
    uint dimIdx = (gid % dVal);
    uint dataPointIdx = indices[centroidIdx];
    uint srcIdx = ((dataPointIdx * dVal) + dimIdx);
    float val = data[srcIdx];
    centroids[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_ml_assign_to_centroids_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_assign_to_centroids_f32(device const float* data [[buffer(0)]], device const float* centroids [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* d [[buffer(4)]], device uint* assignments [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint kVal = k[0];
  uint dVal = d[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint dataBase = (gid * dVal);
    uint bestClusterAccum = 0u;
    float bestDistAccum = as_type<float>(0x7f7fffffu);
    float distSumAccum = as_type<float>(0x00000000u);
    for (int c = 0; c < (int)(kVal); ++c) {
      uint cU32 = ((uint)(c));
      uint centroidBase = (cU32 * dVal);
      distSumAccum = as_type<float>(0x00000000u);
      for (int j = 0; j < (int)(dVal); ++j) {
        uint jU32 = ((uint)(j));
        uint dataIdx = (dataBase + jU32);
        uint centIdx = (centroidBase + jU32);
        float dataVal = data[dataIdx];
        float centVal = centroids[centIdx];
        float diff = (dataVal - centVal);
        float diffSq = (diff * diff);
        float currDistSum = distSumAccum;
        float newDistSum = (currDistSum + diffSq);
        distSumAccum = newDistSum;
      }
      float distSum = distSumAccum;
      float bestDist = bestDistAccum;
      if (distSum < bestDist) {
        bestDistAccum = distSum;
        bestClusterAccum = cU32;
      }
    }
    uint bestCluster = bestClusterAccum;
    assignments[gid] = bestCluster;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_cluster_sums_f32 = R"kernel(
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

kernel void navatala_ml_compute_cluster_sums_f32(device const float* data [[buffer(0)]], device const uint* assignments [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* d [[buffer(4)]], device float* clusterSums [[buffer(5)]], device uint* clusterCounts [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint dVal = d[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint cluster = assignments[gid];
    atomic_fetch_add_explicit((device atomic_uint*)(&clusterCounts[cluster]), 1u, memory_order_relaxed);
    uint dataBase = (gid * dVal);
    uint sumBase = (cluster * dVal);
    for (int j = 0; j < (int)(dVal); ++j) {
      uint jU32 = ((uint)(j));
      uint dataIdx = (dataBase + jU32);
      uint sumIdx = (sumBase + jU32);
      float dataVal = data[dataIdx];
      atomic_fetch_add_explicit((device atomic_float*)(&clusterSums[sumIdx]), dataVal, memory_order_relaxed);
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_update_centroids_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_update_centroids_f32(device const float* clusterSums [[buffer(0)]], device const uint* clusterCounts [[buffer(1)]], device const float* oldCentroids [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* d [[buffer(4)]], device float* centroids [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint kVal = k[0];
  uint dVal = d[0];
  uint totalElems = (kVal * dVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint clusterIdx = (gid / dVal);
    uint count = clusterCounts[clusterIdx];
    float sumVal = clusterSums[gid];
    float oldVal = oldCentroids[gid];
    bool isEmpty = (count == 0u);
    float countF = ((float)(count));
    float newVal = (sumVal / countF);
    float result = ((isEmpty) ? (oldVal) : (newVal));
    centroids[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_inertia_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_inertia_f32(device const float* data [[buffer(0)]], device const float* centroids [[buffer(1)]], device const uint* assignments [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device const uint* d [[buffer(5)]], device float* inertia [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint nVal = n[0];
  uint dVal = d[0];
  bool inBounds = (gid < nVal);
  float pointDistAccum = as_type<float>(0x00000000u);
  float distSumAccum = as_type<float>(0x00000000u);
  if (inBounds) {
    uint cluster = assignments[gid];
    uint dataBase = (gid * dVal);
    uint centBase = (cluster * dVal);
    distSumAccum = as_type<float>(0x00000000u);
    for (int d = 0; d < (int)(dVal); ++d) {
      uint dU32 = ((uint)(d));
      uint dataIdx = (dataBase + dU32);
      uint centIdx = (centBase + dU32);
      float dataVal = data[dataIdx];
      float centVal = centroids[centIdx];
      float diff = (dataVal - centVal);
      float diffSq = (diff * diff);
      float currDistSum = distSumAccum;
      float newDistSum = (currDistSum + diffSq);
      distSumAccum = newDistSum;
    }
    float distSum = distSumAccum;
    pointDistAccum = distSum;
  }
  float pointDist = pointDistAccum;
  sdata[lid] = pointDist;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint inertiaF32RedStride = 128u;
  for (int inertiaF32RedStep = 0; inertiaF32RedStep < (int)(8); ++inertiaF32RedStep) {
    uint inertiaF32Stride = inertiaF32RedStride;
    if (lid < inertiaF32Stride) {
      float other = sdata[(lid + inertiaF32Stride)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    uint inertiaF32StrideToHalve = inertiaF32RedStride;
    uint inertiaF32NextStride = (inertiaF32StrideToHalve >> 1u);
    inertiaF32RedStride = inertiaF32NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    inertia[0] = sdata[0];
  }
}

)kernel";
const char* k_metal_navatala_ml_init_gmm_state_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_init_gmm_state_f32(device const float* initialMeans [[buffer(0)]], device const uint* n [[buffer(1)]], device const uint* k [[buffer(2)]], device const uint* d [[buffer(3)]], device float* means [[buffer(4)]], device float* variances [[buffer(5)]], device float* weights [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numK = k[0];
  uint numD = d[0];
  uint kd = (numK * numD);
  bool inMatrixBounds = (gid < kd);
  if (inMatrixBounds) {
    float meanVal = initialMeans[gid];
    means[gid] = meanVal;
    variances[gid] = as_type<float>(0x3f800000u);
  }
  bool inWeightBounds = (gid < numK);
  if (inWeightBounds) {
    float kFloat = ((float)(numK));
    float invK = (as_type<float>(0x3f800000u) / kFloat);
    weights[gid] = invK;
  }
}

)kernel";
const char* k_metal_navatala_ml_gaussian_log_pdf_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_gaussian_log_pdf_f32(device const float* data [[buffer(0)]], device const float* means [[buffer(1)]], device const float* variances [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device const uint* d [[buffer(5)]], device float* logPdf [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0];
  uint numK = k[0];
  uint numD = d[0];
  uint nk = (numN * numK);
  bool inBounds = (gid < nk);
  if (inBounds) {
    uint i = (gid / numK);
    uint j = (gid % numK);
    float log2Pi = as_type<float>(0x3feb3f8eu);
    float dFloat = ((float)(numD));
    float dLog2Pi = (dFloat * log2Pi);
    float logVarSumAccum = as_type<float>(0x00000000u);
    float mahalSumAccum = as_type<float>(0x00000000u);
    for (int l = 0; l < (int)(numD); ++l) {
      uint lU32 = ((uint)(l));
      uint dataIdx = ((i * numD) + lU32);
      uint paramIdx = ((j * numD) + lU32);
      float x_il = data[dataIdx];
      float mu_jl = means[paramIdx];
      float var_jl = variances[paramIdx];
      float logVar = log(var_jl);
      float currLogVarSum = logVarSumAccum;
      float newLogVarSum = (currLogVarSum + logVar);
      logVarSumAccum = newLogVarSum;
      float diff = (x_il - mu_jl);
      float diffSq = (diff * diff);
      float mahal = (diffSq / var_jl);
      float currMahalSum = mahalSumAccum;
      float newMahalSum = (currMahalSum + mahal);
      mahalSumAccum = newMahalSum;
    }
    float logVarSum = logVarSumAccum;
    float mahalSum = mahalSumAccum;
    float innerSum = (dLog2Pi + (logVarSum + mahalSum));
    float logPdfVal = (as_type<float>(0xbf000000u) * innerSum);
    logPdf[gid] = logPdfVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_responsibilities_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_responsibilities_f32(device const float* logPdf [[buffer(0)]], device const float* weights [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* k [[buffer(3)]], device float* responsibilities [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0];
  uint numK = k[0];
  uint i = gid;
  bool inBounds = (i < numN);
  if (inBounds) {
    float maxLogProbAccum = as_type<float>(0xfe967699u);
    float expSumAccum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(numK); ++j) {
      uint jU32 = ((uint)(j));
      uint logPdfIdx = ((i * numK) + jU32);
      float logP = logPdf[logPdfIdx];
      float w_j = weights[jU32];
      float logW = log(w_j);
      float logProb = (logW + logP);
      float currMax = maxLogProbAccum;
      bool isGreater = (logProb > currMax);
      float newMax = ((isGreater) ? (logProb) : (currMax));
      maxLogProbAccum = newMax;
    }
    float maxLogProb = maxLogProbAccum;
    for (int j = 0; j < (int)(numK); ++j) {
      uint jU32 = ((uint)(j));
      uint logPdfIdx = ((i * numK) + jU32);
      float logP = logPdf[logPdfIdx];
      float w_j = weights[jU32];
      float logW = log(w_j);
      float logProb = (logW + logP);
      float shifted = (logProb - maxLogProb);
      float expVal = exp(shifted);
      float currExpSum = expSumAccum;
      float newExpSum = (currExpSum + expVal);
      expSumAccum = newExpSum;
    }
    float expSum = expSumAccum;
    float logExpSum = log(expSum);
    float logSumExp = (maxLogProb + logExpSum);
    for (int j = 0; j < (int)(numK); ++j) {
      uint jU32 = ((uint)(j));
      uint logPdfIdx = ((i * numK) + jU32);
      float logP = logPdf[logPdfIdx];
      float w_j = weights[jU32];
      float logW = log(w_j);
      float logProb = (logW + logP);
      float logResp = (logProb - logSumExp);
      float resp = exp(logResp);
      responsibilities[logPdfIdx] = resp;
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_update_means_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_update_means_f32(device const float* data [[buffer(0)]], device const float* responsibilities [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* d [[buffer(4)]], device float* means [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0];
  uint numK = k[0];
  uint numD = d[0];
  uint kd = (numK * numD);
  bool inBounds = (gid < kd);
  if (inBounds) {
    uint j = (gid / numD);
    uint l = (gid % numD);
    float weightedSumAccum = as_type<float>(0x00000000u);
    float respSumAccum = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(numN); ++i) {
      uint iU32 = ((uint)(i));
      uint respIdx = ((iU32 * numK) + j);
      float r_ij = responsibilities[respIdx];
      uint dataIdx = ((iU32 * numD) + l);
      float x_il = data[dataIdx];
      float weighted = (r_ij * x_il);
      float currWeightedSum = weightedSumAccum;
      float newWeightedSum = (currWeightedSum + weighted);
      weightedSumAccum = newWeightedSum;
      float currRespSum = respSumAccum;
      float newRespSum = (currRespSum + r_ij);
      respSumAccum = newRespSum;
    }
    float weightedSum = weightedSumAccum;
    float respSum = respSumAccum;
    float meanVal = (weightedSum / respSum);
    means[gid] = meanVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_update_variances_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_update_variances_f32(device const float* data [[buffer(0)]], device const float* means [[buffer(1)]], device const float* responsibilities [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device const uint* d [[buffer(5)]], device float* variances [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0];
  uint numK = k[0];
  uint numD = d[0];
  uint kd = (numK * numD);
  bool inBounds = (gid < kd);
  if (inBounds) {
    uint j = (gid / numD);
    uint l = (gid % numD);
    float mu_jl = means[gid];
    float weightedSqSumAccum = as_type<float>(0x00000000u);
    float respSumAccum = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(numN); ++i) {
      uint iU32 = ((uint)(i));
      uint respIdx = ((iU32 * numK) + j);
      float r_ij = responsibilities[respIdx];
      uint dataIdx = ((iU32 * numD) + l);
      float x_il = data[dataIdx];
      float diff = (x_il - mu_jl);
      float diffSq = (diff * diff);
      float weightedSq = (r_ij * diffSq);
      float currWeightedSqSum = weightedSqSumAccum;
      float newWeightedSqSum = (currWeightedSqSum + weightedSq);
      weightedSqSumAccum = newWeightedSqSum;
      float currRespSum = respSumAccum;
      float newRespSum = (currRespSum + r_ij);
      respSumAccum = newRespSum;
    }
    float weightedSqSum = weightedSqSumAccum;
    float respSum = respSumAccum;
    float rawVar = (weightedSqSum / respSum);
    float varVal = (rawVar + as_type<float>(0x358637bdu));
    variances[gid] = varVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_update_weights_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_update_weights_f32(device const float* responsibilities [[buffer(0)]], device const uint* n [[buffer(1)]], device const uint* k [[buffer(2)]], device float* weights [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0];
  uint numK = k[0];
  uint j = gid;
  bool inBounds = (j < numK);
  if (inBounds) {
    float respSumAccum = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(numN); ++i) {
      uint iU32 = ((uint)(i));
      uint respIdx = ((iU32 * numK) + j);
      float r_ij = responsibilities[respIdx];
      float currRespSum = respSumAccum;
      float newRespSum = (currRespSum + r_ij);
      respSumAccum = newRespSum;
    }
    float respSum = respSumAccum;
    float nFloat = ((float)(numN));
    float weightVal = (respSum / nFloat);
    weights[j] = weightVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_log_likelihood_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_log_likelihood_f32(device const float* logPdf [[buffer(0)]], device const float* weights [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* k [[buffer(3)]], device float* logLikelihood [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  if (gid == 0u) {
    uint numN = n[0];
    uint numK = k[0];
    float totalLLAccum = as_type<float>(0x00000000u);
    float maxLogProbAccum = as_type<float>(0xfe967699u);
    float expSumAccum = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(numN); ++i) {
      uint iU32 = ((uint)(i));
      maxLogProbAccum = as_type<float>(0xfe967699u);
      for (int j = 0; j < (int)(numK); ++j) {
        uint jU32 = ((uint)(j));
        uint logPdfIdx = ((iU32 * numK) + jU32);
        float logP = logPdf[logPdfIdx];
        float w_j = weights[jU32];
        float logW = log(w_j);
        float logProb = (logW + logP);
        float currMax = maxLogProbAccum;
        bool isGreater = (logProb > currMax);
        float newMax = ((isGreater) ? (logProb) : (currMax));
        maxLogProbAccum = newMax;
      }
      float maxLogProb = maxLogProbAccum;
      expSumAccum = as_type<float>(0x00000000u);
      for (int j = 0; j < (int)(numK); ++j) {
        uint jU32 = ((uint)(j));
        uint logPdfIdx = ((iU32 * numK) + jU32);
        float logP = logPdf[logPdfIdx];
        float w_j = weights[jU32];
        float logW = log(w_j);
        float logProb = (logW + logP);
        float shifted = (logProb - maxLogProb);
        float expVal = exp(shifted);
        float currExpSum = expSumAccum;
        float newExpSum = (currExpSum + expVal);
        expSumAccum = newExpSum;
      }
      float expSum = expSumAccum;
      float logExpSum = log(expSum);
      float logSumExp = (maxLogProb + logExpSum);
      float currTotalLL = totalLLAccum;
      float newTotalLL = (currTotalLL + logSumExp);
      totalLLAccum = newTotalLL;
    }
    float totalLL = totalLLAccum;
    logLikelihood[0] = totalLL;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_gram_matrix_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_gram_matrix_f32(device const float* X [[buffer(0)]], device const uint* nSamples [[buffer(1)]], device const uint* nFeatures [[buffer(2)]], device float* G [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nSamples[0];
  uint p = nFeatures[0];
  uint totalElems = (p * p);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / p);
    uint j = (gid % p);
    float dotProduct = as_type<float>(0x00000000u);
    uint idx0_i = i;
    uint idx0_j = j;
    float x0_i = X[idx0_i];
    float x0_j = X[idx0_j];
    float prod0 = (x0_i * x0_j);
    float sum = prod0;
    G[gid] = sum;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_xt_y_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_xt_y_f32(device const float* X [[buffer(0)]], device const float* y [[buffer(1)]], device const uint* nSamples [[buffer(2)]], device const uint* nFeatures [[buffer(3)]], device float* Xty [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint wgid = ((uint)(int(__tgid.x)));
  threadgroup float sdata[256];
  uint n = nSamples[0];
  uint p = nFeatures[0];
  uint featureIdx = wgid;
  bool featureInBounds = (featureIdx < p);
  if (featureInBounds) {
    uint sampleIdx = lid;
    bool sampleInBounds = (sampleIdx < n);
    uint xIdx = ((sampleIdx * p) + featureIdx);
    if (sampleInBounds) {
      float xVal = X[xIdx];
      float yVal = y[sampleIdx];
      float product = (xVal * yVal);
      sdata[lid] = product;
    } else {
      sdata[lid] = as_type<float>(0x00000000u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint xtyF32Stride = 128u;
    for (int reductionStep1 = 0; reductionStep1 < (int)(8); ++reductionStep1) {
      uint stride1 = xtyF32Stride;
      if (lid < stride1) {
        float other = sdata[(lid + stride1)];
        float mine = sdata[lid];
        float sum = (mine + other);
        sdata[lid] = sum;
      }
      uint strideToHalve1 = xtyF32Stride;
      uint nextStride1 = (strideToHalve1 >> 1u);
      xtyF32Stride = nextStride1;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
      float totalSum = sdata[0];
      Xty[featureIdx] = totalSum;
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_predict_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_predict_f32(device const float* X [[buffer(0)]], device const float* beta [[buffer(1)]], device const uint* nSamples [[buffer(2)]], device const uint* nFeatures [[buffer(3)]], device float* yPred [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nSamples[0];
  uint p = nFeatures[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint sampleIdx = gid;
    uint baseIdx = (sampleIdx * p);
    float x0 = X[baseIdx];
    float beta0 = beta[0];
    float dot = (x0 * beta0);
    yPred[gid] = dot;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_residuals_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_residuals_f32(device const float* y [[buffer(0)]], device const float* yPred [[buffer(1)]], device const uint* nSamples [[buffer(2)]], device float* residuals [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nSamples[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    float yVal = y[gid];
    float yPredVal = yPred[gid];
    float residual = (yVal - yPredVal);
    residuals[gid] = residual;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_r_squared_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_r_squared_f32(device const float* y [[buffer(0)]], device const float* yPred [[buffer(1)]], device const float* yMean [[buffer(2)]], device const uint* nSamples [[buffer(3)]], device float* rSquared [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdataRes[256];
  threadgroup float sdataTot[256];
  uint n = nSamples[0];
  float meanVal = yMean[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    float yVal = y[gid];
    float yPredVal = yPred[gid];
    float resid = (yVal - yPredVal);
    float residSq = (resid * resid);
    float dev = (yVal - meanVal);
    float devSq = (dev * dev);
    sdataRes[lid] = residSq;
    sdataTot[lid] = devSq;
  } else {
    sdataRes[lid] = as_type<float>(0x00000000u);
    sdataTot[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint r2F32Stride = 128u;
  for (int reductionStep3 = 0; reductionStep3 < (int)(8); ++reductionStep3) {
    uint stride3 = r2F32Stride;
    if (lid < stride3) {
      float otherRes = sdataRes[(lid + stride3)];
      float mineRes = sdataRes[lid];
      float sumRes = (mineRes + otherRes);
      sdataRes[lid] = sumRes;
      float otherTot = sdataTot[(lid + stride3)];
      float mineTot = sdataTot[lid];
      float sumTot = (mineTot + otherTot);
      sdataTot[lid] = sumTot;
    }
    uint strideToHalve3 = r2F32Stride;
    uint nextStride3 = (strideToHalve3 >> 1u);
    r2F32Stride = nextStride3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float ssRes = sdataRes[0];
    float ssTot = sdataTot[0];
    float ratio = (ssRes / ssTot);
    float r2 = (as_type<float>(0x3f800000u) - ratio);
    rSquared[0] = r2;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_mean_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_mean_f32(device const float* y [[buffer(0)]], device const uint* nSamples [[buffer(1)]], device float* mean [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint n = nSamples[0];
  bool inBounds = (gid < n);
  float val = y[gid];
  if (inBounds) {
    sdata[lid] = val;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint meanF32Stride = 128u;
  for (int reductionStep5 = 0; reductionStep5 < (int)(8); ++reductionStep5) {
    uint stride5 = meanF32Stride;
    if (lid < stride5) {
      float other = sdata[(lid + stride5)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    uint strideToHalve5 = meanF32Stride;
    uint nextStride5 = (strideToHalve5 >> 1u);
    meanF32Stride = nextStride5;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float totalSum = sdata[0];
    float nFloat = ((float)(n));
    float meanVal = (totalSum / nFloat);
    mean[0] = meanVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_add_regularization_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_add_regularization_f32(device float* G [[buffer(0)]], device const float* lambda [[buffer(1)]], device const uint* p [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint pVal = p[0];
  float lambdaVal = lambda[0];
  bool inBounds = (gid < pVal);
  if (inBounds) {
    uint diagIdx = ((gid * pVal) + gid);
    float currentVal = G[diagIdx];
    float newVal = (currentVal + lambdaVal);
    G[diagIdx] = newVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_ridge_predict_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_ridge_predict_f32(device const float* X [[buffer(0)]], device const float* beta [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* p [[buffer(3)]], device float* yPred [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint wgid = ((uint)(int(__tgid.x)));
  threadgroup float sdata[256];
  uint nVal = n[0];
  uint pVal = p[0];
  uint rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    uint colIdx = lid;
    bool colInBounds = (colIdx < pVal);
    if (colInBounds) {
      uint elemIdx = ((rowIdx * pVal) + colIdx);
      float xVal = X[elemIdx];
      float betaVal = beta[colIdx];
      float product = (xVal * betaVal);
      sdata[lid] = product;
    } else {
      sdata[lid] = as_type<float>(0x00000000u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint pred32_reductionStride = 128u;
    for (int pred32_reductionStep = 0; pred32_reductionStep < (int)(8); ++pred32_reductionStep) {
      uint pred32_stride = pred32_reductionStride;
      if (lid < pred32_stride) {
        float pred32_other = sdata[(lid + pred32_stride)];
        float pred32_mine = sdata[lid];
        float pred32_sum = (pred32_mine + pred32_other);
        sdata[lid] = pred32_sum;
      }
      uint pred32_strideToHalve = pred32_reductionStride;
      uint pred32_nextStride = (pred32_strideToHalve >> 1u);
      pred32_reductionStride = pred32_nextStride;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
      float dotProduct = sdata[0];
      yPred[rowIdx] = dotProduct;
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_gcv_score_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_gcv_score_f32(device const float* y [[buffer(0)]], device const float* yPred [[buffer(1)]], device const float* traceH [[buffer(2)]], device const uint* n [[buffer(3)]], device float* gcv [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint nVal = n[0];
  float traceHVal = traceH[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    float yVal = y[gid];
    float yPredVal = yPred[gid];
    float residual = (yVal - yPredVal);
    float residualSq = (residual * residual);
    sdata[lid] = residualSq;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint gcv32_reductionStride = 128u;
  for (int gcv32_reductionStep = 0; gcv32_reductionStep < (int)(8); ++gcv32_reductionStep) {
    uint gcv32_stride = gcv32_reductionStride;
    if (lid < gcv32_stride) {
      float gcv32_other = sdata[(lid + gcv32_stride)];
      float gcv32_mine = sdata[lid];
      float gcv32_sum = (gcv32_mine + gcv32_other);
      sdata[lid] = gcv32_sum;
    }
    uint gcv32_strideToHalve = gcv32_reductionStride;
    uint gcv32_nextStride = (gcv32_strideToHalve >> 1u);
    gcv32_reductionStride = gcv32_nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float rss = sdata[0];
    float nFloat = ((float)(nVal));
    float traceHOverN = (traceHVal / nFloat);
    float oneMinusRatio = (as_type<float>(0x3f800000u) - traceHOverN);
    float denom = (oneMinusRatio * oneMinusRatio);
    float fullDenom = (nFloat * denom);
    float gcvVal = (rss / fullDenom);
    gcv[0] = gcvVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_effective_dof_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_effective_dof_f32(device const float* eigenvalues [[buffer(0)]], device const float* lambda [[buffer(1)]], device const uint* p [[buffer(2)]], device float* dof [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint pVal = p[0];
  float lambdaVal = lambda[0];
  bool inBounds = (gid < pVal);
  if (inBounds) {
    float eigVal = eigenvalues[gid];
    float denominator = (eigVal + lambdaVal);
    float contribution = (eigVal / denominator);
    sdata[lid] = contribution;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint dof32_reductionStride = 128u;
  for (int dof32_reductionStep = 0; dof32_reductionStep < (int)(8); ++dof32_reductionStep) {
    uint dof32_stride = dof32_reductionStride;
    if (lid < dof32_stride) {
      float dof32_other = sdata[(lid + dof32_stride)];
      float dof32_mine = sdata[lid];
      float dof32_sum = (dof32_mine + dof32_other);
      sdata[lid] = dof32_sum;
    }
    uint dof32_strideToHalve = dof32_reductionStride;
    uint dof32_nextStride = (dof32_strideToHalve >> 1u);
    dof32_reductionStride = dof32_nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float dofVal = sdata[0];
    dof[0] = dofVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_init_lasso_state_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_init_lasso_state_f32(device const float* y [[buffer(0)]], device const uint* nSamples [[buffer(1)]], device const uint* nFeatures [[buffer(2)]], device float* beta [[buffer(3)]], device float* residuals [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nSamples[0];
  uint d = nFeatures[0];
  bool betaInBounds = (gid < d);
  if (betaInBounds) {
    beta[gid] = as_type<float>(0x00000000u);
  }
  bool residInBounds = (gid < n);
  if (residInBounds) {
    float yVal = y[gid];
    residuals[gid] = yVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_partial_residual_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_partial_residual_f32(device const float* X [[buffer(0)]], device const float* beta [[buffer(1)]], device const float* y [[buffer(2)]], device const uint* j [[buffer(3)]], device const uint* nSamples [[buffer(4)]], device const uint* nFeatures [[buffer(5)]], device float* residuals [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint jVal = j[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    float yVal = y[gid];
    float betaJ = beta[jVal];
    uint rowOffset = (gid * d);
    uint xijIdx = (rowOffset + jVal);
    float xij = X[xijIdx];
    float xijBetaJ = (xij * betaJ);
    float dotAccum = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint kU32 = ((uint)(k));
      uint xikIdx = (rowOffset + kU32);
      float xik = X[xikIdx];
      float betak = beta[kU32];
      float prod = (xik * betak);
      float currentDot = dotAccum;
      float newDot = (currentDot + prod);
      dotAccum = newDot;
    }
    float dotProduct = dotAccum;
    float diff = (yVal - dotProduct);
    float residVal = (diff + xijBetaJ);
    residuals[gid] = residVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_soft_threshold_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_soft_threshold_f32(device const float* value [[buffer(0)]], device const float* threshold [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  bool inBounds = (gid == 0u);
  if (inBounds) {
    float v = value[0];
    float t = threshold[0];
    float absV = abs(v);
    float diff = (absV - t);
    float maxDiff = (((diff > as_type<float>(0x00000000u))) ? (diff) : (as_type<float>(0x00000000u)));
    bool positive = (v > as_type<float>(0x00000000u));
    bool negative = (v < as_type<float>(0x00000000u));
    float signVal = ((positive) ? (as_type<float>(0x3f800000u)) : (((negative) ? (as_type<float>(0xbf800000u)) : (as_type<float>(0x00000000u)))));
    float res = (signVal * maxDiff);
    result[0] = res;
  }
}

)kernel";
const char* k_metal_navatala_ml_coordinate_update_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_coordinate_update_f32(device const float* X [[buffer(0)]], device const float* residuals [[buffer(1)]], device const uint* j [[buffer(2)]], device const float* lambda [[buffer(3)]], device const uint* nSamples [[buffer(4)]], device const uint* nFeatures [[buffer(5)]], device float* newBetaJ [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdataDot[256];
  threadgroup float sdataNorm[256];
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint jVal = j[0];
  float lambdaVal = lambda[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint xijIdx = ((gid * d) + jVal);
    float xij = X[xijIdx];
    float ri = residuals[gid];
    float dotProd = (xij * ri);
    float xijSq = (xij * xij);
    sdataDot[lid] = dotProd;
    sdataNorm[lid] = xijSq;
  } else {
    sdataDot[lid] = as_type<float>(0x00000000u);
    sdataNorm[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint cu32RedStride = 128u;
  for (int cu32RedStep = 0; cu32RedStep < (int)(8); ++cu32RedStep) {
    uint cu32RedStrideVal = cu32RedStride;
    if (lid < cu32RedStrideVal) {
      float otherDot = sdataDot[(lid + cu32RedStrideVal)];
      float mineDot = sdataDot[lid];
      float sumDot = (mineDot + otherDot);
      sdataDot[lid] = sumDot;
      float otherNorm = sdataNorm[(lid + cu32RedStrideVal)];
      float mineNorm = sdataNorm[lid];
      float sumNorm = (mineNorm + otherNorm);
      sdataNorm[lid] = sumNorm;
    }
    uint cu32RedStrideToHalve = cu32RedStride;
    uint cu32RedNextStride = (cu32RedStrideToHalve >> 1u);
    cu32RedStride = cu32RedNextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float totalDot = sdataDot[0];
    float totalNorm = sdataNorm[0];
    float rho = (totalDot / totalNorm);
    float thresh = (lambdaVal / totalNorm);
    float absRho = abs(rho);
    float diff = (absRho - thresh);
    float maxDiff = (((diff > as_type<float>(0x00000000u))) ? (diff) : (as_type<float>(0x00000000u)));
    bool positive = (rho > as_type<float>(0x00000000u));
    bool negative = (rho < as_type<float>(0x00000000u));
    float signVal = ((positive) ? (as_type<float>(0x3f800000u)) : (((negative) ? (as_type<float>(0xbf800000u)) : (as_type<float>(0x00000000u)))));
    float newBeta = (signVal * maxDiff);
    newBetaJ[0] = newBeta;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_objective_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_objective_f32(device const float* X [[buffer(0)]], device const float* beta [[buffer(1)]], device const float* y [[buffer(2)]], device const float* lambda [[buffer(3)]], device const uint* nSamples [[buffer(4)]], device const uint* nFeatures [[buffer(5)]], device float* objective [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdataMse[256];
  threadgroup float sdataL1[256];
  uint n = nSamples[0];
  uint d = nFeatures[0];
  float lambdaVal = lambda[0];
  bool sampleInBounds = (gid < n);
  if (sampleInBounds) {
    float yVal = y[gid];
    uint rowOffset = (gid * d);
    float dotAccum = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint kU32 = ((uint)(k));
      uint xikIdx = (rowOffset + kU32);
      float xik = X[xikIdx];
      float betak = beta[kU32];
      float prod = (xik * betak);
      float currentDot = dotAccum;
      float newDot = (currentDot + prod);
      dotAccum = newDot;
    }
    float dotProduct = dotAccum;
    float residual = (yVal - dotProduct);
    float residSq = (residual * residual);
    sdataMse[lid] = residSq;
  } else {
    sdataMse[lid] = as_type<float>(0x00000000u);
  }
  bool featureInBounds = (gid < d);
  if (featureInBounds) {
    float betaJ = beta[gid];
    float absBetaJ = abs(betaJ);
    sdataL1[lid] = absBetaJ;
  } else {
    sdataL1[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint obj32RedStride = 128u;
  for (int obj32RedStep = 0; obj32RedStep < (int)(8); ++obj32RedStep) {
    uint obj32RedStrideVal = obj32RedStride;
    if (lid < obj32RedStrideVal) {
      float otherMse = sdataMse[(lid + obj32RedStrideVal)];
      float mineMse = sdataMse[lid];
      float sumMse = (mineMse + otherMse);
      sdataMse[lid] = sumMse;
      float otherL1 = sdataL1[(lid + obj32RedStrideVal)];
      float mineL1 = sdataL1[lid];
      float sumL1 = (mineL1 + otherL1);
      sdataL1[lid] = sumL1;
    }
    uint obj32RedStrideToHalve = obj32RedStride;
    uint obj32RedNextStride = (obj32RedStrideToHalve >> 1u);
    obj32RedStride = obj32RedNextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float totalMse = sdataMse[0];
    float totalL1 = sdataL1[0];
    float halfMse = (as_type<float>(0x3f000000u) * totalMse);
    float lambdaL1 = (lambdaVal * totalL1);
    float obj = (halfMse + lambdaL1);
    objective[0] = obj;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_active_set_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_active_set_f32(device const float* beta [[buffer(0)]], device const uint* nFeatures [[buffer(1)]], device uint* activeMask [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint d = nFeatures[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    float betaJ = beta[gid];
    float absBetaJ = abs(betaJ);
    bool isActive = (absBetaJ > as_type<float>(0x00000000u));
    uint maskVal = ((isActive) ? (1u) : (0u));
    activeMask[gid] = maskVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_center_data_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_center_data_f32(device const float* _input [[buffer(0)]], device const uint* nSamples [[buffer(1)]], device const uint* nFeatures [[buffer(2)]], device const float* means [[buffer(3)]], device float* _output [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint totalElems = (n * d);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint colIdx = (gid % d);
    float x = _input[gid];
    float colMean = means[colIdx];
    float centered = (x - colMean);
    _output[gid] = centered;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_covariance_matrix_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_covariance_matrix_f32(device const float* Xc [[buffer(0)]], device const uint* nSamples [[buffer(1)]], device const uint* nFeatures [[buffer(2)]], device float* Cov [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint covSize = (d * d);
  bool inBounds = (gid < covSize);
  if (inBounds) {
    uint i = (gid / d);
    uint j = (gid % d);
    float sum = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(n); ++k) {
      uint kU32 = ((uint)(k));
      uint idxI = ((kU32 * d) + i);
      uint idxJ = ((kU32 * d) + j);
      float xi = Xc[idxI];
      float xj = Xc[idxJ];
      float prod = (xi * xj);
      float currSum = sum;
      float newSum = (currSum + prod);
      sum = newSum;
    }
    float finalSum = sum;
    uint nMinus1 = (n - 1u);
    float divisor = ((float)(nMinus1));
    float covVal = (finalSum / divisor);
    Cov[gid] = covVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_project_to_principal_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_project_to_principal_f32(device const float* X [[buffer(0)]], device const float* V [[buffer(1)]], device const uint* nSamples [[buffer(2)]], device const uint* nFeatures [[buffer(3)]], device const uint* nComponents [[buffer(4)]], device float* Z [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint kComp = nComponents[0];
  uint totalOutput = (n * kComp);
  bool inBounds = (gid < totalOutput);
  if (inBounds) {
    uint i = (gid / kComp);
    uint c = (gid % kComp);
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(d); ++j) {
      uint jU32 = ((uint)(j));
      uint xIdx = ((i * d) + jU32);
      uint vIdx = ((c * d) + jU32);
      float xVal = X[xIdx];
      float vVal = V[vIdx];
      float prod = (xVal * vVal);
      float currSum = sum;
      float newSum = (currSum + prod);
      sum = newSum;
    }
    float finalSum = sum;
    Z[gid] = finalSum;
  }
}

)kernel";
const char* k_metal_navatala_ml_explained_variance_ratio_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_explained_variance_ratio_f32(device const float* eigenvalues [[buffer(0)]], device const uint* nFeatures [[buffer(1)]], device const uint* nComponents [[buffer(2)]], device float* ratios [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint d = nFeatures[0];
  uint kComp = nComponents[0];
  bool inBoundsLoad = (gid < d);
  if (inBoundsLoad) {
    float eigVal = eigenvalues[gid];
    sdata[lid] = eigVal;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint varRatioF32RedStride = 128u;
  for (int varRatioF32RedStep = 0; varRatioF32RedStep < (int)(8); ++varRatioF32RedStep) {
    uint varRatioF32Stride = varRatioF32RedStride;
    if (lid < varRatioF32Stride) {
      float other = sdata[(lid + varRatioF32Stride)];
      float mine = sdata[lid];
      float sumVal = (mine + other);
      sdata[lid] = sumVal;
    }
    uint varRatioF32StrideToHalve = varRatioF32RedStride;
    uint varRatioF32NextStride = (varRatioF32StrideToHalve >> 1u);
    varRatioF32RedStride = varRatioF32NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float totalVariance = sdata[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool inBoundsRatio = (gid < kComp);
  if (inBoundsRatio) {
    float eigVal = eigenvalues[gid];
    float ratio = (eigVal / totalVariance);
    ratios[gid] = ratio;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_pairwise_affinity_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_pairwise_affinity_f32(device const float* X [[buffer(0)]], device const float* sigma [[buffer(1)]], device const uint* nPoints [[buffer(2)]], device const uint* nDims [[buffer(3)]], device float* P [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  uint d = nDims[0];
  uint totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint i = (gid / n);
    uint j = (gid % n);
    float sigmaI = sigma[i];
    float twoSigmaSq = (as_type<float>(0x40000000u) * (sigmaI * sigmaI));
    uint baseI = (i * d);
    uint baseJ = (j * d);
    float distSqAccum = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint kU32 = ((uint)(k));
      uint idxI = (baseI + kU32);
      uint idxJ = (baseJ + kU32);
      float xi_k = X[idxI];
      float xj_k = X[idxJ];
      float diff = (xi_k - xj_k);
      float diffSq = (diff * diff);
      float currentDistSq = distSqAccum;
      float newDistSq = (currentDistSq + diffSq);
      distSqAccum = newDistSq;
    }
    float distSq = distSqAccum;
    bool isDiag = (i == j);
    float negDistOverSigma = ((as_type<float>(0x00000000u) - distSq) / twoSigmaSq);
    float pVal = exp(negDistOverSigma);
    if (isDiag) {
      P[gid] = as_type<float>(0x00000000u);
    } else {
      P[gid] = pVal;
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_symmetrize_affinity_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_symmetrize_affinity_f32(device float* P [[buffer(0)]], device const uint* nPoints [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  uint totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint i = (gid / n);
    uint j = (gid % n);
    bool isUpper = (i < j);
    if (isUpper) {
      uint idxIJ = gid;
      uint idxJI = ((j * n) + i);
      float pij = P[idxIJ];
      float pji = P[idxJI];
      float sumP = (pij + pji);
      float twoN = (as_type<float>(0x40000000u) * ((float)(n)));
      float symVal = (sumP / twoN);
      P[idxIJ] = symVal;
      P[idxJI] = symVal;
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_init_embedding_random_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_init_embedding_random_f32(device const uint* nPoints [[buffer(0)]], device const uint* outputDims [[buffer(1)]], device const uint* seed [[buffer(2)]], device float* Y [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  uint d = outputDims[0];
  uint s = seed[0];
  uint totalElems = (n * d);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint state = (s + gid);
    uint lcgA = 48271u;
    uint state1 = (lcgA * state);
    uint state2 = (lcgA * state1);
    float maxVal = as_type<float>(0x4f000000u);
    float u1 = (((float)(state1)) / maxVal);
    float u2 = (((float)(state2)) / maxVal);
    float negTwoLnU1 = (as_type<float>(0xc0000000u) * log(u1));
    float sqrtTerm = sqrt(negTwoLnU1);
    float twoPiU2 = (as_type<float>(0x40c90fdbu) * u2);
    float z = (sqrtTerm * cos(twoPiU2));
    float yVal = (z * as_type<float>(0x3c23d70au));
    Y[gid] = yVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_q_distribution_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_q_distribution_f32(device const float* Y [[buffer(0)]], device const uint* nPoints [[buffer(1)]], device const uint* outputDims [[buffer(2)]], device float* Q [[buffer(3)]], device float* sumQ [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  threadgroup float sdata[256];
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint n = nPoints[0];
  uint d = outputDims[0];
  uint totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  float localQ = as_type<float>(0x00000000u);
  if (inBounds) {
    uint i = (gid / n);
    uint j = (gid % n);
    uint baseI = (i * d);
    uint baseJ = (j * d);
    float distSqAccum = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint kU32 = ((uint)(k));
      uint idxI = (baseI + kU32);
      uint idxJ = (baseJ + kU32);
      float yi_k = Y[idxI];
      float yj_k = Y[idxJ];
      float diff = (yi_k - yj_k);
      float diffSq = (diff * diff);
      float currentDistSq = distSqAccum;
      float newDistSq = (currentDistSq + diffSq);
      distSqAccum = newDistSq;
    }
    float distSq = distSqAccum;
    bool isDiag = (i == j);
    float onePlusDistSq = (as_type<float>(0x3f800000u) + distSq);
    float qVal = (as_type<float>(0x3f800000u) / onePlusDistSq);
    if (isDiag) {
      Q[gid] = as_type<float>(0x00000000u);
      sdata[lid] = as_type<float>(0x00000000u);
    } else {
      Q[gid] = qVal;
      sdata[lid] = qVal;
    }
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint qF32ReductionStride = 128u;
  for (int qF32ReductionStep = 0; qF32ReductionStep < (int)(8); ++qF32ReductionStep) {
    uint qF32Stride = qF32ReductionStride;
    if (lid < qF32Stride) {
      float qF32Other = sdata[(lid + qF32Stride)];
      float qF32Mine = sdata[lid];
      float qF32Sum = (qF32Mine + qF32Other);
      sdata[lid] = qF32Sum;
    }
    uint qF32StrideToHalve = qF32ReductionStride;
    uint qF32NextStride = (qF32StrideToHalve >> 1u);
    qF32ReductionStride = qF32NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float partialSum = sdata[0];
    float oldSum = sumQ[0];
    float newSum = (oldSum + partialSum);
    sumQ[0] = newSum;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_gradient_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_gradient_f32(device const float* P [[buffer(0)]], device const float* Q [[buffer(1)]], device const float* Y [[buffer(2)]], device const uint* nPoints [[buffer(3)]], device const uint* outputDims [[buffer(4)]], device float* grad [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  uint d = outputDims[0];
  uint totalGradElems = (n * d);
  bool inBounds = (gid < totalGradElems);
  if (inBounds) {
    uint i = (gid / d);
    uint k = (gid % d);
    float yi_k = Y[gid];
    float gradSumAccum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(n); ++j) {
      uint jU32 = ((uint)(j));
      bool notDiag = (i != jU32);
      if (notDiag) {
        uint pIdx = ((i * n) + jU32);
        float pij = P[pIdx];
        float qij = Q[pIdx];
        uint yjIdx = ((jU32 * d) + k);
        float yj_k = Y[yjIdx];
        uint baseI = (i * d);
        uint baseJ = (jU32 * d);
        float distSqAccum = as_type<float>(0x00000000u);
        for (int m = 0; m < (int)(d); ++m) {
          uint mU32 = ((uint)(m));
          uint idxIM = (baseI + mU32);
          uint idxJM = (baseJ + mU32);
          float yi_m = Y[idxIM];
          float yj_m = Y[idxJM];
          float diffM = (yi_m - yj_m);
          float diffSqM = (diffM * diffM);
          float currentDistSq = distSqAccum;
          float newDistSq = (currentDistSq + diffSqM);
          distSqAccum = newDistSq;
        }
        float distSq = distSqAccum;
        float invDist = (as_type<float>(0x3f800000u) / (as_type<float>(0x3f800000u) + distSq));
        float diffK = (yi_k - yj_k);
        float pMinusQ = (pij - qij);
        float term = (pMinusQ * (diffK * invDist));
        float currentGradSum = gradSumAccum;
        float newGradSum = (currentGradSum + term);
        gradSumAccum = newGradSum;
      }
    }
    float gradSum = gradSumAccum;
    float gradVal = (as_type<float>(0x40800000u) * gradSum);
    grad[gid] = gradVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_apply_momentum_update_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_apply_momentum_update_f32(device const float* Y [[buffer(0)]], device const float* Y_prev [[buffer(1)]], device const float* grad [[buffer(2)]], device const float* lr [[buffer(3)]], device const float* momentum [[buffer(4)]], device const uint* nPoints [[buffer(5)]], device const uint* outputDims [[buffer(6)]], device float* Y_new [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  uint d = outputDims[0];
  uint totalElems = (n * d);
  float lrVal = lr[0];
  float momVal = momentum[0];
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    float y = Y[gid];
    float yPrev = Y_prev[gid];
    float g = grad[gid];
    float lrGrad = (lrVal * g);
    float yDiff = (y - yPrev);
    float momTerm = (momVal * yDiff);
    float yNew = (y + (lrGrad + momTerm));
    Y_new[gid] = yNew;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_k_l_divergence_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_k_l_divergence_f32(device const float* P [[buffer(0)]], device const float* Q [[buffer(1)]], device const uint* nPoints [[buffer(2)]], device float* kl_divergence [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  threadgroup float sdata[256];
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint n = nPoints[0];
  uint totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    float pij = P[gid];
    float qij = Q[gid];
    float minVal = as_type<float>(0x33d6bf95u);
    bool pSafe = (pij > minVal);
    bool qSafe = (qij > minVal);
    bool bothSafe = (pSafe && qSafe);
    if (bothSafe) {
      float ratio = (pij / qij);
      float logRatio = log(ratio);
      float klTerm = (pij * logRatio);
      sdata[lid] = klTerm;
    } else {
      sdata[lid] = as_type<float>(0x00000000u);
    }
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint klF32ReductionStride = 128u;
  for (int klF32ReductionStep = 0; klF32ReductionStep < (int)(8); ++klF32ReductionStep) {
    uint klF32Stride = klF32ReductionStride;
    if (lid < klF32Stride) {
      float klF32Other = sdata[(lid + klF32Stride)];
      float klF32Mine = sdata[lid];
      float klF32Sum = (klF32Mine + klF32Other);
      sdata[lid] = klF32Sum;
    }
    uint klF32StrideToHalve = klF32ReductionStride;
    uint klF32NextStride = (klF32StrideToHalve >> 1u);
    klF32ReductionStride = klF32NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float klVal = sdata[0];
    kl_divergence[0] = klVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_perplexity_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_perplexity_f32(device const float* distances [[buffer(0)]], device const float* targetPerplexity [[buffer(1)]], device const uint* nPoints [[buffer(2)]], device const uint* maxIter [[buffer(3)]], device float* sigma [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  float perpTarget = targetPerplexity[0];
  uint maxIt = maxIter[0];
  float logPerpTarget = log(perpTarget);
  bool inBounds = (gid < n);
  if (inBounds) {
    uint i = gid;
    float sigmaLowAccum = as_type<float>(0x38d1b717u);
    float sigmaHighAccum = as_type<float>(0x461c4000u);
    float sigmaMidAccum = as_type<float>(0x3f800000u);
    for (int iter = 0; iter < (int)(maxIt); ++iter) {
      uint iterU32 = ((uint)(iter));
      float sigmaLow = sigmaLowAccum;
      float sigmaHigh = sigmaHighAccum;
      float sigmaMid = (as_type<float>(0x3f000000u) * (sigmaLow + sigmaHigh));
      sigmaMidAccum = sigmaMid;
      float twoSigmaSq = (as_type<float>(0x40000000u) * (sigmaMid * sigmaMid));
      float sumPAccum = as_type<float>(0x00000000u);
      float entropyAccum = as_type<float>(0x00000000u);
      for (int j = 0; j < (int)(n); ++j) {
        uint jU32 = ((uint)(j));
        bool notDiag = (i != jU32);
        if (notDiag) {
          uint distIdx = ((i * n) + jU32);
          float dij = distances[distIdx];
          float negDOverSigma = ((as_type<float>(0x00000000u) - dij) / twoSigmaSq);
          float pVal = exp(negDOverSigma);
          float currentSumP = sumPAccum;
          float newSumP = (currentSumP + pVal);
          sumPAccum = newSumP;
        }
      }
      float sumP = sumPAccum;
      for (int j = 0; j < (int)(n); ++j) {
        uint jU32 = ((uint)(j));
        bool notDiag = (i != jU32);
        if (notDiag) {
          uint distIdx = ((i * n) + jU32);
          float dij = distances[distIdx];
          float negDOverSigma = ((as_type<float>(0x00000000u) - dij) / twoSigmaSq);
          float pVal = exp(negDOverSigma);
          float pNorm = (pVal / sumP);
          bool pGtZero = (pNorm > as_type<float>(0x33d6bf95u));
          if (pGtZero) {
            float logP = log(pNorm);
            float entropyTerm = ((as_type<float>(0x00000000u) - pNorm) * logP);
            float currentEntropy = entropyAccum;
            float newEntropy = (currentEntropy + entropyTerm);
            entropyAccum = newEntropy;
          }
        }
      }
      float entropy = entropyAccum;
      bool entropyTooHigh = (entropy > logPerpTarget);
      if (entropyTooHigh) {
        sigmaHighAccum = sigmaMid;
      } else {
        sigmaLowAccum = sigmaMid;
      }
    }
    float finalSigma = sigmaMidAccum;
    sigma[gid] = finalSigma;
  }
}

)kernel";
const char* k_metal_navatala_ml_normalize_embedding_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_normalize_embedding_f32(device float* Y [[buffer(0)]], device const uint* nPoints [[buffer(1)]], device const uint* outputDims [[buffer(2)]], device const float* means [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  uint d = outputDims[0];
  uint totalElems = (n * d);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint k = (gid % d);
    float yVal = Y[gid];
    float meanK = means[k];
    float centered = (yVal - meanK);
    Y[gid] = centered;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_exaggerated_p_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_exaggerated_p_f32(device float* P [[buffer(0)]], device const float* alpha [[buffer(1)]], device const uint* nPoints [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  float alphaVal = alpha[0];
  uint totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    float pVal = P[gid];
    float pExag = (pVal * alphaVal);
    P[gid] = pExag;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_neighbor_counts_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_neighbor_counts_f32(device const float* points [[buffer(0)]], device const float* epsilon [[buffer(1)]], device const uint* nPoints [[buffer(2)]], device const uint* nDims [[buffer(3)]], device uint* counts [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  uint d = nDims[0];
  float eps = epsilon[0];
  float epsSq = (eps * eps);
  bool inBounds = (gid < n);
  if (inBounds) {
    uint iBase = (gid * d);
    uint countAccum = 0u;
    for (int j = 0; j < (int)(n); ++j) {
      uint jU32 = ((uint)(j));
      uint jBase = (jU32 * d);
      float distSqAccum = as_type<float>(0x00000000u);
      for (int k = 0; k < (int)(d); ++k) {
        uint kU32 = ((uint)(k));
        uint iIdx = (iBase + kU32);
        uint jIdx = (jBase + kU32);
        float iVal = points[iIdx];
        float jVal = points[jIdx];
        float diff = (iVal - jVal);
        float diffSq = (diff * diff);
        float currentDistSq = distSqAccum;
        float newDistSq = (currentDistSq + diffSq);
        distSqAccum = newDistSq;
      }
      float finalDistSq = distSqAccum;
      if (finalDistSq < epsSq) {
        uint currentCount = countAccum;
        uint newCount = (currentCount + 1u);
        countAccum = newCount;
      }
    }
    uint finalCount = countAccum;
    counts[gid] = finalCount;
  }
}

)kernel";
const char* k_metal_navatala_ml_identify_core_points_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_identify_core_points_f32(device const uint* counts [[buffer(0)]], device const uint* minPts [[buffer(1)]], device const uint* nPoints [[buffer(2)]], device uint* isCore [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  uint minP = minPts[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint cnt = counts[gid];
    bool isCoreVal = (cnt >= minP);
    uint result = ((isCoreVal) ? (1u) : (0u));
    isCore[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_ml_identify_core_points_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_identify_core_points_f64(device const uint* counts [[buffer(0)]], device const uint* minPts [[buffer(1)]], device const uint* nPoints [[buffer(2)]], device uint* isCore [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  uint minP = minPts[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint cnt = counts[gid];
    bool isCoreVal = (cnt >= minP);
    uint result = ((isCoreVal) ? (1u) : (0u));
    isCore[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_ml_init_cluster_labels_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_init_cluster_labels_f32(device const uint* nPoints [[buffer(0)]], device int* labels [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    labels[gid] = -1;
  }
}

)kernel";
const char* k_metal_navatala_ml_init_cluster_labels_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_init_cluster_labels_f64(device const uint* nPoints [[buffer(0)]], device int* labels [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    labels[gid] = -1;
  }
}

)kernel";
const char* k_metal_navatala_ml_expand_cluster_step_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_expand_cluster_step_f32(device const float* points [[buffer(0)]], device const int* labelsIn [[buffer(1)]], device const uint* isCore [[buffer(2)]], device const float* epsilon [[buffer(3)]], device const uint* nPoints [[buffer(4)]], device const uint* nDims [[buffer(5)]], device int* labelsOut [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  uint d = nDims[0];
  float eps = epsilon[0];
  float epsSq = (eps * eps);
  bool inBounds = (gid < n);
  if (inBounds) {
    int myLabel = labelsIn[gid];
    int newLabelAccum = myLabel;
    bool isUnlabeled = (myLabel == -1);
    if (isUnlabeled) {
      uint iBase = (gid * d);
      for (int j = 0; j < (int)(n); ++j) {
        uint jU32 = ((uint)(j));
        uint jIsCore = isCore[jU32];
        int jLabel = labelsIn[jU32];
        bool jHasLabel = (jLabel >= 0);
        bool jIsCoreVal = (jIsCore == 1u);
        bool jIsLabeledCore = (jIsCoreVal && jHasLabel);
        if (jIsLabeledCore) {
          uint jBase = (jU32 * d);
          float distSqAccum = as_type<float>(0x00000000u);
          for (int k = 0; k < (int)(d); ++k) {
            uint kU32 = ((uint)(k));
            uint iIdx = (iBase + kU32);
            uint jIdx = (jBase + kU32);
            float iVal = points[iIdx];
            float jVal = points[jIdx];
            float diff = (iVal - jVal);
            float diffSq = (diff * diff);
            float currentDistSq = distSqAccum;
            float newDistSq = (currentDistSq + diffSq);
            distSqAccum = newDistSq;
          }
          float finalDistSq = distSqAccum;
          if (finalDistSq < epsSq) {
            newLabelAccum = jLabel;
          }
        }
      }
    }
    int finalNewLabel = newLabelAccum;
    labelsOut[gid] = finalNewLabel;
  }
}

)kernel";
const char* k_metal_navatala_ml_propagate_core_labels_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_propagate_core_labels_f32(device const float* points [[buffer(0)]], device const int* labelsIn [[buffer(1)]], device const uint* isCore [[buffer(2)]], device const float* epsilon [[buffer(3)]], device const uint* nPoints [[buffer(4)]], device const uint* nDims [[buffer(5)]], device int* labelsOut [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  uint d = nDims[0];
  float eps = epsilon[0];
  float epsSq = (eps * eps);
  bool inBounds = (gid < n);
  if (inBounds) {
    int myLabel = labelsIn[gid];
    uint myIsCore = isCore[gid];
    int newLabelAccum = myLabel;
    bool notCore = (myIsCore == 0u);
    bool isUnlabeled = (myLabel == -1);
    bool shouldProcess = (notCore && isUnlabeled);
    if (shouldProcess) {
      uint iBase = (gid * d);
      float bestDistAccum = as_type<float>(0x7f7fffffu);
      for (int j = 0; j < (int)(n); ++j) {
        uint jU32 = ((uint)(j));
        uint jIsCore = isCore[jU32];
        int jLabel = labelsIn[jU32];
        bool jIsCoreVal = (jIsCore == 1u);
        bool jHasLabel = (jLabel >= 0);
        bool jIsLabeledCore = (jIsCoreVal && jHasLabel);
        if (jIsLabeledCore) {
          uint jBase = (jU32 * d);
          float distSqAccum = as_type<float>(0x00000000u);
          for (int k = 0; k < (int)(d); ++k) {
            uint kU32 = ((uint)(k));
            uint iIdx = (iBase + kU32);
            uint jIdx = (jBase + kU32);
            float iVal = points[iIdx];
            float jVal = points[jIdx];
            float diff = (iVal - jVal);
            float diffSq = (diff * diff);
            float currentDistSq = distSqAccum;
            float newDistSq = (currentDistSq + diffSq);
            distSqAccum = newDistSq;
          }
          float finalDistSq = distSqAccum;
          float currentBestDist = bestDistAccum;
          bool withinEps = (finalDistSq < epsSq);
          bool isBetter = (finalDistSq < currentBestDist);
          bool shouldUpdate = (withinEps && isBetter);
          if (shouldUpdate) {
            bestDistAccum = finalDistSq;
            newLabelAccum = jLabel;
          }
        }
      }
    }
    int finalNewLabel = newLabelAccum;
    labelsOut[gid] = finalNewLabel;
  }
}

)kernel";
const char* k_metal_navatala_ml_mark_noise_points_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_mark_noise_points_f32(device const int* labelsIn [[buffer(0)]], device const uint* isCore [[buffer(1)]], device const uint* nPoints [[buffer(2)]], device int* labelsOut [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    int myLabel = labelsIn[gid];
    bool isUnlabeled = (myLabel == -1);
    int newLabel = ((isUnlabeled) ? (-2) : (myLabel));
    labelsOut[gid] = newLabel;
  }
}

)kernel";
const char* k_metal_navatala_ml_mark_noise_points_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_mark_noise_points_f64(device const int* labelsIn [[buffer(0)]], device const uint* isCore [[buffer(1)]], device const uint* nPoints [[buffer(2)]], device int* labelsOut [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    int myLabel = labelsIn[gid];
    bool isUnlabeled = (myLabel == -1);
    int newLabel = ((isUnlabeled) ? (-2) : (myLabel));
    labelsOut[gid] = newLabel;
  }
}

)kernel";
const char* k_metal_navatala_ml_count_clusters_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_count_clusters_f32(device const int* labels [[buffer(0)]], device const uint* nPoints [[buffer(1)]], device int* numClusters [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup int sdata[256];
  uint n = nPoints[0];
  bool inBounds = (gid < n);
  int myLabel = ((inBounds) ? (labels[gid]) : (-1));
  sdata[lid] = myLabel;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint countClstF32RedStride = 128u;
  for (int countClstF32RedStep = 0; countClstF32RedStep < (int)(8); ++countClstF32RedStep) {
    uint countClstF32Stride = countClstF32RedStride;
    if (lid < countClstF32Stride) {
      uint otherIdx = (lid + countClstF32Stride);
      int other = sdata[otherIdx];
      int mine = sdata[lid];
      bool isGreater = (other > mine);
      int maxVal = ((isGreater) ? (other) : (mine));
      sdata[lid] = maxVal;
    }
    uint countClstF32StrideToHalve = countClstF32RedStride;
    uint countClstF32NextStride = (countClstF32StrideToHalve >> 1u);
    countClstF32RedStride = countClstF32NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    int maxLabel = sdata[0];
    int count = (maxLabel + 1);
    numClusters[0] = count;
  }
}

)kernel";
const char* k_metal_navatala_ml_count_clusters_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_count_clusters_f64(device const int* labels [[buffer(0)]], device const uint* nPoints [[buffer(1)]], device int* numClusters [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup int sdata[256];
  uint n = nPoints[0];
  bool inBounds = (gid < n);
  int myLabel = ((inBounds) ? (labels[gid]) : (-1));
  sdata[lid] = myLabel;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint countClstF64RedStride = 128u;
  for (int countClstF64RedStep = 0; countClstF64RedStep < (int)(8); ++countClstF64RedStep) {
    uint countClstF64Stride = countClstF64RedStride;
    if (lid < countClstF64Stride) {
      uint otherIdx = (lid + countClstF64Stride);
      int other = sdata[otherIdx];
      int mine = sdata[lid];
      bool isGreater = (other > mine);
      int maxVal = ((isGreater) ? (other) : (mine));
      sdata[lid] = maxVal;
    }
    uint countClstF64StrideToHalve = countClstF64RedStride;
    uint countClstF64NextStride = (countClstF64StrideToHalve >> 1u);
    countClstF64RedStride = countClstF64NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    int maxLabel = sdata[0];
    int count = (maxLabel + 1);
    numClusters[0] = count;
  }
}

)kernel";
const char* k_metal_navatala_ml_neighborhood_recall_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_neighborhood_recall_f32(device const uint* origNeighbors [[buffer(0)]], device const uint* embedNeighbors [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* k [[buffer(3)]], device float* recall [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0u];
  uint kVal = k[0u];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint rowBase = (gid * kVal);
    uint intersectCountAccum = 0u;
    uint foundAccum = 0u;
    for (int origIdx = 0; origIdx < (int)(kVal); ++origIdx) {
      uint origIdxU32 = ((uint)(origIdx));
      uint origNeighborIdx = (rowBase + origIdxU32);
      uint origNeighbor = origNeighbors[origNeighborIdx];
      foundAccum = 0u;
      for (int embedIdx = 0; embedIdx < (int)(kVal); ++embedIdx) {
        uint embedIdxU32 = ((uint)(embedIdx));
        uint embedNeighborIdx = (rowBase + embedIdxU32);
        uint embedNeighbor = embedNeighbors[embedNeighborIdx];
        bool isMatch = (origNeighbor == embedNeighbor);
        uint currentFound = foundAccum;
        uint matchAsU32 = ((isMatch) ? (1u) : (0u));
        uint newFound = (currentFound | matchAsU32);
        foundAccum = newFound;
      }
      uint finalFound = foundAccum;
      bool wasFound = (finalFound > 0u);
      if (wasFound) {
        uint currentCount = intersectCountAccum;
        uint newCount = (currentCount + 1u);
        intersectCountAccum = newCount;
      }
    }
    uint finalIntersectCount = intersectCountAccum;
    float kFloat = ((float)(kVal));
    float countFloat = ((float)(finalIntersectCount));
    float recallVal = (countFloat / kFloat);
    recall[gid] = recallVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_trustworthiness_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_trustworthiness_f32(device const uint* origNeighbors [[buffer(0)]], device const uint* embedNeighbors [[buffer(1)]], device const uint* origRanks [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device float* penalty [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0u];
  uint kVal = k[0u];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint kRowBase = (gid * kVal);
    uint nRowBase = (gid * nVal);
    float penaltySumAccum = as_type<float>(0x00000000u);
    uint inOriginalAccum = 0u;
    for (int embedIdx = 0; embedIdx < (int)(kVal); ++embedIdx) {
      uint embedIdxU32 = ((uint)(embedIdx));
      uint embedNeighborIdx = (kRowBase + embedIdxU32);
      uint embedNeighbor = embedNeighbors[embedNeighborIdx];
      inOriginalAccum = 0u;
      for (int origIdx = 0; origIdx < (int)(kVal); ++origIdx) {
        uint origIdxU32 = ((uint)(origIdx));
        uint origNeighborIdx = (kRowBase + origIdxU32);
        uint origNeighbor = origNeighbors[origNeighborIdx];
        bool isMatch = (embedNeighbor == origNeighbor);
        uint currentInOrig = inOriginalAccum;
        uint matchAsU32 = ((isMatch) ? (1u) : (0u));
        uint newInOrig = (currentInOrig | matchAsU32);
        inOriginalAccum = newInOrig;
      }
      uint finalInOrig = inOriginalAccum;
      bool inOriginal = (finalInOrig > 0u);
      bool isFalseNeighbor = (!inOriginal);
      if (isFalseNeighbor) {
        uint rankIdx = (nRowBase + embedNeighbor);
        uint rank = origRanks[rankIdx];
        bool rankGtK = (rank > kVal);
        if (rankGtK) {
          uint diff = (rank - kVal);
          float diffFloat = ((float)(diff));
          float currentPenalty = penaltySumAccum;
          float newPenalty = (currentPenalty + diffFloat);
          penaltySumAccum = newPenalty;
        }
      }
    }
    float finalPenaltySum = penaltySumAccum;
    penalty[gid] = finalPenaltySum;
  }
}

)kernel";
const char* k_metal_navatala_ml_continuity_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_continuity_f32(device const uint* origNeighbors [[buffer(0)]], device const uint* embedNeighbors [[buffer(1)]], device const uint* embedRanks [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device float* penalty [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0u];
  uint kVal = k[0u];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint kRowBase = (gid * kVal);
    uint nRowBase = (gid * nVal);
    float penaltySumAccum = as_type<float>(0x00000000u);
    uint inEmbedAccum = 0u;
    for (int origIdx = 0; origIdx < (int)(kVal); ++origIdx) {
      uint origIdxU32 = ((uint)(origIdx));
      uint origNeighborIdx = (kRowBase + origIdxU32);
      uint origNeighbor = origNeighbors[origNeighborIdx];
      inEmbedAccum = 0u;
      for (int embedIdx = 0; embedIdx < (int)(kVal); ++embedIdx) {
        uint embedIdxU32 = ((uint)(embedIdx));
        uint embedNeighborIdx = (kRowBase + embedIdxU32);
        uint embedNeighbor = embedNeighbors[embedNeighborIdx];
        bool isMatch = (origNeighbor == embedNeighbor);
        uint currentInEmbed = inEmbedAccum;
        uint matchAsU32 = ((isMatch) ? (1u) : (0u));
        uint newInEmbed = (currentInEmbed | matchAsU32);
        inEmbedAccum = newInEmbed;
      }
      uint finalInEmbed = inEmbedAccum;
      bool inEmbed = (finalInEmbed > 0u);
      bool isMissingNeighbor = (!inEmbed);
      if (isMissingNeighbor) {
        uint rankIdx = (nRowBase + origNeighbor);
        uint rank = embedRanks[rankIdx];
        bool rankGtK = (rank > kVal);
        if (rankGtK) {
          uint diff = (rank - kVal);
          float diffFloat = ((float)(diff));
          float currentPenalty = penaltySumAccum;
          float newPenalty = (currentPenalty + diffFloat);
          penaltySumAccum = newPenalty;
        }
      }
    }
    float finalPenaltySum = penaltySumAccum;
    penalty[gid] = finalPenaltySum;
  }
}

)kernel";
const char* k_metal_navatala_ml_make_blobs_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_make_blobs_f32(device const float* centers [[buffer(0)]], device const uint* labels [[buffer(1)]], device const float* randomNormal [[buffer(2)]], device const float* stddev [[buffer(3)]], device const uint* n [[buffer(4)]], device const uint* d [[buffer(5)]], device float* data [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0];
  uint numD = d[0];
  uint totalElems = (numN * numD);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    float sigma = stddev[0];
    uint sampleIdx = (gid / numD);
    uint dimIdx = (gid % numD);
    uint label = labels[sampleIdx];
    uint centerIdx = ((label * numD) + dimIdx);
    float centerVal = centers[centerIdx];
    float randVal = randomNormal[gid];
    float noise = (sigma * randVal);
    float result = (centerVal + noise);
    data[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_ml_make_blobs_anisotropic_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_make_blobs_anisotropic_f32(device const float* centers [[buffer(0)]], device const float* scales [[buffer(1)]], device const uint* labels [[buffer(2)]], device const float* randomNormal [[buffer(3)]], device const uint* n [[buffer(4)]], device const uint* d [[buffer(5)]], device float* data [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0];
  uint numD = d[0];
  uint totalElems = (numN * numD);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint sampleIdx = (gid / numD);
    uint dimIdx = (gid % numD);
    uint label = labels[sampleIdx];
    uint paramIdx = ((label * numD) + dimIdx);
    float centerVal = centers[paramIdx];
    float scaleVal = scales[paramIdx];
    float randVal = randomNormal[gid];
    float noise = (scaleVal * randVal);
    float result = (centerVal + noise);
    data[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_ml_make_moons_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_make_moons_f32(device const uint* labels [[buffer(0)]], device const float* randomUniform [[buffer(1)]], device const float* randomNormal [[buffer(2)]], device const float* noise [[buffer(3)]], device const uint* n [[buffer(4)]], device float* data [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0];
  bool inBounds = (gid < numN);
  if (inBounds) {
    float sigma = noise[0];
    uint label = labels[gid];
    float u = randomUniform[gid];
    float theta = (as_type<float>(0x40490fdbu) * u);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint outIdxX = (gid * 2u);
    uint outIdxY = (outIdxX + 1u);
    float noiseX = randomNormal[outIdxX];
    float noiseY = randomNormal[outIdxY];
    bool isLowerMoon = (label == 1u);
    float baseX_upper = cosTheta;
    float baseY_upper = sinTheta;
    float baseX_lower = (as_type<float>(0x3f800000u) - cosTheta);
    float baseY_lower = (as_type<float>(0x3f000000u) - sinTheta);
    float baseX = ((isLowerMoon) ? (baseX_lower) : (baseX_upper));
    float baseY = ((isLowerMoon) ? (baseY_lower) : (baseY_upper));
    float resultX = (baseX + (sigma * noiseX));
    float resultY = (baseY + (sigma * noiseY));
    data[outIdxX] = resultX;
    data[outIdxY] = resultY;
  }
}

)kernel";
const char* k_metal_navatala_ml_make_circles_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_make_circles_f32(device const uint* labels [[buffer(0)]], device const float* randomUniform [[buffer(1)]], device const float* randomNormal [[buffer(2)]], device const float* noise [[buffer(3)]], device const float* factor [[buffer(4)]], device const uint* n [[buffer(5)]], device float* data [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0];
  bool inBounds = (gid < numN);
  if (inBounds) {
    float sigma = noise[0];
    float innerFactor = factor[0];
    uint label = labels[gid];
    float u = randomUniform[gid];
    float theta = (as_type<float>(0x40c90fdbu) * u);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint outIdxX = (gid * 2u);
    uint outIdxY = (outIdxX + 1u);
    float noiseX = randomNormal[outIdxX];
    float noiseY = randomNormal[outIdxY];
    bool isOuterCircle = (label == 1u);
    float radius = ((isOuterCircle) ? (as_type<float>(0x3f800000u)) : (innerFactor));
    float baseX = (radius * cosTheta);
    float baseY = (radius * sinTheta);
    float resultX = (baseX + (sigma * noiseX));
    float resultY = (baseY + (sigma * noiseY));
    data[outIdxX] = resultX;
    data[outIdxY] = resultY;
  }
}

)kernel";
const char* k_metal_navatala_ml_make_regression_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_make_regression_f32(device const float* X [[buffer(0)]], device const float* w [[buffer(1)]], device const float* bias [[buffer(2)]], device const float* noise [[buffer(3)]], device const uint* n [[buffer(4)]], device const uint* d [[buffer(5)]], device float* y [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0];
  uint numD = d[0];
  bool inBounds = (gid < numN);
  if (inBounds) {
    float biasVal = bias[0];
    uint sampleIdx = gid;
    uint baseIdx = (sampleIdx * numD);
    float x0 = X[baseIdx];
    float w0 = w[0];
    float dot = (x0 * w0);
    float noiseVal = noise[gid];
    float withBias = (dot + biasVal);
    float result = (withBias + noiseVal);
    y[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_ml_make_classification_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_make_classification_f32(device const float* X [[buffer(0)]], device const float* w [[buffer(1)]], device const float* bias [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* d [[buffer(4)]], device uint* labels [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0];
  uint numD = d[0];
  bool inBounds = (gid < numN);
  if (inBounds) {
    float biasVal = bias[0];
    uint sampleIdx = gid;
    uint baseIdx = (sampleIdx * numD);
    float x0 = X[baseIdx];
    float w0 = w[0];
    float dot = (x0 * w0);
    float decision = (dot + biasVal);
    bool isPositive = (decision >= as_type<float>(0x00000000u));
    uint label = ((isPositive) ? (1u) : (0u));
    labels[gid] = label;
  }
}

)kernel";
const char* k_metal_navatala_ml_make_sparse_uncorrelated_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_make_sparse_uncorrelated_f32(device const float* randomNormal [[buffer(0)]], device const uint* sparseMask [[buffer(1)]], device const float* scale [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* d [[buffer(4)]], device float* X [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0];
  uint numD = d[0];
  uint totalElems = (numN * numD);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint featureIdx = (gid % numD);
    uint maskVal = sparseMask[gid];
    float randVal = randomNormal[gid];
    float scaleVal = scale[featureIdx];
    bool isNonZero = (maskVal == 1u);
    float scaledRand = (randVal * scaleVal);
    float result = ((isNonZero) ? (scaledRand) : (as_type<float>(0x00000000u)));
    X[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_ml_knn_brute_force_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_knn_brute_force_f32(device const float* queries [[buffer(0)]], device const float* data [[buffer(1)]], device const uint* q [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* d [[buffer(4)]], device const uint* k [[buffer(5)]], device uint* indices [[buffer(6)]], device float* distances [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint nVal = n[0];
  uint dVal = d[0];
  uint kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    uint queryBase = (gid * dVal);
    uint outBase = (gid * kVal);
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      uint initIdxU32 = ((uint)(initIdx));
      uint outIdx = (outBase + initIdxU32);
      distances[outIdx] = as_type<float>(0x7f7fc99eu);
      indices[outIdx] = 4294967295u;
    }
    for (int dataIdx = 0; dataIdx < (int)(nVal); ++dataIdx) {
      uint dataIdxU32 = ((uint)(dataIdx));
      uint dataBase = (dataIdxU32 * dVal);
      float distSumAccum = as_type<float>(0x00000000u);
      for (int dimIdx = 0; dimIdx < (int)(dVal); ++dimIdx) {
        uint dimIdxU32 = ((uint)(dimIdx));
        uint qIdx = (queryBase + dimIdxU32);
        uint dIdx = (dataBase + dimIdxU32);
        float qv = queries[qIdx];
        float dv = data[dIdx];
        float diff = (qv - dv);
        float diffSq = (diff * diff);
        float currDistSum = distSumAccum;
        float newDistSum = (currDistSum + diffSq);
        distSumAccum = newDistSum;
      }
      float distSum = distSumAccum;
      uint worstIdx = (outBase + (kVal - 1u));
      float worstDist = distances[worstIdx];
      bool shouldInsert = (distSum < worstDist);
      if (shouldInsert) {
        uint initInsertPos = (kVal - 1u);
        uint insertPosAccum = initInsertPos;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          uint scanIdxU32 = ((uint)(scanIdx));
          uint checkPos = ((kVal - 1u) - scanIdxU32);
          uint checkIdx = (outBase + checkPos);
          float checkDist = distances[checkIdx];
          bool shouldShift = (distSum < checkDist);
          uint currInsertPos = insertPosAccum;
          uint newInsertPos = ((shouldShift) ? (checkPos) : (currInsertPos));
          insertPosAccum = newInsertPos;
        }
        uint insertPos = insertPosAccum;
        for (int shiftIdx = 0; shiftIdx < (int)((kVal - 1u)); ++shiftIdx) {
          uint shiftIdxU32 = ((uint)(shiftIdx));
          uint shiftPos = ((kVal - 2u) - shiftIdxU32);
          bool shouldDoShift = (shiftPos >= insertPos);
          if (shouldDoShift) {
            uint srcIdx = (outBase + shiftPos);
            uint dstIdx = (outBase + (shiftPos + 1u));
            float shiftDist = distances[srcIdx];
            uint shiftIndex = indices[srcIdx];
            distances[dstIdx] = shiftDist;
            indices[dstIdx] = shiftIndex;
          }
        }
        uint insertIdx = (outBase + insertPos);
        distances[insertIdx] = distSum;
        indices[insertIdx] = dataIdxU32;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_knn_brute_force_euclidean_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_knn_brute_force_euclidean_f32(device const float* queries [[buffer(0)]], device const float* data [[buffer(1)]], device const uint* q [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* d [[buffer(4)]], device const uint* k [[buffer(5)]], device uint* indices [[buffer(6)]], device float* distances [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint nVal = n[0];
  uint dVal = d[0];
  uint kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    uint queryBase = (gid * dVal);
    uint outBase = (gid * kVal);
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      uint initIdxU32 = ((uint)(initIdx));
      uint outIdx = (outBase + initIdxU32);
      distances[outIdx] = as_type<float>(0x7f7fc99eu);
      indices[outIdx] = 4294967295u;
    }
    for (int dataIdx = 0; dataIdx < (int)(nVal); ++dataIdx) {
      uint dataIdxU32 = ((uint)(dataIdx));
      uint dataBase = (dataIdxU32 * dVal);
      float distSumAccum = as_type<float>(0x00000000u);
      for (int dimIdx = 0; dimIdx < (int)(dVal); ++dimIdx) {
        uint dimIdxU32 = ((uint)(dimIdx));
        uint qIdx = (queryBase + dimIdxU32);
        uint dIdx = (dataBase + dimIdxU32);
        float qv = queries[qIdx];
        float dv = data[dIdx];
        float diff = (qv - dv);
        float diffSq = (diff * diff);
        float currDistSum = distSumAccum;
        float newDistSum = (currDistSum + diffSq);
        distSumAccum = newDistSum;
      }
      float distSum = distSumAccum;
      uint worstIdx = (outBase + (kVal - 1u));
      float worstDist = distances[worstIdx];
      bool shouldInsert = (distSum < worstDist);
      if (shouldInsert) {
        uint initInsertPos = (kVal - 1u);
        uint insertPosAccum = initInsertPos;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          uint scanIdxU32 = ((uint)(scanIdx));
          uint checkPos = ((kVal - 1u) - scanIdxU32);
          uint checkIdx = (outBase + checkPos);
          float checkDist = distances[checkIdx];
          bool shouldShift = (distSum < checkDist);
          uint currInsertPos = insertPosAccum;
          uint newInsertPos = ((shouldShift) ? (checkPos) : (currInsertPos));
          insertPosAccum = newInsertPos;
        }
        uint insertPos = insertPosAccum;
        for (int shiftIdx = 0; shiftIdx < (int)((kVal - 1u)); ++shiftIdx) {
          uint shiftIdxU32 = ((uint)(shiftIdx));
          uint shiftPos = ((kVal - 2u) - shiftIdxU32);
          bool shouldDoShift = (shiftPos >= insertPos);
          if (shouldDoShift) {
            uint srcIdx = (outBase + shiftPos);
            uint dstIdx = (outBase + (shiftPos + 1u));
            float shiftDist = distances[srcIdx];
            uint shiftIndex = indices[srcIdx];
            distances[dstIdx] = shiftDist;
            indices[dstIdx] = shiftIndex;
          }
        }
        uint insertIdx = (outBase + insertPos);
        distances[insertIdx] = distSum;
        indices[insertIdx] = dataIdxU32;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_knn_brute_force_cosine_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_knn_brute_force_cosine_f32(device const float* queries [[buffer(0)]], device const float* data [[buffer(1)]], device const uint* q [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* d [[buffer(4)]], device const uint* k [[buffer(5)]], device uint* indices [[buffer(6)]], device float* distances [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint nVal = n[0];
  uint dVal = d[0];
  uint kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    uint queryBase = (gid * dVal);
    uint outBase = (gid * kVal);
    float queryNormSqAccum = as_type<float>(0x00000000u);
    for (int dimIdx = 0; dimIdx < (int)(dVal); ++dimIdx) {
      uint dimIdxU32 = ((uint)(dimIdx));
      uint qIdx = (queryBase + dimIdxU32);
      float qv = queries[qIdx];
      float qvSq = (qv * qv);
      float currQueryNormSq = queryNormSqAccum;
      float newQueryNormSq = (currQueryNormSq + qvSq);
      queryNormSqAccum = newQueryNormSq;
    }
    float queryNormSq = queryNormSqAccum;
    float queryNorm = sqrt(queryNormSq);
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      uint initIdxU32 = ((uint)(initIdx));
      uint outIdx = (outBase + initIdxU32);
      distances[outIdx] = as_type<float>(0x7f7fc99eu);
      indices[outIdx] = 4294967295u;
    }
    for (int dataIdx = 0; dataIdx < (int)(nVal); ++dataIdx) {
      uint dataIdxU32 = ((uint)(dataIdx));
      uint dataBase = (dataIdxU32 * dVal);
      float dotProdAccum = as_type<float>(0x00000000u);
      float dataNormSqAccum = as_type<float>(0x00000000u);
      for (int dimIdx = 0; dimIdx < (int)(dVal); ++dimIdx) {
        uint dimIdxU32 = ((uint)(dimIdx));
        uint qIdx = (queryBase + dimIdxU32);
        uint dIdx = (dataBase + dimIdxU32);
        float qv = queries[qIdx];
        float dv = data[dIdx];
        float prod = (qv * dv);
        float currDotProd = dotProdAccum;
        float newDotProd = (currDotProd + prod);
        dotProdAccum = newDotProd;
        float dvSq = (dv * dv);
        float currDataNormSq = dataNormSqAccum;
        float newDataNormSq = (currDataNormSq + dvSq);
        dataNormSqAccum = newDataNormSq;
      }
      float dotProd = dotProdAccum;
      float dataNormSq = dataNormSqAccum;
      float dataNorm = sqrt(dataNormSq);
      float normProduct = (queryNorm * dataNorm);
      bool normIsZero = (normProduct == as_type<float>(0x00000000u));
      float cosineSim = (dotProd / normProduct);
      float cosineDist = (as_type<float>(0x3f800000u) - cosineSim);
      float dist = ((normIsZero) ? (as_type<float>(0x3f800000u)) : (cosineDist));
      uint worstIdx = (outBase + (kVal - 1u));
      float worstDist = distances[worstIdx];
      bool shouldInsert = (dist < worstDist);
      if (shouldInsert) {
        uint initInsertPos = (kVal - 1u);
        uint insertPosAccum = initInsertPos;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          uint scanIdxU32 = ((uint)(scanIdx));
          uint checkPos = ((kVal - 1u) - scanIdxU32);
          uint checkIdx = (outBase + checkPos);
          float checkDist = distances[checkIdx];
          bool shouldShift = (dist < checkDist);
          uint currInsertPos = insertPosAccum;
          uint newInsertPos = ((shouldShift) ? (checkPos) : (currInsertPos));
          insertPosAccum = newInsertPos;
        }
        uint insertPos = insertPosAccum;
        for (int shiftIdx = 0; shiftIdx < (int)((kVal - 1u)); ++shiftIdx) {
          uint shiftIdxU32 = ((uint)(shiftIdx));
          uint shiftPos = ((kVal - 2u) - shiftIdxU32);
          bool shouldDoShift = (shiftPos >= insertPos);
          if (shouldDoShift) {
            uint srcIdx = (outBase + shiftPos);
            uint dstIdx = (outBase + (shiftPos + 1u));
            float shiftDist = distances[srcIdx];
            uint shiftIndex = indices[srcIdx];
            distances[dstIdx] = shiftDist;
            indices[dstIdx] = shiftIndex;
          }
        }
        uint insertIdx = (outBase + insertPos);
        distances[insertIdx] = dist;
        indices[insertIdx] = dataIdxU32;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_knn_brute_force_manhattan_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_knn_brute_force_manhattan_f32(device const float* queries [[buffer(0)]], device const float* data [[buffer(1)]], device const uint* q [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* d [[buffer(4)]], device const uint* k [[buffer(5)]], device uint* indices [[buffer(6)]], device float* distances [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint nVal = n[0];
  uint dVal = d[0];
  uint kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    uint queryBase = (gid * dVal);
    uint outBase = (gid * kVal);
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      uint initIdxU32 = ((uint)(initIdx));
      uint outIdx = (outBase + initIdxU32);
      distances[outIdx] = as_type<float>(0x7f7fc99eu);
      indices[outIdx] = 4294967295u;
    }
    for (int dataIdx = 0; dataIdx < (int)(nVal); ++dataIdx) {
      uint dataIdxU32 = ((uint)(dataIdx));
      uint dataBase = (dataIdxU32 * dVal);
      float distSumAccum = as_type<float>(0x00000000u);
      for (int dimIdx = 0; dimIdx < (int)(dVal); ++dimIdx) {
        uint dimIdxU32 = ((uint)(dimIdx));
        uint qIdx = (queryBase + dimIdxU32);
        uint dIdx = (dataBase + dimIdxU32);
        float qv = queries[qIdx];
        float dv = data[dIdx];
        float diff = (qv - dv);
        float absDiff = abs(diff);
        float currDistSum = distSumAccum;
        float newDistSum = (currDistSum + absDiff);
        distSumAccum = newDistSum;
      }
      float distSum = distSumAccum;
      uint worstIdx = (outBase + (kVal - 1u));
      float worstDist = distances[worstIdx];
      bool shouldInsert = (distSum < worstDist);
      if (shouldInsert) {
        uint initInsertPos = (kVal - 1u);
        uint insertPosAccum = initInsertPos;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          uint scanIdxU32 = ((uint)(scanIdx));
          uint checkPos = ((kVal - 1u) - scanIdxU32);
          uint checkIdx = (outBase + checkPos);
          float checkDist = distances[checkIdx];
          bool shouldShift = (distSum < checkDist);
          uint currInsertPos = insertPosAccum;
          uint newInsertPos = ((shouldShift) ? (checkPos) : (currInsertPos));
          insertPosAccum = newInsertPos;
        }
        uint insertPos = insertPosAccum;
        for (int shiftIdx = 0; shiftIdx < (int)((kVal - 1u)); ++shiftIdx) {
          uint shiftIdxU32 = ((uint)(shiftIdx));
          uint shiftPos = ((kVal - 2u) - shiftIdxU32);
          bool shouldDoShift = (shiftPos >= insertPos);
          if (shouldDoShift) {
            uint srcIdx = (outBase + shiftPos);
            uint dstIdx = (outBase + (shiftPos + 1u));
            float shiftDist = distances[srcIdx];
            uint shiftIndex = indices[srcIdx];
            distances[dstIdx] = shiftDist;
            indices[dstIdx] = shiftIndex;
          }
        }
        uint insertIdx = (outBase + insertPos);
        distances[insertIdx] = distSum;
        indices[insertIdx] = dataIdxU32;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_knn_with_radius_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_knn_with_radius_f32(device const float* queries [[buffer(0)]], device const float* data [[buffer(1)]], device const uint* q [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* d [[buffer(4)]], device const uint* k [[buffer(5)]], device const float* radius [[buffer(6)]], device uint* indices [[buffer(7)]], device float* distances [[buffer(8)]], device uint* counts [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint nVal = n[0];
  uint dVal = d[0];
  uint kVal = k[0];
  float radiusVal = radius[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    uint queryBase = (gid * dVal);
    uint outBase = (gid * kVal);
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      uint initIdxU32 = ((uint)(initIdx));
      uint outIdx = (outBase + initIdxU32);
      distances[outIdx] = as_type<float>(0x7f7fc99eu);
      indices[outIdx] = 4294967295u;
    }
    uint neighborCountAccum = 0u;
    for (int dataIdx = 0; dataIdx < (int)(nVal); ++dataIdx) {
      uint dataIdxU32 = ((uint)(dataIdx));
      uint dataBase = (dataIdxU32 * dVal);
      float distSumAccum = as_type<float>(0x00000000u);
      for (int dimIdx = 0; dimIdx < (int)(dVal); ++dimIdx) {
        uint dimIdxU32 = ((uint)(dimIdx));
        uint qIdx = (queryBase + dimIdxU32);
        uint dIdx = (dataBase + dimIdxU32);
        float qv = queries[qIdx];
        float dv = data[dIdx];
        float diff = (qv - dv);
        float diffSq = (diff * diff);
        float currDistSum = distSumAccum;
        float newDistSum = (currDistSum + diffSq);
        distSumAccum = newDistSum;
      }
      float distSum = distSumAccum;
      bool withinRadius = (distSum <= radiusVal);
      if (withinRadius) {
        uint worstIdx = (outBase + (kVal - 1u));
        float worstDist = distances[worstIdx];
        bool shouldInsert = (distSum < worstDist);
        if (shouldInsert) {
          uint currNeighborCount = neighborCountAccum;
          uint countPlusOne = (currNeighborCount + 1u);
          bool countCapped = (currNeighborCount < kVal);
          uint newNeighborCount = ((countCapped) ? (countPlusOne) : (kVal));
          neighborCountAccum = newNeighborCount;
          uint initInsertPos = (kVal - 1u);
          uint insertPosAccum = initInsertPos;
          for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
            uint scanIdxU32 = ((uint)(scanIdx));
            uint checkPos = ((kVal - 1u) - scanIdxU32);
            uint checkIdx = (outBase + checkPos);
            float checkDist = distances[checkIdx];
            bool shouldShift = (distSum < checkDist);
            uint currInsertPos = insertPosAccum;
            uint newInsertPos = ((shouldShift) ? (checkPos) : (currInsertPos));
            insertPosAccum = newInsertPos;
          }
          uint insertPos = insertPosAccum;
          for (int shiftIdx = 0; shiftIdx < (int)((kVal - 1u)); ++shiftIdx) {
            uint shiftIdxU32 = ((uint)(shiftIdx));
            uint shiftPos = ((kVal - 2u) - shiftIdxU32);
            bool shouldDoShift = (shiftPos >= insertPos);
            if (shouldDoShift) {
              uint srcIdx = (outBase + shiftPos);
              uint dstIdx = (outBase + (shiftPos + 1u));
              float shiftDist = distances[srcIdx];
              uint shiftIndex = indices[srcIdx];
              distances[dstIdx] = shiftDist;
              indices[dstIdx] = shiftIndex;
            }
          }
          uint insertIdx = (outBase + insertPos);
          distances[insertIdx] = distSum;
          indices[insertIdx] = dataIdxU32;
        }
      }
    }
    uint neighborCount = neighborCountAccum;
    counts[gid] = neighborCount;
  }
}

)kernel";
const char* k_metal_navatala_ml_knn_classify_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_knn_classify_f32(device const uint* neighborIndices [[buffer(0)]], device const uint* labels [[buffer(1)]], device const uint* q [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* numClasses [[buffer(4)]], device uint* predictions [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint kVal = k[0];
  uint nClasses = numClasses[0];
  uint queryIdx = gid;
  bool inBounds = (queryIdx < qVal);
  if (inBounds) {
    uint baseIdx = (queryIdx * kVal);
    uint bestClass = 0u;
    uint bestVotes = 0u;
    for (int classIdx = 0; classIdx < (int)(nClasses); ++classIdx) {
      uint classIdxU32 = ((uint)(classIdx));
      uint classVotes = 0u;
      for (int j = 0; j < (int)(kVal); ++j) {
        uint jU32 = ((uint)(j));
        uint neighborIdx = (baseIdx + jU32);
        uint neighborDataIdx = neighborIndices[neighborIdx];
        uint neighborLabel = labels[neighborDataIdx];
        bool isMatch = (neighborLabel == classIdxU32);
        uint increment = ((isMatch) ? (1u) : (0u));
        uint currentVotes = classVotes;
        uint newClassVotes = (currentVotes + increment);
        classVotes = newClassVotes;
      }
      uint finalClassVotes = classVotes;
      uint currentBestVotes = bestVotes;
      uint currentBestClass = bestClass;
      bool isBetter = (finalClassVotes > currentBestVotes);
      uint newBestClass = ((isBetter) ? (classIdxU32) : (currentBestClass));
      uint newBestVotes = ((isBetter) ? (finalClassVotes) : (currentBestVotes));
      bestClass = newBestClass;
      bestVotes = newBestVotes;
    }
    uint finalBestClass = bestClass;
    predictions[queryIdx] = finalBestClass;
  }
}

)kernel";
const char* k_metal_navatala_ml_knn_classify_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_knn_classify_f64(device const uint* neighborIndices [[buffer(0)]], device const uint* labels [[buffer(1)]], device const uint* q [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* numClasses [[buffer(4)]], device uint* predictions [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint kVal = k[0];
  uint nClasses = numClasses[0];
  uint queryIdx = gid;
  bool inBounds = (queryIdx < qVal);
  if (inBounds) {
    uint baseIdx = (queryIdx * kVal);
    uint bestClass = 0u;
    uint bestVotes = 0u;
    for (int classIdx = 0; classIdx < (int)(nClasses); ++classIdx) {
      uint classIdxU32 = ((uint)(classIdx));
      uint classVotes = 0u;
      for (int j = 0; j < (int)(kVal); ++j) {
        uint jU32 = ((uint)(j));
        uint neighborIdx = (baseIdx + jU32);
        uint neighborDataIdx = neighborIndices[neighborIdx];
        uint neighborLabel = labels[neighborDataIdx];
        bool isMatch = (neighborLabel == classIdxU32);
        uint increment = ((isMatch) ? (1u) : (0u));
        uint currentVotes = classVotes;
        uint newClassVotes = (currentVotes + increment);
        classVotes = newClassVotes;
      }
      uint finalClassVotes = classVotes;
      uint currentBestVotes = bestVotes;
      uint currentBestClass = bestClass;
      bool isBetter = (finalClassVotes > currentBestVotes);
      uint newBestClass = ((isBetter) ? (classIdxU32) : (currentBestClass));
      uint newBestVotes = ((isBetter) ? (finalClassVotes) : (currentBestVotes));
      bestClass = newBestClass;
      bestVotes = newBestVotes;
    }
    uint finalBestClass = bestClass;
    predictions[queryIdx] = finalBestClass;
  }
}

)kernel";
const char* k_metal_navatala_ml_knn_classify_weighted_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_knn_classify_weighted_f32(device const uint* neighborIndices [[buffer(0)]], device const float* neighborDistances [[buffer(1)]], device const uint* labels [[buffer(2)]], device const uint* q [[buffer(3)]], device const uint* k [[buffer(4)]], device const uint* numClasses [[buffer(5)]], device uint* predictions [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint kVal = k[0];
  uint nClasses = numClasses[0];
  uint queryIdx = gid;
  bool inBounds = (queryIdx < qVal);
  if (inBounds) {
    uint baseIdx = (queryIdx * kVal);
    uint bestClass = 0u;
    float bestWeight = as_type<float>(0x00000000u);
    float epsilon = as_type<float>(0x2edbe6ffu);
    for (int classIdx = 0; classIdx < (int)(nClasses); ++classIdx) {
      uint classIdxU32 = ((uint)(classIdx));
      float classWeight = as_type<float>(0x00000000u);
      for (int j = 0; j < (int)(kVal); ++j) {
        uint jU32 = ((uint)(j));
        uint neighborIdx = (baseIdx + jU32);
        uint neighborDataIdx = neighborIndices[neighborIdx];
        uint neighborLabel = labels[neighborDataIdx];
        float dist = neighborDistances[neighborIdx];
        float weight = (as_type<float>(0x3f800000u) / (dist + epsilon));
        bool isMatch = (neighborLabel == classIdxU32);
        float addWeight = ((isMatch) ? (weight) : (as_type<float>(0x00000000u)));
        float currentWeight = classWeight;
        float newClassWeight = (currentWeight + addWeight);
        classWeight = newClassWeight;
      }
      float finalClassWeight = classWeight;
      float currentBestWeight = bestWeight;
      uint currentBestClass = bestClass;
      bool isBetter = (finalClassWeight > currentBestWeight);
      uint newBestClass = ((isBetter) ? (classIdxU32) : (currentBestClass));
      float newBestWeight = ((isBetter) ? (finalClassWeight) : (currentBestWeight));
      bestClass = newBestClass;
      bestWeight = newBestWeight;
    }
    uint finalBestClass = bestClass;
    predictions[queryIdx] = finalBestClass;
  }
}

)kernel";
const char* k_metal_navatala_ml_knn_classify_probs_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_knn_classify_probs_f32(device const uint* neighborIndices [[buffer(0)]], device const uint* labels [[buffer(1)]], device const uint* q [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* numClasses [[buffer(4)]], device float* probs [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint kVal = k[0];
  uint nClasses = numClasses[0];
  uint queryIdx = gid;
  bool inBounds = (queryIdx < qVal);
  if (inBounds) {
    uint neighborBase = (queryIdx * kVal);
    uint probBase = (queryIdx * nClasses);
    float kFloat = ((float)(kVal));
    float normFactor = (as_type<float>(0x3f800000u) / kFloat);
    for (int classIdx = 0; classIdx < (int)(nClasses); ++classIdx) {
      uint classIdxU32 = ((uint)(classIdx));
      uint classVotes = 0u;
      for (int j = 0; j < (int)(kVal); ++j) {
        uint jU32 = ((uint)(j));
        uint neighborIdx = (neighborBase + jU32);
        uint neighborDataIdx = neighborIndices[neighborIdx];
        uint neighborLabel = labels[neighborDataIdx];
        bool isMatch = (neighborLabel == classIdxU32);
        uint increment = ((isMatch) ? (1u) : (0u));
        uint currentVotes = classVotes;
        uint newClassVotes = (currentVotes + increment);
        classVotes = newClassVotes;
      }
      uint finalVotes = classVotes;
      float votesFloat = ((float)(finalVotes));
      float prob = (votesFloat * normFactor);
      uint probIdx = (probBase + classIdxU32);
      probs[probIdx] = prob;
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_vote_majority_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_vote_majority_f32(device const uint* neighborLabels [[buffer(0)]], device const uint* q [[buffer(1)]], device const uint* k [[buffer(2)]], device const uint* numClasses [[buffer(3)]], device uint* predictions [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint kVal = k[0];
  uint nClasses = numClasses[0];
  uint queryIdx = gid;
  bool inBounds = (queryIdx < qVal);
  if (inBounds) {
    uint baseIdx = (queryIdx * kVal);
    uint bestClass = 0u;
    uint bestVotes = 0u;
    for (int classIdx = 0; classIdx < (int)(nClasses); ++classIdx) {
      uint classIdxU32 = ((uint)(classIdx));
      uint classVotes = 0u;
      for (int j = 0; j < (int)(kVal); ++j) {
        uint jU32 = ((uint)(j));
        uint labelIdx = (baseIdx + jU32);
        uint neighborLabel = neighborLabels[labelIdx];
        bool isMatch = (neighborLabel == classIdxU32);
        uint increment = ((isMatch) ? (1u) : (0u));
        uint currentVotes = classVotes;
        uint newClassVotes = (currentVotes + increment);
        classVotes = newClassVotes;
      }
      uint finalClassVotes = classVotes;
      uint currentBestVotes = bestVotes;
      uint currentBestClass = bestClass;
      bool isBetter = (finalClassVotes > currentBestVotes);
      uint newBestClass = ((isBetter) ? (classIdxU32) : (currentBestClass));
      uint newBestVotes = ((isBetter) ? (finalClassVotes) : (currentBestVotes));
      bestClass = newBestClass;
      bestVotes = newBestVotes;
    }
    uint finalBestClass = bestClass;
    predictions[queryIdx] = finalBestClass;
  }
}

)kernel";
const char* k_metal_navatala_ml_vote_majority_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_vote_majority_f64(device const uint* neighborLabels [[buffer(0)]], device const uint* q [[buffer(1)]], device const uint* k [[buffer(2)]], device const uint* numClasses [[buffer(3)]], device uint* predictions [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint kVal = k[0];
  uint nClasses = numClasses[0];
  uint queryIdx = gid;
  bool inBounds = (queryIdx < qVal);
  if (inBounds) {
    uint baseIdx = (queryIdx * kVal);
    uint bestClass = 0u;
    uint bestVotes = 0u;
    for (int classIdx = 0; classIdx < (int)(nClasses); ++classIdx) {
      uint classIdxU32 = ((uint)(classIdx));
      uint classVotes = 0u;
      for (int j = 0; j < (int)(kVal); ++j) {
        uint jU32 = ((uint)(j));
        uint labelIdx = (baseIdx + jU32);
        uint neighborLabel = neighborLabels[labelIdx];
        bool isMatch = (neighborLabel == classIdxU32);
        uint increment = ((isMatch) ? (1u) : (0u));
        uint currentVotes = classVotes;
        uint newClassVotes = (currentVotes + increment);
        classVotes = newClassVotes;
      }
      uint finalClassVotes = classVotes;
      uint currentBestVotes = bestVotes;
      uint currentBestClass = bestClass;
      bool isBetter = (finalClassVotes > currentBestVotes);
      uint newBestClass = ((isBetter) ? (classIdxU32) : (currentBestClass));
      uint newBestVotes = ((isBetter) ? (finalClassVotes) : (currentBestVotes));
      bestClass = newBestClass;
      bestVotes = newBestVotes;
    }
    uint finalBestClass = bestClass;
    predictions[queryIdx] = finalBestClass;
  }
}

)kernel";
const char* k_metal_navatala_ml_knn_regress_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_knn_regress_f32(device const uint* neighborIndices [[buffer(0)]], device const float* targetValues [[buffer(1)]], device const uint* q [[buffer(2)]], device const uint* k [[buffer(3)]], device float* predictions [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    uint baseIdx = (gid * kVal);
    float sumAccum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(kVal); ++j) {
      uint jU32 = ((uint)(j));
      uint neighborOffset = (baseIdx + jU32);
      uint neighborIdx = neighborIndices[neighborOffset];
      float targetVal = targetValues[neighborIdx];
      float currentSum = sumAccum;
      float newSum = (currentSum + targetVal);
      sumAccum = newSum;
    }
    float finalSum = sumAccum;
    float kFloat = ((float)(kVal));
    float meanVal = (finalSum / kFloat);
    predictions[gid] = meanVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_knn_regress_weighted_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_knn_regress_weighted_f32(device const uint* neighborIndices [[buffer(0)]], device const float* neighborDistances [[buffer(1)]], device const float* targetValues [[buffer(2)]], device const uint* q [[buffer(3)]], device const uint* k [[buffer(4)]], device float* predictions [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    uint baseIdx = (gid * kVal);
    float weightedSumAccum = as_type<float>(0x00000000u);
    float weightSumAccum = as_type<float>(0x00000000u);
    float epsilon = as_type<float>(0x2edbe6ffu);
    for (int j = 0; j < (int)(kVal); ++j) {
      uint jU32 = ((uint)(j));
      uint neighborOffset = (baseIdx + jU32);
      uint neighborIdx = neighborIndices[neighborOffset];
      float dist = neighborDistances[neighborOffset];
      float targetVal = targetValues[neighborIdx];
      float safeDist = (((dist < epsilon)) ? (epsilon) : (dist));
      float weight = (as_type<float>(0x3f800000u) / safeDist);
      float currentWeightedSum = weightedSumAccum;
      float newWeightedSum = (currentWeightedSum + (targetVal * weight));
      weightedSumAccum = newWeightedSum;
      float currentWeightSum = weightSumAccum;
      float newWeightSum = (currentWeightSum + weight);
      weightSumAccum = newWeightSum;
    }
    float finalWeightedSum = weightedSumAccum;
    float finalWeightSum = weightSumAccum;
    float safeDenom = (((finalWeightSum == as_type<float>(0x00000000u))) ? (as_type<float>(0x3f800000u)) : (finalWeightSum));
    float prediction = (finalWeightedSum / safeDenom);
    predictions[gid] = prediction;
  }
}

)kernel";
const char* k_metal_navatala_ml_knn_regress_median_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_knn_regress_median_f32(device const uint* neighborIndices [[buffer(0)]], device const float* targetValues [[buffer(1)]], device const uint* q [[buffer(2)]], device const uint* k [[buffer(3)]], device float* predictions [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    threadgroup float sortBuf[256];
    uint baseIdx = (gid * kVal);
    for (int j = 0; j < (int)(kVal); ++j) {
      uint jU32 = ((uint)(j));
      uint neighborOffset = (baseIdx + jU32);
      uint neighborIdx = neighborIndices[neighborOffset];
      float targetVal = targetValues[neighborIdx];
      sortBuf[jU32] = targetVal;
    }
    uint jIdxAccum = 0u;
    for (int i = 0; i < (int)(kVal); ++i) {
      uint iU32 = ((uint)(i));
      uint zeroU32 = ((uint)(0));
      if (iU32 > zeroU32) {
        float key = sortBuf[iU32];
        uint initialJIdx = (iU32 - 1u);
        jIdxAccum = initialJIdx;
        for (int iter = 0; iter < (int)(iU32); ++iter) {
          uint currJIdx = jIdxAccum;
          float currentVal = sortBuf[currJIdx];
          bool shouldShift = (currentVal > key);
          if (shouldShift) {
            uint jPlus1 = (currJIdx + 1u);
            sortBuf[jPlus1] = currentVal;
            uint newJIdx = (((currJIdx == 0u)) ? (0u) : ((currJIdx - 1u)));
            jIdxAccum = newJIdx;
          }
        }
        uint finalJIdx = jIdxAccum;
        float lastCurrentVal = sortBuf[finalJIdx];
        uint insertPos = (((lastCurrentVal > key)) ? (finalJIdx) : ((finalJIdx + 1u)));
        sortBuf[insertPos] = key;
      }
    }
    uint midIdx = (kVal / 2u);
    uint kMod2 = (kVal % 2u);
    bool isOdd = (kMod2 != 0u);
    float midVal = sortBuf[midIdx];
    uint midMinus1Idx = (((midIdx == 0u)) ? (0u) : ((midIdx - 1u)));
    float midMinus1Val = sortBuf[midMinus1Idx];
    float evenMedian = ((midMinus1Val + midVal) / as_type<float>(0x40000000u));
    float median = ((isOdd) ? (midVal) : (evenMedian));
    predictions[gid] = median;
  }
}

)kernel";
const char* k_metal_navatala_ml_traverse_tree_classify_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_traverse_tree_classify_f32(device const float* X [[buffer(0)]], device const int* feature_indices [[buffer(1)]], device const float* thresholds [[buffer(2)]], device const int* left_children [[buffer(3)]], device const int* right_children [[buffer(4)]], device const int* leaf_value_indices [[buffer(5)]], device const float* leaf_values [[buffer(6)]], device const uint* n_samples [[buffer(7)]], device const uint* n_features [[buffer(8)]], device const uint* n_classes [[buffer(9)]], device int* predictions [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nFeatures = n_features[0];
  uint nClasses = n_classes[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    int currentNode = 0;
    for (int depth = 0; depth < (int)(64u); ++depth) {
      int nodeIdx = currentNode;
      uint nodeIdxU32 = ((uint)(nodeIdx));
      int featureIdx = feature_indices[nodeIdxU32];
      bool isLeaf = (featureIdx == -1);
      if (isLeaf) {
        int leafIdx = leaf_value_indices[nodeIdxU32];
        uint leafIdxU32 = ((uint)(leafIdx));
        uint leafBase = (leafIdxU32 * nClasses);
        int bestClass = 0;
        float bestProb = as_type<float>(0xbf800000u);
        for (int c = 0; c < (int)(nClasses); ++c) {
          uint cU32 = ((uint)(c));
          uint probIdx = (leafBase + cU32);
          float prob = leaf_values[probIdx];
          float currBest = bestProb;
          if (prob > currBest) {
            bestClass = c;
            bestProb = prob;
          }
        }
        int finalClass = bestClass;
        predictions[gid] = finalClass;
      } else {
        float threshold = thresholds[nodeIdxU32];
        uint featureIdxU32 = ((uint)(featureIdx));
        uint featureAddr = (sampleBase + featureIdxU32);
        float featureVal = X[featureAddr];
        bool goLeft = (featureVal <= threshold);
        if (goLeft) {
          int leftChild = left_children[nodeIdxU32];
          currentNode = leftChild;
        } else {
          int rightChild = right_children[nodeIdxU32];
          currentNode = rightChild;
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_traverse_tree_regress_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_traverse_tree_regress_f32(device const float* X [[buffer(0)]], device const int* feature_indices [[buffer(1)]], device const float* thresholds [[buffer(2)]], device const int* left_children [[buffer(3)]], device const int* right_children [[buffer(4)]], device const int* leaf_value_indices [[buffer(5)]], device const float* leaf_values [[buffer(6)]], device const uint* n_samples [[buffer(7)]], device const uint* n_features [[buffer(8)]], device float* predictions [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nFeatures = n_features[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    int currentNode = 0;
    for (int depth = 0; depth < (int)(64u); ++depth) {
      int nodeIdx = currentNode;
      uint nodeIdxU32 = ((uint)(nodeIdx));
      int featureIdx = feature_indices[nodeIdxU32];
      bool isLeaf = (featureIdx == -1);
      if (isLeaf) {
        int leafIdx = leaf_value_indices[nodeIdxU32];
        uint leafIdxU32 = ((uint)(leafIdx));
        float leafVal = leaf_values[leafIdxU32];
        predictions[gid] = leafVal;
      } else {
        float threshold = thresholds[nodeIdxU32];
        uint featureIdxU32 = ((uint)(featureIdx));
        uint featureAddr = (sampleBase + featureIdxU32);
        float featureVal = X[featureAddr];
        bool goLeft = (featureVal <= threshold);
        if (goLeft) {
          int leftChild = left_children[nodeIdxU32];
          currentNode = leftChild;
        } else {
          int rightChild = right_children[nodeIdxU32];
          currentNode = rightChild;
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_gini_impurity_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_gini_impurity_f32(device const int* labels [[buffer(0)]], device const uint* sample_indices [[buffer(1)]], device const uint* n_node_samples [[buffer(2)]], device const uint* n_classes [[buffer(3)]], device float* gini [[buffer(4)]], device uint* class_counts [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint nSamples = n_node_samples[0];
  uint nClasses = n_classes[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleIdx = sample_indices[gid];
    int label = labels[sampleIdx];
    uint labelU32 = ((uint)(label));
    uint currCount = class_counts[labelU32];
    uint newCount = (currCount + 1u);
    class_counts[labelU32] = newCount;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool isFirstThread = (gid == 0u);
  if (isFirstThread) {
    float sumPSquared = as_type<float>(0x00000000u);
    float nSamplesF = ((float)(nSamples));
    for (int c = 0; c < (int)(nClasses); ++c) {
      uint cU32 = ((uint)(c));
      uint count = class_counts[cU32];
      float countF = ((float)(count));
      float p = (countF / nSamplesF);
      float pSquared = (p * p);
      float currSum = sumPSquared;
      float newSum = (currSum + pSquared);
      sumPSquared = newSum;
    }
    float finalSum = sumPSquared;
    float giniVal = (as_type<float>(0x3f800000u) - finalSum);
    gini[0] = giniVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_m_s_e_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_m_s_e_f32(device const float* targets [[buffer(0)]], device const uint* sample_indices [[buffer(1)]], device const uint* n_node_samples [[buffer(2)]], device float* mse [[buffer(3)]], device float* mean_val [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint nSamples = n_node_samples[0];
  bool inBounds = (gid < nSamples);
  threadgroup float sdata[256];
  if (inBounds) {
    uint sampleIdx = sample_indices[gid];
    float targetVal = targets[sampleIdx];
    sdata[lid] = targetVal;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint mseF32_reductionStride = 128u;
  for (int mseF32_reductionStep = 0; mseF32_reductionStep < (int)(8); ++mseF32_reductionStep) {
    uint mseF32_stride = mseF32_reductionStride;
    if (lid < mseF32_stride) {
      uint otherIdx = (lid + mseF32_stride);
      float mine = sdata[lid];
      float other = sdata[otherIdx];
      float combined = (mine + other);
      sdata[lid] = combined;
    }
    uint mseF32_strideToHalve = mseF32_reductionStride;
    uint mseF32_nextStride = (mseF32_strideToHalve >> 1u);
    mseF32_reductionStride = mseF32_nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  bool isFirst = (lid == 0u);
  if (isFirst) {
    float totalSum = sdata[0];
    float nSamplesF = ((float)(nSamples));
    float meanValue = (totalSum / nSamplesF);
    mean_val[0] = meanValue;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float meanForDev = mean_val[0];
  if (inBounds) {
    uint sampleIdx2 = sample_indices[gid];
    float targetVal2 = targets[sampleIdx2];
    float dev = (targetVal2 - meanForDev);
    float devSq = (dev * dev);
    sdata[lid] = devSq;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint mseF32b_reductionStride = 128u;
  for (int mseF32b_reductionStep = 0; mseF32b_reductionStep < (int)(8); ++mseF32b_reductionStep) {
    uint mseF32b_stride = mseF32b_reductionStride;
    if (lid < mseF32b_stride) {
      uint otherIdx2 = (lid + mseF32b_stride);
      float mine2 = sdata[lid];
      float other2 = sdata[otherIdx2];
      float combined2 = (mine2 + other2);
      sdata[lid] = combined2;
    }
    uint mseF32b_strideToHalve = mseF32b_reductionStride;
    uint mseF32b_nextStride = (mseF32b_strideToHalve >> 1u);
    mseF32b_reductionStride = mseF32b_nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (isFirst) {
    float totalDevSq = sdata[0];
    float nSamplesF2 = ((float)(nSamples));
    float mseValue = (totalDevSq / nSamplesF2);
    mse[0] = mseValue;
  }
}

)kernel";
const char* k_metal_navatala_ml_split_samples_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_split_samples_f32(device const float* X [[buffer(0)]], device const uint* sample_indices [[buffer(1)]], device const uint* n_node_samples [[buffer(2)]], device const uint* n_features [[buffer(3)]], device const uint* split_feature [[buffer(4)]], device const float* threshold [[buffer(5)]], device uint* left_indices [[buffer(6)]], device uint* right_indices [[buffer(7)]], device uint* left_count [[buffer(8)]], device uint* right_count [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_node_samples[0];
  uint nFeatures = n_features[0];
  uint splitFeat = split_feature[0];
  float thresh = threshold[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleIdx = sample_indices[gid];
    uint featureAddr = ((sampleIdx * nFeatures) + splitFeat);
    float featureVal = X[featureAddr];
    bool goLeft = (featureVal <= thresh);
    if (goLeft) {
      uint pos = left_count[0];
      uint newPos = (pos + 1u);
      left_count[0] = newPos;
      left_indices[pos] = sampleIdx;
    } else {
      uint pos = right_count[0];
      uint newPos = (pos + 1u);
      right_count[0] = newPos;
      right_indices[pos] = sampleIdx;
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_leaf_class_probs_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_leaf_class_probs_f32(device const int* labels [[buffer(0)]], device const uint* sample_indices [[buffer(1)]], device const uint* n_leaf_samples [[buffer(2)]], device const uint* n_classes [[buffer(3)]], device float* probs [[buffer(4)]], device uint* counts [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_leaf_samples[0];
  uint nClasses = n_classes[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleIdx = sample_indices[gid];
    int label = labels[sampleIdx];
    uint labelU32 = ((uint)(label));
    uint currCount = counts[labelU32];
    uint newCount = (currCount + 1u);
    counts[labelU32] = newCount;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool inBoundsClass = (gid < nClasses);
  if (inBoundsClass) {
    uint count = counts[gid];
    float countF = ((float)(count));
    float nSamplesF = ((float)(nSamples));
    float prob = (countF / nSamplesF);
    probs[gid] = prob;
  }
}

)kernel";
const char* k_metal_navatala_ml_forest_classify_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_forest_classify_f32(device const float* X [[buffer(0)]], device const int* feature_indices [[buffer(1)]], device const float* thresholds [[buffer(2)]], device const int* left_children [[buffer(3)]], device const int* right_children [[buffer(4)]], device const int* leaf_value_indices [[buffer(5)]], device const float* forest_leaves [[buffer(6)]], device const int* tree_offsets [[buffer(7)]], device const int* leaf_offsets [[buffer(8)]], device const uint* n_samples [[buffer(9)]], device const uint* n_features [[buffer(10)]], device const uint* n_trees [[buffer(11)]], device const uint* n_classes [[buffer(12)]], device int* predictions [[buffer(13)]], device uint* vote_counts [[buffer(14)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nFeatures = n_features[0];
  uint nTrees = n_trees[0];
  uint nClasses = n_classes[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    uint voteBase = (gid * nClasses);
    for (int c = 0; c < (int)(nClasses); ++c) {
      uint cU32 = ((uint)(c));
      uint voteIdx = (voteBase + cU32);
      vote_counts[voteIdx] = 0u;
    }
    for (int t = 0; t < (int)(nTrees); ++t) {
      uint tU32 = ((uint)(t));
      int treeNodeOffset = tree_offsets[tU32];
      uint treeNodeOffsetU32 = ((uint)(treeNodeOffset));
      int treeLeafOffset = leaf_offsets[tU32];
      uint treeLeafOffsetU32 = ((uint)(treeLeafOffset));
      int currentNode = 0;
      for (int depth = 0; depth < (int)(64u); ++depth) {
        int nodeIdx = currentNode;
        uint nodeIdxU32 = ((uint)(nodeIdx));
        uint absNodeIdx = (treeNodeOffsetU32 + nodeIdxU32);
        int featureIdx = feature_indices[absNodeIdx];
        bool isLeaf = (featureIdx == -1);
        if (isLeaf) {
          int leafIdxI32 = leaf_value_indices[absNodeIdx];
          uint leafIdx = ((uint)(leafIdxI32));
          uint absLeafBase = (treeLeafOffsetU32 + (leafIdx * nClasses));
          int bestClass = 0;
          float bestProb = as_type<float>(0xbf800000u);
          for (int classIdx = 0; classIdx < (int)(nClasses); ++classIdx) {
            uint classIdxU32 = ((uint)(classIdx));
            uint probIdx = (absLeafBase + classIdxU32);
            float prob = forest_leaves[probIdx];
            float currBest = bestProb;
            if (prob > currBest) {
              bestClass = classIdx;
              bestProb = prob;
            }
          }
          int predictedClass = bestClass;
          uint predictedClassU32 = ((uint)(predictedClass));
          uint voteIdx = (voteBase + predictedClassU32);
          uint currVotes = vote_counts[voteIdx];
          uint newVotes = (currVotes + 1u);
          vote_counts[voteIdx] = newVotes;
        } else {
          float threshold = thresholds[absNodeIdx];
          uint featureIdxU32 = ((uint)(featureIdx));
          uint featureAddr = (sampleBase + featureIdxU32);
          float featureVal = X[featureAddr];
          bool goLeft = (featureVal <= threshold);
          if (goLeft) {
            int leftChild = left_children[absNodeIdx];
            currentNode = leftChild;
          } else {
            int rightChild = right_children[absNodeIdx];
            currentNode = rightChild;
          }
        }
      }
    }
    int finalClass = 0;
    uint maxVotes = 0u;
    for (int c = 0; c < (int)(nClasses); ++c) {
      uint cU32 = ((uint)(c));
      uint voteIdx = (voteBase + cU32);
      uint votes = vote_counts[voteIdx];
      uint currMax = maxVotes;
      if (votes > currMax) {
        finalClass = c;
        maxVotes = votes;
      }
    }
    int result = finalClass;
    predictions[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_ml_forest_regress_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_forest_regress_f32(device const float* X [[buffer(0)]], device const int* feature_indices [[buffer(1)]], device const float* thresholds [[buffer(2)]], device const int* left_children [[buffer(3)]], device const int* right_children [[buffer(4)]], device const int* leaf_value_indices [[buffer(5)]], device const float* forest_leaves [[buffer(6)]], device const int* tree_offsets [[buffer(7)]], device const int* leaf_offsets [[buffer(8)]], device const uint* n_samples [[buffer(9)]], device const uint* n_features [[buffer(10)]], device const uint* n_trees [[buffer(11)]], device float* predictions [[buffer(12)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nFeatures = n_features[0];
  uint nTrees = n_trees[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    float sumPreds = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(nTrees); ++t) {
      uint tU32 = ((uint)(t));
      int treeNodeOffset = tree_offsets[tU32];
      uint treeNodeOffsetU32 = ((uint)(treeNodeOffset));
      int treeLeafOffset = leaf_offsets[tU32];
      uint treeLeafOffsetU32 = ((uint)(treeLeafOffset));
      int currentNode = 0;
      for (int depth = 0; depth < (int)(64u); ++depth) {
        int nodeIdx = currentNode;
        uint nodeIdxU32 = ((uint)(nodeIdx));
        uint absNodeIdx = (treeNodeOffsetU32 + nodeIdxU32);
        int featureIdx = feature_indices[absNodeIdx];
        bool isLeaf = (featureIdx == -1);
        if (isLeaf) {
          int leafIdxI32 = leaf_value_indices[absNodeIdx];
          uint leafIdx = ((uint)(leafIdxI32));
          uint absLeafIdx = (treeLeafOffsetU32 + leafIdx);
          float leafVal = forest_leaves[absLeafIdx];
          float currSum = sumPreds;
          float newSum = (currSum + leafVal);
          sumPreds = newSum;
        } else {
          float threshold = thresholds[absNodeIdx];
          uint featureIdxU32 = ((uint)(featureIdx));
          uint featureAddr = (sampleBase + featureIdxU32);
          float featureVal = X[featureAddr];
          bool goLeft = (featureVal <= threshold);
          if (goLeft) {
            int leftChild = left_children[absNodeIdx];
            currentNode = leftChild;
          } else {
            int rightChild = right_children[absNodeIdx];
            currentNode = rightChild;
          }
        }
      }
    }
    float finalSum = sumPreds;
    float nTreesF = ((float)(nTrees));
    float avgPred = (finalSum / nTreesF);
    predictions[gid] = avgPred;
  }
}

)kernel";
const char* k_metal_navatala_ml_aggregate_importances_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_aggregate_importances_f32(device const float* tree_importances [[buffer(0)]], device const uint* n_trees [[buffer(1)]], device const uint* n_features [[buffer(2)]], device float* importances [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nTrees = n_trees[0];
  uint nFeatures = n_features[0];
  bool inBounds = (gid < nFeatures);
  if (inBounds) {
    float sumImportance = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(nTrees); ++t) {
      uint tU32 = ((uint)(t));
      uint idx = ((tU32 * nFeatures) + gid);
      float imp = tree_importances[idx];
      float currSum = sumImportance;
      float newSum = (currSum + imp);
      sumImportance = newSum;
    }
    float finalSum = sumImportance;
    float nTreesF = ((float)(nTrees));
    float avgImp = (finalSum / nTreesF);
    importances[gid] = avgImp;
  }
}

)kernel";
const char* k_metal_navatala_ml_fil_infer_regression_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_fil_infer_regression_f32(device const float* X [[buffer(0)]], device const int* feature_indices [[buffer(1)]], device const float* thresholds [[buffer(2)]], device const int* left_children [[buffer(3)]], device const int* right_children [[buffer(4)]], device const int* tree_offsets [[buffer(5)]], device const uint* bias [[buffer(6)]], device const uint* n_samples [[buffer(7)]], device const uint* n_features [[buffer(8)]], device const uint* n_trees [[buffer(9)]], device float* predictions [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nFeatures = n_features[0];
  uint nTrees = n_trees[0];
  float biasVal = ((float)(bias[0]));
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    float sumOutput = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(nTrees); ++t) {
      uint tU32 = ((uint)(t));
      int treeOffset = tree_offsets[tU32];
      uint treeOffsetU32 = ((uint)(treeOffset));
      uint nodeIdx = 0u;
      for (int depth = 0; depth < (int)(64u); ++depth) {
        uint currNode = nodeIdx;
        uint absNodeIdx = (treeOffsetU32 + currNode);
        int featureIdx = feature_indices[absNodeIdx];
        float threshold = thresholds[absNodeIdx];
        bool isLeaf = (featureIdx == -1);
        if (isLeaf) {
          float currSum = sumOutput;
          float newSum = (currSum + threshold);
          sumOutput = newSum;
        } else {
          uint featureIdxU32 = ((uint)(featureIdx));
          uint featureAddr = (sampleBase + featureIdxU32);
          float featureVal = X[featureAddr];
          bool goLeft = (featureVal <= threshold);
          if (goLeft) {
            int leftChild = left_children[absNodeIdx];
            uint leftChildU32 = ((uint)(leftChild));
            nodeIdx = leftChildU32;
          } else {
            int rightChild = right_children[absNodeIdx];
            uint rightChildU32 = ((uint)(rightChild));
            nodeIdx = rightChildU32;
          }
        }
      }
    }
    float finalSum = sumOutput;
    float nTreesF = ((float)(nTrees));
    float avgPred = ((finalSum / nTreesF) + biasVal);
    predictions[gid] = avgPred;
  }
}

)kernel";
const char* k_metal_navatala_ml_fil_infer_classification_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_fil_infer_classification_f32(device const float* X [[buffer(0)]], device const int* feature_indices [[buffer(1)]], device const float* thresholds [[buffer(2)]], device const int* left_children [[buffer(3)]], device const int* right_children [[buffer(4)]], device const int* tree_offsets [[buffer(5)]], device const uint* n_samples [[buffer(6)]], device const uint* n_features [[buffer(7)]], device const uint* n_trees [[buffer(8)]], device const uint* n_classes [[buffer(9)]], device int* predictions [[buffer(10)]], device uint* vote_counts [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nFeatures = n_features[0];
  uint nTrees = n_trees[0];
  uint nClasses = n_classes[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    uint voteBase = (gid * nClasses);
    for (int c = 0; c < (int)(nClasses); ++c) {
      uint cU32 = ((uint)(c));
      uint voteIdx = (voteBase + cU32);
      vote_counts[voteIdx] = 0u;
    }
    for (int t = 0; t < (int)(nTrees); ++t) {
      uint tU32 = ((uint)(t));
      int treeOffset = tree_offsets[tU32];
      uint treeOffsetU32 = ((uint)(treeOffset));
      uint nodeIdx = 0u;
      for (int depth = 0; depth < (int)(64u); ++depth) {
        uint currNode = nodeIdx;
        uint absNodeIdx = (treeOffsetU32 + currNode);
        int featureIdx = feature_indices[absNodeIdx];
        float threshold = thresholds[absNodeIdx];
        bool isLeaf = (featureIdx == -1);
        if (isLeaf) {
          uint predictedClass = ((uint)(threshold));
          uint voteIdx = (voteBase + predictedClass);
          uint currVotes = vote_counts[voteIdx];
          uint newVotes = (currVotes + 1u);
          vote_counts[voteIdx] = newVotes;
        } else {
          uint featureIdxU32 = ((uint)(featureIdx));
          uint featureAddr = (sampleBase + featureIdxU32);
          float featureVal = X[featureAddr];
          bool goLeft = (featureVal <= threshold);
          if (goLeft) {
            int leftChild = left_children[absNodeIdx];
            uint leftChildU32 = ((uint)(leftChild));
            nodeIdx = leftChildU32;
          } else {
            int rightChild = right_children[absNodeIdx];
            uint rightChildU32 = ((uint)(rightChild));
            nodeIdx = rightChildU32;
          }
        }
      }
    }
    int finalClass = 0;
    uint maxVotes = 0u;
    for (int c = 0; c < (int)(nClasses); ++c) {
      uint cU32 = ((uint)(c));
      uint voteIdx = (voteBase + cU32);
      uint votes = vote_counts[voteIdx];
      uint currMax = maxVotes;
      if (votes > currMax) {
        finalClass = c;
        maxVotes = votes;
      }
    }
    int result = finalClass;
    predictions[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_ml_svm_predict_classify_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_svm_predict_classify_f32(device const float* support_vectors [[buffer(0)]], device const float* dual_coefs [[buffer(1)]], device const float* X [[buffer(2)]], device const float* bias [[buffer(3)]], device const float* gamma [[buffer(4)]], device const uint* n_samples [[buffer(5)]], device const uint* n_sv [[buffer(6)]], device const uint* n_features [[buffer(7)]], device int* predictions [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nSV = n_sv[0];
  uint nFeatures = n_features[0];
  float biasVal = bias[0];
  float gammaVal = gamma[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    float decisionSum = as_type<float>(0x00000000u);
    for (int sv = 0; sv < (int)(nSV); ++sv) {
      uint svU32 = ((uint)(sv));
      uint svBase = (svU32 * nFeatures);
      float sqDist = as_type<float>(0x00000000u);
      for (int f = 0; f < (int)(nFeatures); ++f) {
        uint fU32 = ((uint)(f));
        uint xIdx = (sampleBase + fU32);
        uint svIdx = (svBase + fU32);
        float xVal = X[xIdx];
        float svVal = support_vectors[svIdx];
        float diff = (xVal - svVal);
        float diffSq = (diff * diff);
        float currSqDist = sqDist;
        float newSqDist = (currSqDist + diffSq);
        sqDist = newSqDist;
      }
      float finalSqDist = sqDist;
      float negGammaDist = (as_type<float>(0xbf800000u) * (gammaVal * finalSqDist));
      float kernelVal = exp(negGammaDist);
      float alpha = dual_coefs[svU32];
      float contribution = (alpha * kernelVal);
      float currSum = decisionSum;
      float newSum = (currSum + contribution);
      decisionSum = newSum;
    }
    float finalSum = decisionSum;
    float decision = (finalSum + biasVal);
    bool isPositive = (decision >= as_type<float>(0x00000000u));
    if (isPositive) {
      predictions[gid] = 1;
    } else {
      predictions[gid] = -1;
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_svm_predict_regress_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_svm_predict_regress_f32(device const float* support_vectors [[buffer(0)]], device const float* dual_coefs [[buffer(1)]], device const float* X [[buffer(2)]], device const float* bias [[buffer(3)]], device const float* gamma [[buffer(4)]], device const uint* n_samples [[buffer(5)]], device const uint* n_sv [[buffer(6)]], device const uint* n_features [[buffer(7)]], device float* predictions [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nSV = n_sv[0];
  uint nFeatures = n_features[0];
  float biasVal = bias[0];
  float gammaVal = gamma[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    float decisionSum = as_type<float>(0x00000000u);
    for (int sv = 0; sv < (int)(nSV); ++sv) {
      uint svU32 = ((uint)(sv));
      uint svBase = (svU32 * nFeatures);
      float sqDist = as_type<float>(0x00000000u);
      for (int f = 0; f < (int)(nFeatures); ++f) {
        uint fU32 = ((uint)(f));
        uint xIdx = (sampleBase + fU32);
        uint svIdx = (svBase + fU32);
        float xVal = X[xIdx];
        float svVal = support_vectors[svIdx];
        float diff = (xVal - svVal);
        float diffSq = (diff * diff);
        float currSqDist = sqDist;
        float newSqDist = (currSqDist + diffSq);
        sqDist = newSqDist;
      }
      float finalSqDist = sqDist;
      float negGammaDist = (as_type<float>(0xbf800000u) * (gammaVal * finalSqDist));
      float kernelVal = exp(negGammaDist);
      float alpha = dual_coefs[svU32];
      float contribution = (alpha * kernelVal);
      float currSum = decisionSum;
      float newSum = (currSum + contribution);
      decisionSum = newSum;
    }
    float finalSum = decisionSum;
    float prediction = (finalSum + biasVal);
    predictions[gid] = prediction;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_decision_function_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_decision_function_f32(device const float* support_vectors [[buffer(0)]], device const float* dual_coefs [[buffer(1)]], device const float* X [[buffer(2)]], device const float* bias [[buffer(3)]], device const float* gamma [[buffer(4)]], device const uint* n_samples [[buffer(5)]], device const uint* n_sv [[buffer(6)]], device const uint* n_features [[buffer(7)]], device float* decision_values [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nSV = n_sv[0];
  uint nFeatures = n_features[0];
  float biasVal = bias[0];
  float gammaVal = gamma[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    float decisionSum = as_type<float>(0x00000000u);
    for (int sv = 0; sv < (int)(nSV); ++sv) {
      uint svU32 = ((uint)(sv));
      uint svBase = (svU32 * nFeatures);
      float sqDist = as_type<float>(0x00000000u);
      for (int f = 0; f < (int)(nFeatures); ++f) {
        uint fU32 = ((uint)(f));
        uint xIdx = (sampleBase + fU32);
        uint svIdx = (svBase + fU32);
        float xVal = X[xIdx];
        float svVal = support_vectors[svIdx];
        float diff = (xVal - svVal);
        float diffSq = (diff * diff);
        float currSqDist = sqDist;
        float newSqDist = (currSqDist + diffSq);
        sqDist = newSqDist;
      }
      float finalSqDist = sqDist;
      float negGammaDist = (as_type<float>(0xbf800000u) * (gammaVal * finalSqDist));
      float kernelVal = exp(negGammaDist);
      float alpha = dual_coefs[svU32];
      float contribution = (alpha * kernelVal);
      float currSum = decisionSum;
      float newSum = (currSum + contribution);
      decisionSum = newSum;
    }
    float finalSum = decisionSum;
    float decision = (finalSum + biasVal);
    decision_values[gid] = decision;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_r_b_f_kernel_matrix_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_r_b_f_kernel_matrix_f32(device const float* X1 [[buffer(0)]], device const float* X2 [[buffer(1)]], device const float* gamma [[buffer(2)]], device const uint* n1 [[buffer(3)]], device const uint* n2 [[buffer(4)]], device const uint* n_features [[buffer(5)]], device float* K [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gidX = ((uint)(int(__gid.x)));
  uint gidY = ((uint)(int(__gid.y)));
  uint nRows = n1[0];
  uint nCols = n2[0];
  uint nFeatures = n_features[0];
  float gammaVal = gamma[0];
  bool inBoundsX = (gidX < nRows);
  bool inBoundsY = (gidY < nCols);
  bool inBounds = (inBoundsX && inBoundsY);
  if (inBounds) {
    uint x1Base = (gidX * nFeatures);
    uint x2Base = (gidY * nFeatures);
    float sqDist = as_type<float>(0x00000000u);
    for (int f = 0; f < (int)(nFeatures); ++f) {
      uint fU32 = ((uint)(f));
      uint x1Idx = (x1Base + fU32);
      uint x2Idx = (x2Base + fU32);
      float x1Val = X1[x1Idx];
      float x2Val = X2[x2Idx];
      float diff = (x1Val - x2Val);
      float diffSq = (diff * diff);
      float currSqDist = sqDist;
      float newSqDist = (currSqDist + diffSq);
      sqDist = newSqDist;
    }
    float finalSqDist = sqDist;
    float negGammaDist = (as_type<float>(0xbf800000u) * (gammaVal * finalSqDist));
    float kernelVal = exp(negGammaDist);
    uint outIdx = ((gidX * nCols) + gidY);
    K[outIdx] = kernelVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_linear_kernel_matrix_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_linear_kernel_matrix_f32(device const float* X1 [[buffer(0)]], device const float* X2 [[buffer(1)]], device const uint* n1 [[buffer(2)]], device const uint* n2 [[buffer(3)]], device const uint* n_features [[buffer(4)]], device float* K [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gidX = ((uint)(int(__gid.x)));
  uint gidY = ((uint)(int(__gid.y)));
  uint nRows = n1[0];
  uint nCols = n2[0];
  uint nFeatures = n_features[0];
  bool inBoundsX = (gidX < nRows);
  bool inBoundsY = (gidY < nCols);
  bool inBounds = (inBoundsX && inBoundsY);
  if (inBounds) {
    uint x1Base = (gidX * nFeatures);
    uint x2Base = (gidY * nFeatures);
    float dotProduct = as_type<float>(0x00000000u);
    for (int f = 0; f < (int)(nFeatures); ++f) {
      uint fU32 = ((uint)(f));
      uint x1Idx = (x1Base + fU32);
      uint x2Idx = (x2Base + fU32);
      float x1Val = X1[x1Idx];
      float x2Val = X2[x2Idx];
      float product = (x1Val * x2Val);
      float currDot = dotProduct;
      float newDot = (currDot + product);
      dotProduct = newDot;
    }
    float kernelVal = dotProduct;
    uint outIdx = ((gidX * nCols) + gidY);
    K[outIdx] = kernelVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_polynomial_kernel_matrix_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_polynomial_kernel_matrix_f32(device const float* X1 [[buffer(0)]], device const float* X2 [[buffer(1)]], device const float* gamma [[buffer(2)]], device const float* coef0 [[buffer(3)]], device const int* degree [[buffer(4)]], device const uint* n1 [[buffer(5)]], device const uint* n2 [[buffer(6)]], device const uint* n_features [[buffer(7)]], device float* K [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gidX = ((uint)(int(__gid.x)));
  uint gidY = ((uint)(int(__gid.y)));
  uint nRows = n1[0];
  uint nCols = n2[0];
  uint nFeatures = n_features[0];
  float gammaVal = gamma[0];
  float coef0Val = coef0[0];
  int degreeVal = degree[0];
  bool inBoundsX = (gidX < nRows);
  bool inBoundsY = (gidY < nCols);
  bool inBounds = (inBoundsX && inBoundsY);
  if (inBounds) {
    uint x1Base = (gidX * nFeatures);
    uint x2Base = (gidY * nFeatures);
    float dotProduct = as_type<float>(0x00000000u);
    for (int f = 0; f < (int)(nFeatures); ++f) {
      uint fU32 = ((uint)(f));
      uint x1Idx = (x1Base + fU32);
      uint x2Idx = (x2Base + fU32);
      float x1Val = X1[x1Idx];
      float x2Val = X2[x2Idx];
      float product = (x1Val * x2Val);
      float currDot = dotProduct;
      float newDot = (currDot + product);
      dotProduct = newDot;
    }
    float finalDot = dotProduct;
    float scaled = (gammaVal * finalDot);
    float base = (scaled + coef0Val);
    float result = as_type<float>(0x3f800000u);
    uint degreeU32 = ((uint)(degreeVal));
    for (int d = 0; d < (int)(degreeU32); ++d) {
      float currResult = result;
      float newResult = (currResult * base);
      result = newResult;
    }
    float kernelVal = result;
    uint outIdx = ((gidX * nCols) + gidY);
    K[outIdx] = kernelVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_core_distances_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_core_distances_f32(device const float* distances [[buffer(0)]], device const uint* n_samples [[buffer(1)]], device const uint* min_samples [[buffer(2)]], device float* core_dists [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint minSamples = min_samples[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint rowBase = (gid * nSamples);
    float kthDist = as_type<float>(0x7149f2cau);
    uint countSmaller = 0u;
    for (int iter = 0; iter < (int)(nSamples); ++iter) {
      uint iterU32 = ((uint)(iter));
      uint distIdx = (rowBase + iterU32);
      float candidateDist = distances[distIdx];
      bool notSelf = (candidateDist > as_type<float>(0x00000000u));
      if (notSelf) {
        countSmaller = 0u;
        for (int j = 0; j < (int)(nSamples); ++j) {
          uint jU32 = ((uint)(j));
          uint jDistIdx = (rowBase + jU32);
          float jDist = distances[jDistIdx];
          bool isPositive = (jDist > as_type<float>(0x00000000u));
          bool isLeCandidate = (jDist <= candidateDist);
          bool shouldCount = (isPositive && isLeCandidate);
          if (shouldCount) {
            uint currCount = countSmaller;
            uint newCount = (currCount + 1u);
            countSmaller = newCount;
          }
        }
        uint finalCount = countSmaller;
        bool isKthOrMore = (finalCount >= minSamples);
        if (isKthOrMore) {
          float currKth = kthDist;
          bool isBetter = (candidateDist < currKth);
          if (isBetter) {
            kthDist = candidateDist;
          }
        }
      }
    }
    float finalCoreDist = kthDist;
    core_dists[gid] = finalCoreDist;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_mutual_reachability_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_mutual_reachability_f32(device const float* distances [[buffer(0)]], device const float* core_dists [[buffer(1)]], device const uint* n_samples [[buffer(2)]], device float* mutual_reach [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint totalPairs = (nSamples * nSamples);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint i = (gid / nSamples);
    uint j = (gid % nSamples);
    float dist = distances[gid];
    float coreI = core_dists[i];
    float coreJ = core_dists[j];
    bool coreIGtJ = (coreI > coreJ);
    float maxCore = as_type<float>(0x00000000u);
    if (coreIGtJ) {
      maxCore = coreI;
    } else {
      maxCore = coreJ;
    }
    float maxCoreVal = maxCore;
    bool distGtCore = (dist > maxCoreVal);
    float mreach = as_type<float>(0x00000000u);
    if (distGtCore) {
      mreach = dist;
    } else {
      mreach = maxCoreVal;
    }
    float finalMreach = mreach;
    mutual_reach[gid] = finalMreach;
  }
}

)kernel";
const char* k_metal_navatala_ml_extract_cluster_labels_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_extract_cluster_labels_f32(device const uint* selected_clusters [[buffer(0)]], device const int* point_cluster_map [[buffer(1)]], device const int* cluster_parent_map [[buffer(2)]], device const uint* n_samples [[buffer(3)]], device const uint* n_clusters [[buffer(4)]], device int* labels [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nClusters = n_clusters[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    int initialCluster = point_cluster_map[gid];
    int currentCluster = initialCluster;
    int foundLabel = -1;
    for (int level = 0; level < (int)(64u); ++level) {
      int currClust = currentCluster;
      bool isValidCluster = (currClust >= 0);
      if (isValidCluster) {
        uint currClustU32 = ((uint)(currClust));
        bool inClusterBounds = (currClustU32 < nClusters);
        if (inClusterBounds) {
          uint isSelected = selected_clusters[currClustU32];
          bool clusterSelected = (isSelected > 0u);
          if (clusterSelected) {
            foundLabel = currClust;
          } else {
            int parentCluster = cluster_parent_map[currClustU32];
            currentCluster = parentCluster;
          }
        }
      }
    }
    int finalLabel = foundLabel;
    labels[gid] = finalLabel;
  }
}

)kernel";
const char* k_metal_navatala_ml_extract_cluster_labels_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_extract_cluster_labels_f64(device const uint* selected_clusters [[buffer(0)]], device const int* point_cluster_map [[buffer(1)]], device const int* cluster_parent_map [[buffer(2)]], device const uint* n_samples [[buffer(3)]], device const uint* n_clusters [[buffer(4)]], device int* labels [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nClusters = n_clusters[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    int initialCluster = point_cluster_map[gid];
    int currentCluster = initialCluster;
    int foundLabel = -1;
    for (int level = 0; level < (int)(64u); ++level) {
      int currClust = currentCluster;
      bool isValidCluster = (currClust >= 0);
      if (isValidCluster) {
        uint currClustU32 = ((uint)(currClust));
        bool inClusterBounds = (currClustU32 < nClusters);
        if (inClusterBounds) {
          uint isSelected = selected_clusters[currClustU32];
          bool clusterSelected = (isSelected > 0u);
          if (clusterSelected) {
            foundLabel = currClust;
          } else {
            int parentCluster = cluster_parent_map[currClustU32];
            currentCluster = parentCluster;
          }
        }
      }
    }
    int finalLabel = foundLabel;
    labels[gid] = finalLabel;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_probabilities_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_probabilities_f32(device const int* labels [[buffer(0)]], device const float* lambdas [[buffer(1)]], device const float* cluster_lambdas_max [[buffer(2)]], device const uint* n_samples [[buffer(3)]], device const uint* n_clusters [[buffer(4)]], device float* probabilities [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    int label = labels[gid];
    bool isNoise = (label < 0);
    if (isNoise) {
      probabilities[gid] = as_type<float>(0x00000000u);
    } else {
      float lambda = lambdas[gid];
      uint labelU32 = ((uint)(label));
      float lambdaMax = cluster_lambdas_max[labelU32];
      bool lambdaMaxIsZero = (lambdaMax <= as_type<float>(0x00000000u));
      float prob = as_type<float>(0x00000000u);
      if (lambdaMaxIsZero) {
        prob = as_type<float>(0x3f800000u);
      } else {
        float rawProb = (lambda / lambdaMax);
        bool probGt1 = (rawProb > as_type<float>(0x3f800000u));
        if (probGt1) {
          prob = as_type<float>(0x3f800000u);
        } else {
          bool probLt0 = (rawProb < as_type<float>(0x00000000u));
          if (probLt0) {
            prob = as_type<float>(0x00000000u);
          } else {
            prob = rawProb;
          }
        }
      }
      float finalProb = prob;
      probabilities[gid] = finalProb;
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_cluster_stability_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_cluster_stability_f32(device const float* point_lambdas [[buffer(0)]], device const int* cluster_ids [[buffer(1)]], device const uint* n_edges [[buffer(2)]], device const uint* n_clusters [[buffer(3)]], device float* stabilities [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nEdges = n_edges[0];
  uint nClusters = n_clusters[0];
  bool inBounds = (gid < nClusters);
  if (inBounds) {
    float sumLambdas = as_type<float>(0x00000000u);
    int targetCluster = ((int)(gid));
    for (int e = 0; e < (int)(nEdges); ++e) {
      uint eU32 = ((uint)(e));
      int edgeCluster = cluster_ids[eU32];
      bool isTargetCluster = (edgeCluster == targetCluster);
      if (isTargetCluster) {
        float lambda = point_lambdas[eU32];
        float currSum = sumLambdas;
        float newSum = (currSum + lambda);
        sumLambdas = newSum;
      }
    }
    float finalStability = sumLambdas;
    stabilities[gid] = finalStability;
  }
}

)kernel";
const char* k_metal_navatala_ml_initialize_labels_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_initialize_labels_f32(device const uint* n_samples [[buffer(0)]], device int* labels [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    labels[gid] = -1;
  }
}

)kernel";
const char* k_metal_navatala_ml_initialize_labels_f64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_initialize_labels_f64(device const uint* n_samples [[buffer(0)]], device int* labels [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    labels[gid] = -1;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_fuzzy_simplicial_set_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_fuzzy_simplicial_set_f32(device const int* knn_indices [[buffer(0)]], device const float* knn_dists [[buffer(1)]], device const uint* n_samples [[buffer(2)]], device const uint* n_neighbors [[buffer(3)]], device const float* local_connectivity [[buffer(4)]], device float* membership [[buffer(5)]], device float* rho [[buffer(6)]], device float* sigma [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nNeighbors = n_neighbors[0];
  float localConn = local_connectivity[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nNeighbors);
    uint localConnFloor = ((uint)(localConn));
    float localConnFrac = (localConn - ((float)(localConnFloor)));
    uint idxFloor = (sampleBase + localConnFloor);
    uint idxCeil = (idxFloor + 1u);
    uint maxIdx = ((sampleBase + nNeighbors) - 1u);
    uint idxCeilClamped = (((idxCeil < maxIdx)) ? (idxCeil) : (maxIdx));
    float distFloor = knn_dists[idxFloor];
    float distCeil = knn_dists[idxCeilClamped];
    float rhoVal = (((as_type<float>(0x3f800000u) - localConnFrac) * distFloor) + (localConnFrac * distCeil));
    rho[gid] = rhoVal;
    float nNeighborsF = ((float)(nNeighbors));
    float targetSum = (log(nNeighborsF) / log(as_type<float>(0x40000000u)));
    float sigmaLow = as_type<float>(0x2edbe6ffu);
    float sigmaHigh = as_type<float>(0x447a0000u);
    float sigmaMid = as_type<float>(0x00000000u);
    for (int iter = 0; iter < (int)(64u); ++iter) {
      float low = sigmaLow;
      float high = sigmaHigh;
      float mid = ((low + high) / as_type<float>(0x40000000u));
      sigmaMid = mid;
      float membershipSum = as_type<float>(0x00000000u);
      for (int k = 0; k < (int)(nNeighbors); ++k) {
        uint kU32 = ((uint)(k));
        uint distIdx = (sampleBase + kU32);
        float dist = knn_dists[distIdx];
        float distMinusRho = (dist - rhoVal);
        float negScaled = ((as_type<float>(0x00000000u) - distMinusRho) / mid);
        float mem = exp(negScaled);
        float memClamped = (((mem > as_type<float>(0x3f800000u))) ? (as_type<float>(0x3f800000u)) : (mem));
        float currSum = membershipSum;
        float newSum = (currSum + memClamped);
        membershipSum = newSum;
      }
      float sum = membershipSum;
      if (sum > targetSum) {
        sigmaHigh = mid;
      } else {
        sigmaLow = mid;
      }
    }
    float finalSigma = sigmaMid;
    sigma[gid] = finalSigma;
    for (int k = 0; k < (int)(nNeighbors); ++k) {
      uint kU32 = ((uint)(k));
      uint memIdx = (sampleBase + kU32);
      float dist = knn_dists[memIdx];
      float distMinusRho = (dist - rhoVal);
      float negScaled = ((as_type<float>(0x00000000u) - distMinusRho) / finalSigma);
      float mem = exp(negScaled);
      float memClamped = (((mem > as_type<float>(0x3f800000u))) ? (as_type<float>(0x3f800000u)) : (mem));
      membership[memIdx] = memClamped;
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_attraction_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_attraction_f32(device const float* embedding [[buffer(0)]], device const int* edge_from [[buffer(1)]], device const int* edge_to [[buffer(2)]], device const float* edge_weights [[buffer(3)]], device const uint* n_edges [[buffer(4)]], device const uint* n_components [[buffer(5)]], device const float* a_param [[buffer(6)]], device const float* b_param [[buffer(7)]], device float* gradients [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nEdges = n_edges[0];
  uint nComp = n_components[0];
  float a = a_param[0];
  float b = b_param[0];
  bool inBounds = (gid < nEdges);
  if (inBounds) {
    int fromIdx = edge_from[gid];
    int toIdx = edge_to[gid];
    float weight = edge_weights[gid];
    uint fromIdxU32 = ((uint)(fromIdx));
    uint toIdxU32 = ((uint)(toIdx));
    uint fromBase = (fromIdxU32 * nComp);
    uint toBase = (toIdxU32 * nComp);
    float distSq = as_type<float>(0x38d1b717u);
    for (int d = 0; d < (int)(nComp); ++d) {
      uint dU32 = ((uint)(d));
      uint fromAddr = (fromBase + dU32);
      uint toAddr = (toBase + dU32);
      float fromVal = embedding[fromAddr];
      float toVal = embedding[toAddr];
      float diff = (fromVal - toVal);
      float diffSq = (diff * diff);
      float currDistSq = distSq;
      float newDistSq = (currDistSq + diffSq);
      distSq = newDistSq;
    }
    float dSq = distSq;
    float bMinus1 = (b - as_type<float>(0x3f800000u));
    float logDSq = log(dSq);
    float powTerm = exp((bMinus1 * logDSq));
    float dPow2b = exp((b * logDSq));
    float denom = (as_type<float>(0x3f800000u) + (a * dPow2b));
    float gradScale = ((((as_type<float>(0xc0000000u) * a) * b) * (weight * powTerm)) / denom);
    for (int d = 0; d < (int)(nComp); ++d) {
      uint dU32 = ((uint)(d));
      uint fromAddr = (fromBase + dU32);
      uint toAddr = (toBase + dU32);
      float fromVal = embedding[fromAddr];
      float toVal = embedding[toAddr];
      float diff = (fromVal - toVal);
      float grad = (gradScale * diff);
      float currGrad = gradients[fromAddr];
      float newGrad = (currGrad + grad);
      gradients[fromAddr] = newGrad;
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_repulsion_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_repulsion_f32(device const float* embedding [[buffer(0)]], device const int* negative_samples [[buffer(1)]], device const uint* n_samples [[buffer(2)]], device const uint* n_neg [[buffer(3)]], device const uint* n_components [[buffer(4)]], device const float* a_param [[buffer(5)]], device const float* b_param [[buffer(6)]], device const float* gamma [[buffer(7)]], device float* gradients [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nNeg = n_neg[0];
  uint nComp = n_components[0];
  float a = a_param[0];
  float b = b_param[0];
  float gammaVal = gamma[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nComp);
    uint negBase = (gid * nNeg);
    for (int n = 0; n < (int)(nNeg); ++n) {
      uint nU32 = ((uint)(n));
      uint negSampleIdx = (negBase + nU32);
      int negIdx = negative_samples[negSampleIdx];
      bool validNeg = (negIdx >= 0);
      if (validNeg) {
        uint negIdxU32 = ((uint)(negIdx));
        uint negEmbBase = (negIdxU32 * nComp);
        float distSqNeg = as_type<float>(0x38d1b717u);
        for (int d = 0; d < (int)(nComp); ++d) {
          uint dU32 = ((uint)(d));
          uint iAddr = (sampleBase + dU32);
          uint jAddr = (negEmbBase + dU32);
          float iVal = embedding[iAddr];
          float jVal = embedding[jAddr];
          float diff = (iVal - jVal);
          float diffSq = (diff * diff);
          float currDSq = distSqNeg;
          float newDSq = (currDSq + diffSq);
          distSqNeg = newDSq;
        }
        float dSq = distSqNeg;
        float logDSqNeg = log(dSq);
        float dPow2b = exp((b * logDSqNeg));
        float denom = (as_type<float>(0x3f800000u) + (a * dPow2b));
        float gradScale = (((as_type<float>(0x40000000u) * gammaVal) * b) / (dSq * denom));
        for (int d = 0; d < (int)(nComp); ++d) {
          uint dU32 = ((uint)(d));
          uint iAddr = (sampleBase + dU32);
          uint jAddr = (negEmbBase + dU32);
          float iVal = embedding[iAddr];
          float jVal = embedding[jAddr];
          float diff = (iVal - jVal);
          float grad = (gradScale * diff);
          float gradClamped = (((grad > as_type<float>(0x40800000u))) ? (as_type<float>(0x40800000u)) : ((((grad < as_type<float>(0xc0800000u))) ? (as_type<float>(0xc0800000u)) : (grad))));
          float currGrad = gradients[iAddr];
          float newGrad = (currGrad + gradClamped);
          gradients[iAddr] = newGrad;
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_optimize_layout_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_optimize_layout_f32(device const float* embedding [[buffer(0)]], device const float* gradients [[buffer(1)]], device const uint* n_samples [[buffer(2)]], device const uint* n_components [[buffer(3)]], device const float* alpha [[buffer(4)]], device float* embedding_out [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nComp = n_components[0];
  uint totalSize = (nSamples * nComp);
  float alphaVal = alpha[0];
  bool inBounds = (gid < totalSize);
  if (inBounds) {
    float embVal = embedding[gid];
    float gradVal = gradients[gid];
    float gradClipped = (((gradVal > as_type<float>(0x40800000u))) ? (as_type<float>(0x40800000u)) : ((((gradVal < as_type<float>(0xc0800000u))) ? (as_type<float>(0xc0800000u)) : (gradVal))));
    float update = (alphaVal * gradClipped);
    float newVal = (embVal + update);
    embedding_out[gid] = newVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_initialize_embedding_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_initialize_embedding_f32(device const uint* seed [[buffer(0)]], device const uint* n_samples [[buffer(1)]], device const uint* n_components [[buffer(2)]], device const float* spread [[buffer(3)]], device float* embedding [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nComp = n_components[0];
  uint totalSize = (nSamples * nComp);
  uint seedVal = seed[0];
  float spreadVal = spread[0];
  bool inBounds = (gid < totalSize);
  if (inBounds) {
    uint state = (seedVal + gid);
    uint a = 1664525u;
    uint c = 1013904223u;
    uint newState = ((a * state) + c);
    float randBitsF = ((float)(newState));
    float maxU32 = as_type<float>(0x4f800000u);
    float normalized = (randBitsF / maxU32);
    float centered = ((normalized * as_type<float>(0x40000000u)) - as_type<float>(0x3f800000u));
    float scaled = (centered * spreadVal);
    embedding[gid] = scaled;
  }
}

)kernel";
const char* k_metal_navatala_ml_clear_gradients_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_clear_gradients_f32(device const uint* n_samples [[buffer(0)]], device const uint* n_components [[buffer(1)]], device float* gradients [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nComp = n_components[0];
  uint totalSize = (nSamples * nComp);
  bool inBounds = (gid < totalSize);
  if (inBounds) {
    gradients[gid] = as_type<float>(0x00000000u);
  }
}

)kernel";
const char* k_metal_navatala_ml_differencing_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_differencing_f32(device const float* y [[buffer(0)]], device const uint* n_obs [[buffer(1)]], device const uint* batch_size [[buffer(2)]], device float* y_diff [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nObs = n_obs[0];
  uint batchSize = batch_size[0];
  uint nDiff = (nObs - 1u);
  uint totalOutputs = (nDiff * batchSize);
  bool inBounds = (gid < totalOutputs);
  if (inBounds) {
    uint batch = (gid / nDiff);
    uint t = (gid % nDiff);
    uint currIdx = ((batch * nObs) + (t + 1u));
    uint prevIdx = ((batch * nObs) + t);
    float currVal = y[currIdx];
    float prevVal = y[prevIdx];
    float diff = (currVal - prevVal);
    y_diff[gid] = diff;
  }
}

)kernel";
const char* k_metal_navatala_ml_holt_winters_forecast_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_holt_winters_forecast_f32(device const float* level [[buffer(0)]], device const float* trend [[buffer(1)]], device const float* season [[buffer(2)]], device const uint* period [[buffer(3)]], device const uint* batch_size [[buffer(4)]], device const uint* horizon [[buffer(5)]], device float* forecasts [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint horizonVal = horizon[0];
  uint batchSize = batch_size[0];
  uint periodVal = period[0];
  uint totalOutputs = (horizonVal * batchSize);
  bool inBounds = (gid < totalOutputs);
  if (inBounds) {
    uint batch = (gid / horizonVal);
    uint h = ((gid % horizonVal) + 1u);
    float levelVal = level[batch];
    float trendVal = trend[batch];
    uint seasonIdx = ((batch * periodVal) + ((h - 1u) % periodVal));
    float seasonVal = season[seasonIdx];
    float hF = ((float)(h));
    float forecast = ((levelVal + (hF * trendVal)) + seasonVal);
    forecasts[gid] = forecast;
  }
}

)kernel";
const char* k_metal_navatala_ml_ar_predict_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_ar_predict_f32(device const float* y [[buffer(0)]], device const float* phi [[buffer(1)]], device const float* intercept [[buffer(2)]], device const uint* p [[buffer(3)]], device const uint* batch_size [[buffer(4)]], device float* predictions [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint batchSize = batch_size[0];
  uint pVal = p[0];
  bool inBounds = (gid < batchSize);
  if (inBounds) {
    float pred = intercept[gid];
    float sumAR = pred;
    for (int i = 0; i < (int)(pVal); ++i) {
      uint iU32 = ((uint)(i));
      uint yIdx = ((gid * pVal) + iU32);
      float yVal = y[yIdx];
      uint phiIdx = yIdx;
      float phiVal = phi[phiIdx];
      float term = (phiVal * yVal);
      float currSum = sumAR;
      float newSum = (currSum + term);
      sumAR = newSum;
    }
    float finalPred = sumAR;
    predictions[gid] = finalPred;
  }
}

)kernel";
const char* k_metal_navatala_ml_shap_values_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_shap_values_f32(device const float* predictions [[buffer(0)]], device const float* weights [[buffer(1)]], device const float* coalitions [[buffer(2)]], device const uint* nSamples [[buffer(3)]], device const uint* nFeatures [[buffer(4)]], device float* shapValues [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamplesVal = nSamples[0];
  uint nFeaturesVal = nFeatures[0];
  uint featureIdx = gid;
  bool inBounds = (featureIdx < nFeaturesVal);
  if (inBounds) {
    float weightedSum = as_type<float>(0x00000000u);
    float totalWeight = as_type<float>(0x00000000u);
    for (int s = 0; s < (int)(nSamplesVal); ++s) {
      uint sU32 = ((uint)(s));
      uint coalIdx = ((sU32 * nFeaturesVal) + featureIdx);
      float coalMembership = coalitions[coalIdx];
      float w = weights[sU32];
      float pred = predictions[sU32];
      float contribution = (w * (coalMembership * pred));
      float currentSum = weightedSum;
      float newSum = (currentSum + contribution);
      weightedSum = newSum;
      float weightContrib = (w * coalMembership);
      float currentTotalW = totalWeight;
      float newTotalW = (currentTotalW + weightContrib);
      totalWeight = newTotalW;
    }
    float finalSum = weightedSum;
    float finalTotalW = totalWeight;
    float eps = as_type<float>(0x2edbe6ffu);
    float safeDenom = (finalTotalW + eps);
    float shapVal = (finalSum / safeDenom);
    shapValues[featureIdx] = shapVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_coalition_marginal_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_coalition_marginal_f32(device const float* predWith [[buffer(0)]], device const float* predWithout [[buffer(1)]], device const uint* nSamples [[buffer(2)]], device float* marginals [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamplesVal = nSamples[0];
  uint sampleIdx = gid;
  bool inBounds = (sampleIdx < nSamplesVal);
  if (inBounds) {
    float pWith = predWith[sampleIdx];
    float pWithout = predWithout[sampleIdx];
    float marginal = (pWith - pWithout);
    marginals[sampleIdx] = marginal;
  }
}

)kernel";
const char* k_metal_navatala_ml_feature_interaction_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_feature_interaction_f32(device const float* jointContrib [[buffer(0)]], device const float* shapValues [[buffer(1)]], device const uint* nFeatures [[buffer(2)]], device float* interactions [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nFeaturesVal = nFeatures[0];
  uint totalElems = (nFeaturesVal * nFeaturesVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / nFeaturesVal);
    uint j = (gid % nFeaturesVal);
    float phiIJ = jointContrib[gid];
    float phiI = shapValues[i];
    float phiJ = shapValues[j];
    float sumIndividual = (phiI + phiJ);
    float interaction = (phiIJ - sumIndividual);
    interactions[gid] = interaction;
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_correlations_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_correlations_f32(device const float* X [[buffer(0)]], device const float* residual [[buffer(1)]], device const uint* nRows [[buffer(2)]], device const uint* nCols [[buffer(3)]], device float* correlations [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nRows[0];
  uint d = nCols[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    uint colOffset = (gid * n);
    float accum = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(n); ++i) {
      uint iU32 = ((uint)(i));
      uint xIdx = (iU32 + colOffset);
      float xVal = X[xIdx];
      float rVal = residual[iU32];
      float prod = (xVal * rVal);
      float currentAccum = accum;
      float newAccum = (currentAccum + prod);
      accum = newAccum;
    }
    float corr = accum;
    correlations[gid] = corr;
  }
}

)kernel";
const char* k_metal_navatala_ml_find_max_correlation_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_find_max_correlation_f32(device const float* correlations [[buffer(0)]], device const uint* activeMask [[buffer(1)]], device const uint* nCols [[buffer(2)]], device uint* maxIndex [[buffer(3)]], device float* maxValue [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float svals[256];
  threadgroup uint sidxs[256];
  float otherVal = as_type<float>(0x00000000u);
  uint otherIdx = 0u;
  float myVal = as_type<float>(0x00000000u);
  uint myIdx = 0u;
  bool takeOther = false;
  uint d = nCols[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    uint mask = activeMask[gid];
    bool isActive = (mask == 1u);
    float corr = correlations[gid];
    float absCorr = abs(corr);
    float val = ((isActive) ? (absCorr) : (as_type<float>(0xff7fffffu)));
    svals[lid] = val;
    sidxs[lid] = gid;
  } else {
    svals[lid] = as_type<float>(0xff7fffffu);
    sidxs[lid] = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint maxCorrF32RedStride = 128u;
  for (int maxCorrF32RedStep = 0; maxCorrF32RedStep < (int)(8); ++maxCorrF32RedStep) {
    uint maxCorrF32Stride = maxCorrF32RedStride;
    if (lid < maxCorrF32Stride) {
      otherVal = svals[(lid + maxCorrF32Stride)];
      otherIdx = sidxs[(lid + maxCorrF32Stride)];
      myVal = svals[lid];
      myIdx = sidxs[lid];
      takeOther = (otherVal > myVal);
      if (takeOther) {
        svals[lid] = otherVal;
        sidxs[lid] = otherIdx;
      }
    }
    uint maxCorrF32StrideToHalve = maxCorrF32RedStride;
    uint maxCorrF32NextStride = (maxCorrF32StrideToHalve >> 1u);
    maxCorrF32RedStride = maxCorrF32NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    maxValue[0] = svals[0];
    maxIndex[0] = sidxs[0];
  }
}

)kernel";
const char* k_metal_navatala_ml_update_coefs_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_update_coefs_f32(device const float* equiangular [[buffer(0)]], device const float* gamma [[buffer(1)]], device const uint* activeMask [[buffer(2)]], device const uint* nCols [[buffer(3)]], device float* beta [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint d = nCols[0];
  float gammaVal = gamma[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    uint mask = activeMask[gid];
    bool isActive = (mask == 1u);
    if (isActive) {
      float betaJ = beta[gid];
      float eqJ = equiangular[gid];
      float update = (gammaVal * eqJ);
      float newBeta = (betaJ + update);
      beta[gid] = newBeta;
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_compute_equiangular_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_equiangular_f32(device const float* correlations [[buffer(0)]], device const uint* activeMask [[buffer(1)]], device const uint* nCols [[buffer(2)]], device float* equiangular [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint d = nCols[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    uint mask = activeMask[gid];
    bool isActive = (mask == 1u);
    if (isActive) {
      float corr = correlations[gid];
      bool positive = (corr > as_type<float>(0x00000000u));
      bool negative = (corr < as_type<float>(0x00000000u));
      float signVal = ((positive) ? (as_type<float>(0x3f800000u)) : (((negative) ? (as_type<float>(0xbf800000u)) : (as_type<float>(0x00000000u)))));
      equiangular[gid] = signVal;
    } else {
      equiangular[gid] = as_type<float>(0x00000000u);
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_evaluate_fitness_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_evaluate_fitness_f32(device const float* predictions [[buffer(0)]], device const float* targets [[buffer(1)]], device const uint* popSize [[buffer(2)]], device const uint* nSamples [[buffer(3)]], device float* fitness [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint popSizeVal = popSize[0];
  uint nSamplesVal = nSamples[0];
  bool inBounds = (gid < popSizeVal);
  if (inBounds) {
    uint baseIdx = (gid * nSamplesVal);
    float mseAccum = as_type<float>(0x00000000u);
    int nSamplesI32 = ((int)(nSamplesVal));
    for (int j = 0; j < (int)(nSamplesI32); ++j) {
      uint jU32 = ((uint)(j));
      uint predIdx = (baseIdx + jU32);
      float pred = predictions[predIdx];
      float target = targets[jU32];
      float diff = (pred - target);
      float sqErr = (diff * diff);
      float currAccum = mseAccum;
      float newAccum = (currAccum + sqErr);
      mseAccum = newAccum;
    }
    float finalSum = mseAccum;
    float nSamplesF32 = ((float)(nSamplesVal));
    float mse = (finalSum / nSamplesF32);
    fitness[gid] = mse;
  }
}

)kernel";
const char* k_metal_navatala_ml_mutate_population_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_mutate_population_f32(device const float* population [[buffer(0)]], device const ulong* rngState [[buffer(1)]], device const float* mutationRate [[buffer(2)]], device const float* mutationStrength [[buffer(3)]], device const uint* totalGenes [[buffer(4)]], device float* mutated [[buffer(5)]], device ulong* rngStateOut [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint totalGenesVal = totalGenes[0];
  float mutRateVal = mutationRate[0];
  float mutStrengthVal = mutationStrength[0];
  bool inBounds = (gid < totalGenesVal);
  if (inBounds) {
    float geneVal = population[gid];
    ulong s = rngState[gid];
    ulong aU64 = ((ulong)(1103515245u));
    ulong cU64 = ((ulong)(12345u));
    ulong s1 = (((aU64 * s) + cU64) & ((ulong)(2147483647u)));
    float u1 = (((float)(s1)) / as_type<float>(0x4f000000u));
    ulong s2 = (((aU64 * s1) + cU64) & ((ulong)(2147483647u)));
    float u2 = (((float)(s2)) / as_type<float>(0x4f000000u));
    ulong s3 = (((aU64 * s2) + cU64) & ((ulong)(2147483647u)));
    float u3 = (((float)(s3)) / as_type<float>(0x4f000000u));
    rngStateOut[gid] = s3;
    float u2Clamped = (((u2 < as_type<float>(0x33d6bf95u))) ? (as_type<float>(0x33d6bf95u)) : (u2));
    float logU2 = log(u2Clamped);
    float negTwoLogU2 = (as_type<float>(0xc0000000u) * logU2);
    float sqrtTerm = sqrt(negTwoLogU2);
    float angle = (as_type<float>(0x40c90fdbu) * u3);
    float cosTerm = cos(angle);
    float gaussian = (sqrtTerm * cosTerm);
    float perturbation = (gaussian * mutStrengthVal);
    bool shouldMutate = (u1 < mutRateVal);
    float mutatedVal = (geneVal + perturbation);
    float finalVal = ((shouldMutate) ? (mutatedVal) : (geneVal));
    mutated[gid] = finalVal;
  }
}

)kernel";
const char* k_metal_navatala_ml_crossover_population_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_crossover_population_f32(device const float* parent1 [[buffer(0)]], device const float* parent2 [[buffer(1)]], device const uint* crossoverPoints [[buffer(2)]], device const uint* nPairs [[buffer(3)]], device const uint* indivSize [[buffer(4)]], device float* offspring1 [[buffer(5)]], device float* offspring2 [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nPairsVal = nPairs[0];
  uint indivSizeVal = indivSize[0];
  uint totalGenes = (nPairsVal * indivSizeVal);
  bool inBounds = (gid < totalGenes);
  if (inBounds) {
    uint pairIdx = (gid / indivSizeVal);
    uint geneIdx = (gid % indivSizeVal);
    uint crossPt = crossoverPoints[pairIdx];
    float gene1 = parent1[gid];
    float gene2 = parent2[gid];
    bool beforeCrossover = (geneIdx < crossPt);
    float off1Gene = ((beforeCrossover) ? (gene1) : (gene2));
    float off2Gene = ((beforeCrossover) ? (gene2) : (gene1));
    offspring1[gid] = off1Gene;
    offspring2[gid] = off2Gene;
  }
}

)kernel";
const char* k_metal_navatala_ml_select_tournament_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_select_tournament_f32(device const float* fitness [[buffer(0)]], device const uint* tournamentIndices [[buffer(1)]], device const uint* nTournaments [[buffer(2)]], device const uint* tournamentSize [[buffer(3)]], device uint* selectedIndices [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nTournamentsVal = nTournaments[0];
  uint tournamentSizeVal = tournamentSize[0];
  bool inBounds = (gid < nTournamentsVal);
  if (inBounds) {
    uint baseIdx = (gid * tournamentSizeVal);
    uint firstParticipantIdx = tournamentIndices[baseIdx];
    float firstFitness = fitness[firstParticipantIdx];
    uint bestIdx = firstParticipantIdx;
    float bestFit = firstFitness;
    int tournamentSizeI32 = ((int)(tournamentSizeVal));
    for (int k = 0; k < (int)(tournamentSizeI32); ++k) {
      uint kU32 = ((uint)(k));
      bool kGt0 = (kU32 > 0u);
      if (kGt0) {
        uint participantIdxLoc = (baseIdx + kU32);
        uint participantIdx = tournamentIndices[participantIdxLoc];
        float participantFitness = fitness[participantIdx];
        float currBestFit = bestFit;
        bool isBetter = (participantFitness < currBestFit);
        if (isBetter) {
          bestIdx = participantIdx;
          bestFit = participantFitness;
        }
      }
    }
    uint winner = bestIdx;
    selectedIndices[gid] = winner;
  }
}

)kernel";
const char* k_metal_navatala_ml_batch_traverse_trees_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_batch_traverse_trees_f32(device const float* X [[buffer(0)]], device const int* feature_indices [[buffer(1)]], device const float* thresholds [[buffer(2)]], device const uint* left_children [[buffer(3)]], device const uint* right_children [[buffer(4)]], device const int* tree_offsets [[buffer(5)]], device const uint* n_samples [[buffer(6)]], device const uint* n_features [[buffer(7)]], device const uint* n_trees [[buffer(8)]], device int* leaf_indices [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nFeatures = n_features[0];
  uint nTrees = n_trees[0];
  uint totalWork = (nSamples * nTrees);
  bool inBounds = (gid < totalWork);
  if (inBounds) {
    uint sampleIdx = (gid / nTrees);
    uint treeIdx = (gid % nTrees);
    uint sampleBase = (sampleIdx * nFeatures);
    int treeOffset = tree_offsets[treeIdx];
    uint treeOffsetU32 = ((uint)(treeOffset));
    uint currentNode = 0u;
    for (int depth = 0; depth < (int)(64u); ++depth) {
      uint nodeIdx = currentNode;
      uint absNodeIdx = (treeOffsetU32 + nodeIdx);
      int featureIdx = feature_indices[absNodeIdx];
      float threshold = thresholds[absNodeIdx];
      bool isLeaf = (featureIdx == -1);
      if (isLeaf) {
        int leafIdxI32 = ((int)(nodeIdx));
        leaf_indices[gid] = leafIdxI32;
      } else {
        uint featureIdxU32 = ((uint)(featureIdx));
        uint featureAddr = (sampleBase + featureIdxU32);
        float featureVal = X[featureAddr];
        bool goLeft = (featureVal <= threshold);
        if (goLeft) {
          uint leftChild = left_children[absNodeIdx];
          currentNode = leftChild;
        } else {
          uint rightChild = right_children[absNodeIdx];
          currentNode = rightChild;
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_ml_collect_leaf_values_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_collect_leaf_values_f32(device const int* leaf_indices [[buffer(0)]], device const float* thresholds [[buffer(1)]], device const int* tree_offsets [[buffer(2)]], device const uint* n_samples [[buffer(3)]], device const uint* n_trees [[buffer(4)]], device float* leaf_values [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nTrees = n_trees[0];
  uint totalWork = (nSamples * nTrees);
  bool inBounds = (gid < totalWork);
  if (inBounds) {
    uint treeIdx = (gid % nTrees);
    int leafNodeIdx = leaf_indices[gid];
    uint leafNodeIdxU32 = ((uint)(leafNodeIdx));
    int treeOffset = tree_offsets[treeIdx];
    uint treeOffsetU32 = ((uint)(treeOffset));
    uint absLeafIdx = (treeOffsetU32 + leafNodeIdxU32);
    float leafValue = thresholds[absLeafIdx];
    leaf_values[gid] = leafValue;
  }
}

)kernel";
const char* k_metal_navatala_ml_aggregate_tree_outputs_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_aggregate_tree_outputs_f32(device const float* leaf_values [[buffer(0)]], device const uint* n_samples [[buffer(1)]], device const uint* n_trees [[buffer(2)]], device float* predictions [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nTrees = n_trees[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nTrees);
    float sumOutput = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(nTrees); ++t) {
      uint tU32 = ((uint)(t));
      uint leafValIdx = (sampleBase + tU32);
      float leafVal = leaf_values[leafValIdx];
      float currSum = sumOutput;
      float newSum = (currSum + leafVal);
      sumOutput = newSum;
    }
    float finalSum = sumOutput;
    float nTreesF = ((float)(nTrees));
    float avgPred = (finalSum / nTreesF);
    predictions[gid] = avgPred;
  }
}

)kernel";
const char* k_metal_navatala_ml_traverse_with_path_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_traverse_with_path_f32(device const float* X [[buffer(0)]], device const int* feature_indices [[buffer(1)]], device const float* thresholds [[buffer(2)]], device const uint* left_children [[buffer(3)]], device const uint* right_children [[buffer(4)]], device const int* tree_offsets [[buffer(5)]], device const uint* n_samples [[buffer(6)]], device const uint* n_features [[buffer(7)]], device const uint* n_trees [[buffer(8)]], device const uint* max_depth [[buffer(9)]], device int* paths [[buffer(10)]], device int* path_lengths [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nFeatures = n_features[0];
  uint nTrees = n_trees[0];
  uint maxDepth = max_depth[0];
  uint totalWork = (nSamples * nTrees);
  bool inBounds = (gid < totalWork);
  if (inBounds) {
    uint sampleIdx = (gid / nTrees);
    uint treeIdx = (gid % nTrees);
    uint sampleBase = (sampleIdx * nFeatures);
    uint pathBase = (gid * maxDepth);
    int treeOffset = tree_offsets[treeIdx];
    uint treeOffsetU32 = ((uint)(treeOffset));
    for (int d = 0; d < (int)(maxDepth); ++d) {
      uint dU32 = ((uint)(d));
      uint pathIdx = (pathBase + dU32);
      paths[pathIdx] = -1;
    }
    uint currentNode = 0u;
    uint currentDepth = 0u;
    for (int iter = 0; iter < (int)(64u); ++iter) {
      uint nodeIdx = currentNode;
      uint depth = currentDepth;
      bool depthOk = (depth < maxDepth);
      if (depthOk) {
        uint pathIdx = (pathBase + depth);
        int nodeIdxI32 = ((int)(nodeIdx));
        paths[pathIdx] = nodeIdxI32;
        uint absNodeIdx = (treeOffsetU32 + nodeIdx);
        int featureIdx = feature_indices[absNodeIdx];
        float threshold = thresholds[absNodeIdx];
        bool isLeaf = (featureIdx == -1);
        if (isLeaf) {
          int pathLen = ((int)((depth + 1u)));
          path_lengths[gid] = pathLen;
        } else {
          uint featureIdxU32 = ((uint)(featureIdx));
          uint featureAddr = (sampleBase + featureIdxU32);
          float featureVal = X[featureAddr];
          bool goLeft = (featureVal <= threshold);
          if (goLeft) {
            uint leftChild = left_children[absNodeIdx];
            currentNode = leftChild;
          } else {
            uint rightChild = right_children[absNodeIdx];
            currentNode = rightChild;
          }
          uint newDepth = (depth + 1u);
          currentDepth = newDepth;
        }
      }
    }
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_init_ema_f32[] = {
  { "alphaIn", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ema", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_init_ema_f32 = {
  1,
  "navatala_ml_init_ema_f32",
  "metal",
  "navatala_ml_init_ema_f32",
  "kernel:metal:navatala_ml_init_ema_f32",
  "abi-r1:metal:navatala_ml_init_ema_f32",
  "abi-r1:metal:navatala_ml_init_ema_f32",
  4,
  kAbiArgs_metal_navatala_ml_init_ema_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_init_ema_from_span_f32[] = {
  { "span", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ema", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_init_ema_from_span_f32 = {
  1,
  "navatala_ml_init_ema_from_span_f32",
  "metal",
  "navatala_ml_init_ema_from_span_f32",
  "kernel:metal:navatala_ml_init_ema_from_span_f32",
  "abi-r1:metal:navatala_ml_init_ema_from_span_f32",
  "abi-r1:metal:navatala_ml_init_ema_from_span_f32",
  4,
  kAbiArgs_metal_navatala_ml_init_ema_from_span_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_ema_update_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ema", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_ema_update_f32 = {
  1,
  "navatala_ml_ema_update_f32",
  "metal",
  "navatala_ml_ema_update_f32",
  "kernel:metal:navatala_ml_ema_update_f32",
  "abi-r1:metal:navatala_ml_ema_update_f32",
  "abi-r1:metal:navatala_ml_ema_update_f32",
  4,
  kAbiArgs_metal_navatala_ml_ema_update_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_ema_extract_f32[] = {
  { "ema", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_ema_extract_f32 = {
  1,
  "navatala_ml_ema_extract_f32",
  "metal",
  "navatala_ml_ema_extract_f32",
  "kernel:metal:navatala_ml_ema_extract_f32",
  "abi-r1:metal:navatala_ml_ema_extract_f32",
  "abi-r1:metal:navatala_ml_ema_extract_f32",
  2,
  kAbiArgs_metal_navatala_ml_ema_extract_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_ema_reset_f32[] = {
  { "ema", 0, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_ema_reset_f32 = {
  1,
  "navatala_ml_ema_reset_f32",
  "metal",
  "navatala_ml_ema_reset_f32",
  "kernel:metal:navatala_ml_ema_reset_f32",
  "abi-r1:metal:navatala_ml_ema_reset_f32",
  "abi-r1:metal:navatala_ml_ema_reset_f32",
  2,
  kAbiArgs_metal_navatala_ml_ema_reset_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_init_dema_f32[] = {
  { "alphaIn", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ema1", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ema2", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_init_dema_f32 = {
  1,
  "navatala_ml_init_dema_f32",
  "metal",
  "navatala_ml_init_dema_f32",
  "kernel:metal:navatala_ml_init_dema_f32",
  "abi-r1:metal:navatala_ml_init_dema_f32",
  "abi-r1:metal:navatala_ml_init_dema_f32",
  5,
  kAbiArgs_metal_navatala_ml_init_dema_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_dema_update_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ema1", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ema2", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_dema_update_f32 = {
  1,
  "navatala_ml_dema_update_f32",
  "metal",
  "navatala_ml_dema_update_f32",
  "kernel:metal:navatala_ml_dema_update_f32",
  "abi-r1:metal:navatala_ml_dema_update_f32",
  "abi-r1:metal:navatala_ml_dema_update_f32",
  5,
  kAbiArgs_metal_navatala_ml_dema_update_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_dema_extract_f32[] = {
  { "ema1", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ema2", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_dema_extract_f32 = {
  1,
  "navatala_ml_dema_extract_f32",
  "metal",
  "navatala_ml_dema_extract_f32",
  "kernel:metal:navatala_ml_dema_extract_f32",
  "abi-r1:metal:navatala_ml_dema_extract_f32",
  "abi-r1:metal:navatala_ml_dema_extract_f32",
  3,
  kAbiArgs_metal_navatala_ml_dema_extract_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_maxpool_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "window", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stride", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "inCount", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outCount", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_maxpool_f32 = {
  1,
  "navatala_ml_maxpool_f32",
  "metal",
  "navatala_ml_maxpool_f32",
  "kernel:metal:navatala_ml_maxpool_f32",
  "abi-r1:metal:navatala_ml_maxpool_f32",
  "abi-r1:metal:navatala_ml_maxpool_f32",
  6,
  kAbiArgs_metal_navatala_ml_maxpool_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_avgpool_exclude_pad_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "window", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stride", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "inCount", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outCount", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_avgpool_exclude_pad_f32 = {
  1,
  "navatala_ml_avgpool_exclude_pad_f32",
  "metal",
  "navatala_ml_avgpool_exclude_pad_f32",
  "kernel:metal:navatala_ml_avgpool_exclude_pad_f32",
  "abi-r1:metal:navatala_ml_avgpool_exclude_pad_f32",
  "abi-r1:metal:navatala_ml_avgpool_exclude_pad_f32",
  6,
  kAbiArgs_metal_navatala_ml_avgpool_exclude_pad_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_avgpool_include_pad_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "window", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stride", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "inCount", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outCount", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_avgpool_include_pad_f32 = {
  1,
  "navatala_ml_avgpool_include_pad_f32",
  "metal",
  "navatala_ml_avgpool_include_pad_f32",
  "kernel:metal:navatala_ml_avgpool_include_pad_f32",
  "abi-r1:metal:navatala_ml_avgpool_include_pad_f32",
  "abi-r1:metal:navatala_ml_avgpool_include_pad_f32",
  6,
  kAbiArgs_metal_navatala_ml_avgpool_include_pad_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_resize_nearest_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "inSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_resize_nearest_f32 = {
  1,
  "navatala_ml_resize_nearest_f32",
  "metal",
  "navatala_ml_resize_nearest_f32",
  "kernel:metal:navatala_ml_resize_nearest_f32",
  "abi-r1:metal:navatala_ml_resize_nearest_f32",
  "abi-r1:metal:navatala_ml_resize_nearest_f32",
  4,
  kAbiArgs_metal_navatala_ml_resize_nearest_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_resize_bilinear_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "inSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_resize_bilinear_f32 = {
  1,
  "navatala_ml_resize_bilinear_f32",
  "metal",
  "navatala_ml_resize_bilinear_f32",
  "kernel:metal:navatala_ml_resize_bilinear_f32",
  "abi-r1:metal:navatala_ml_resize_bilinear_f32",
  "abi-r1:metal:navatala_ml_resize_bilinear_f32",
  4,
  kAbiArgs_metal_navatala_ml_resize_bilinear_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_maxpool_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "window", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stride", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "inCount", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outCount", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_maxpool_f16 = {
  1,
  "navatala_ml_maxpool_f16",
  "metal",
  "navatala_ml_maxpool_f16",
  "kernel:metal:navatala_ml_maxpool_f16",
  "abi-r1:metal:navatala_ml_maxpool_f16",
  "abi-r1:metal:navatala_ml_maxpool_f16",
  6,
  kAbiArgs_metal_navatala_ml_maxpool_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_avgpool_exclude_pad_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "window", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stride", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "inCount", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outCount", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_avgpool_exclude_pad_f16 = {
  1,
  "navatala_ml_avgpool_exclude_pad_f16",
  "metal",
  "navatala_ml_avgpool_exclude_pad_f16",
  "kernel:metal:navatala_ml_avgpool_exclude_pad_f16",
  "abi-r1:metal:navatala_ml_avgpool_exclude_pad_f16",
  "abi-r1:metal:navatala_ml_avgpool_exclude_pad_f16",
  6,
  kAbiArgs_metal_navatala_ml_avgpool_exclude_pad_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_avgpool_include_pad_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "window", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stride", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "inCount", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outCount", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_avgpool_include_pad_f16 = {
  1,
  "navatala_ml_avgpool_include_pad_f16",
  "metal",
  "navatala_ml_avgpool_include_pad_f16",
  "kernel:metal:navatala_ml_avgpool_include_pad_f16",
  "abi-r1:metal:navatala_ml_avgpool_include_pad_f16",
  "abi-r1:metal:navatala_ml_avgpool_include_pad_f16",
  6,
  kAbiArgs_metal_navatala_ml_avgpool_include_pad_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_resize_nearest_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "inSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_resize_nearest_f16 = {
  1,
  "navatala_ml_resize_nearest_f16",
  "metal",
  "navatala_ml_resize_nearest_f16",
  "kernel:metal:navatala_ml_resize_nearest_f16",
  "abi-r1:metal:navatala_ml_resize_nearest_f16",
  "abi-r1:metal:navatala_ml_resize_nearest_f16",
  4,
  kAbiArgs_metal_navatala_ml_resize_nearest_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_resize_bilinear_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "inSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_resize_bilinear_f16 = {
  1,
  "navatala_ml_resize_bilinear_f16",
  "metal",
  "navatala_ml_resize_bilinear_f16",
  "kernel:metal:navatala_ml_resize_bilinear_f16",
  "abi-r1:metal:navatala_ml_resize_bilinear_f16",
  "abi-r1:metal:navatala_ml_resize_bilinear_f16",
  4,
  kAbiArgs_metal_navatala_ml_resize_bilinear_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reshape_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reshape_f32 = {
  1,
  "navatala_ml_reshape_f32",
  "metal",
  "navatala_ml_reshape_f32",
  "kernel:metal:navatala_ml_reshape_f32",
  "abi-r1:metal:navatala_ml_reshape_f32",
  "abi-r1:metal:navatala_ml_reshape_f32",
  3,
  kAbiArgs_metal_navatala_ml_reshape_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_slice_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outCount", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_slice_f32 = {
  1,
  "navatala_ml_slice_f32",
  "metal",
  "navatala_ml_slice_f32",
  "kernel:metal:navatala_ml_slice_f32",
  "abi-r1:metal:navatala_ml_slice_f32",
  "abi-r1:metal:navatala_ml_slice_f32",
  4,
  kAbiArgs_metal_navatala_ml_slice_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_transpose_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nrows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ncols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_transpose_f32 = {
  1,
  "navatala_ml_transpose_f32",
  "metal",
  "navatala_ml_transpose_f32",
  "kernel:metal:navatala_ml_transpose_f32",
  "abi-r1:metal:navatala_ml_transpose_f32",
  "abi-r1:metal:navatala_ml_transpose_f32",
  5,
  kAbiArgs_metal_navatala_ml_transpose_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_concatenate2_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_concatenate2_f32 = {
  1,
  "navatala_ml_concatenate2_f32",
  "metal",
  "navatala_ml_concatenate2_f32",
  "kernel:metal:navatala_ml_concatenate2_f32",
  "abi-r1:metal:navatala_ml_concatenate2_f32",
  "abi-r1:metal:navatala_ml_concatenate2_f32",
  6,
  kAbiArgs_metal_navatala_ml_concatenate2_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reshape_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reshape_f16 = {
  1,
  "navatala_ml_reshape_f16",
  "metal",
  "navatala_ml_reshape_f16",
  "kernel:metal:navatala_ml_reshape_f16",
  "abi-r1:metal:navatala_ml_reshape_f16",
  "abi-r1:metal:navatala_ml_reshape_f16",
  3,
  kAbiArgs_metal_navatala_ml_reshape_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_slice_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outCount", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_slice_f16 = {
  1,
  "navatala_ml_slice_f16",
  "metal",
  "navatala_ml_slice_f16",
  "kernel:metal:navatala_ml_slice_f16",
  "abi-r1:metal:navatala_ml_slice_f16",
  "abi-r1:metal:navatala_ml_slice_f16",
  4,
  kAbiArgs_metal_navatala_ml_slice_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_transpose_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nrows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ncols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_transpose_f16 = {
  1,
  "navatala_ml_transpose_f16",
  "metal",
  "navatala_ml_transpose_f16",
  "kernel:metal:navatala_ml_transpose_f16",
  "abi-r1:metal:navatala_ml_transpose_f16",
  "abi-r1:metal:navatala_ml_transpose_f16",
  5,
  kAbiArgs_metal_navatala_ml_transpose_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_concatenate2_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_concatenate2_f16 = {
  1,
  "navatala_ml_concatenate2_f16",
  "metal",
  "navatala_ml_concatenate2_f16",
  "kernel:metal:navatala_ml_concatenate2_f16",
  "abi-r1:metal:navatala_ml_concatenate2_f16",
  "abi-r1:metal:navatala_ml_concatenate2_f16",
  6,
  kAbiArgs_metal_navatala_ml_concatenate2_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_softmax_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_softmax_f32 = {
  1,
  "navatala_ml_softmax_f32",
  "metal",
  "navatala_ml_softmax_f32",
  "kernel:metal:navatala_ml_softmax_f32",
  "abi-r1:metal:navatala_ml_softmax_f32",
  "abi-r1:metal:navatala_ml_softmax_f32",
  3,
  kAbiArgs_metal_navatala_ml_softmax_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_softmax_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_softmax_f16 = {
  1,
  "navatala_ml_softmax_f16",
  "metal",
  "navatala_ml_softmax_f16",
  "kernel:metal:navatala_ml_softmax_f16",
  "abi-r1:metal:navatala_ml_softmax_f16",
  "abi-r1:metal:navatala_ml_softmax_f16",
  3,
  kAbiArgs_metal_navatala_ml_softmax_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_layernorm_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "eps", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_layernorm_f32 = {
  1,
  "navatala_ml_layernorm_f32",
  "metal",
  "navatala_ml_layernorm_f32",
  "kernel:metal:navatala_ml_layernorm_f32",
  "abi-r1:metal:navatala_ml_layernorm_f32",
  "abi-r1:metal:navatala_ml_layernorm_f32",
  6,
  kAbiArgs_metal_navatala_ml_layernorm_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_spatial_batchnorm_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "N", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "HW", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "eps", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_spatial_batchnorm_f32 = {
  1,
  "navatala_ml_spatial_batchnorm_f32",
  "metal",
  "navatala_ml_spatial_batchnorm_f32",
  "kernel:metal:navatala_ml_spatial_batchnorm_f32",
  "abi-r1:metal:navatala_ml_spatial_batchnorm_f32",
  "abi-r1:metal:navatala_ml_spatial_batchnorm_f32",
  8,
  kAbiArgs_metal_navatala_ml_spatial_batchnorm_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_instancenorm_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "HW", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "eps", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_instancenorm_f32 = {
  1,
  "navatala_ml_instancenorm_f32",
  "metal",
  "navatala_ml_instancenorm_f32",
  "kernel:metal:navatala_ml_instancenorm_f32",
  "abi-r1:metal:navatala_ml_instancenorm_f32",
  "abi-r1:metal:navatala_ml_instancenorm_f32",
  7,
  kAbiArgs_metal_navatala_ml_instancenorm_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_batchnorm_inference_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mean", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "var", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "eps", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_batchnorm_inference_f32 = {
  1,
  "navatala_ml_batchnorm_inference_f32",
  "metal",
  "navatala_ml_batchnorm_inference_f32",
  "kernel:metal:navatala_ml_batchnorm_inference_f32",
  "abi-r1:metal:navatala_ml_batchnorm_inference_f32",
  "abi-r1:metal:navatala_ml_batchnorm_inference_f32",
  8,
  kAbiArgs_metal_navatala_ml_batchnorm_inference_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_layernorm_backward_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dy", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "eps", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dx", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dgamma", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dbeta", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_layernorm_backward_f32 = {
  1,
  "navatala_ml_layernorm_backward_f32",
  "metal",
  "navatala_ml_layernorm_backward_f32",
  "kernel:metal:navatala_ml_layernorm_backward_f32",
  "abi-r1:metal:navatala_ml_layernorm_backward_f32",
  "abi-r1:metal:navatala_ml_layernorm_backward_f32",
  8,
  kAbiArgs_metal_navatala_ml_layernorm_backward_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_layernorm_f16[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "eps", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_layernorm_f16 = {
  1,
  "navatala_ml_layernorm_f16",
  "metal",
  "navatala_ml_layernorm_f16",
  "kernel:metal:navatala_ml_layernorm_f16",
  "abi-r1:metal:navatala_ml_layernorm_f16",
  "abi-r1:metal:navatala_ml_layernorm_f16",
  6,
  kAbiArgs_metal_navatala_ml_layernorm_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_spatial_batchnorm_f16[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "N", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "C", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "HW", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "eps", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_spatial_batchnorm_f16 = {
  1,
  "navatala_ml_spatial_batchnorm_f16",
  "metal",
  "navatala_ml_spatial_batchnorm_f16",
  "kernel:metal:navatala_ml_spatial_batchnorm_f16",
  "abi-r1:metal:navatala_ml_spatial_batchnorm_f16",
  "abi-r1:metal:navatala_ml_spatial_batchnorm_f16",
  8,
  kAbiArgs_metal_navatala_ml_spatial_batchnorm_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_instancenorm_f16[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "C", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "HW", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "eps", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_instancenorm_f16 = {
  1,
  "navatala_ml_instancenorm_f16",
  "metal",
  "navatala_ml_instancenorm_f16",
  "kernel:metal:navatala_ml_instancenorm_f16",
  "abi-r1:metal:navatala_ml_instancenorm_f16",
  "abi-r1:metal:navatala_ml_instancenorm_f16",
  7,
  kAbiArgs_metal_navatala_ml_instancenorm_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_batchnorm_inference_f16[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mean", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "var", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "eps", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_batchnorm_inference_f16 = {
  1,
  "navatala_ml_batchnorm_inference_f16",
  "metal",
  "navatala_ml_batchnorm_inference_f16",
  "kernel:metal:navatala_ml_batchnorm_inference_f16",
  "abi-r1:metal:navatala_ml_batchnorm_inference_f16",
  "abi-r1:metal:navatala_ml_batchnorm_inference_f16",
  8,
  kAbiArgs_metal_navatala_ml_batchnorm_inference_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_layernorm_backward_f16[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dy", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "eps", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dx", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dgamma", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dbeta", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_layernorm_backward_f16 = {
  1,
  "navatala_ml_layernorm_backward_f16",
  "metal",
  "navatala_ml_layernorm_backward_f16",
  "kernel:metal:navatala_ml_layernorm_backward_f16",
  "abi-r1:metal:navatala_ml_layernorm_backward_f16",
  "abi-r1:metal:navatala_ml_layernorm_backward_f16",
  8,
  kAbiArgs_metal_navatala_ml_layernorm_backward_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_uniform_f32[] = {
  { "count", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_uniform_f32 = {
  1,
  "navatala_ml_uniform_f32",
  "metal",
  "navatala_ml_uniform_f32",
  "kernel:metal:navatala_ml_uniform_f32",
  "abi-r1:metal:navatala_ml_uniform_f32",
  "abi-r1:metal:navatala_ml_uniform_f32",
  3,
  kAbiArgs_metal_navatala_ml_uniform_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_bernoulli_f32[] = {
  { "count", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "p", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_bernoulli_f32 = {
  1,
  "navatala_ml_bernoulli_f32",
  "metal",
  "navatala_ml_bernoulli_f32",
  "kernel:metal:navatala_ml_bernoulli_f32",
  "abi-r1:metal:navatala_ml_bernoulli_f32",
  "abi-r1:metal:navatala_ml_bernoulli_f32",
  4,
  kAbiArgs_metal_navatala_ml_bernoulli_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_normal_f32[] = {
  { "count", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_normal_f32 = {
  1,
  "navatala_ml_normal_f32",
  "metal",
  "navatala_ml_normal_f32",
  "kernel:metal:navatala_ml_normal_f32",
  "abi-r1:metal:navatala_ml_normal_f32",
  "abi-r1:metal:navatala_ml_normal_f32",
  3,
  kAbiArgs_metal_navatala_ml_normal_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_uniform_f16[] = {
  { "count", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_uniform_f16 = {
  1,
  "navatala_ml_uniform_f16",
  "metal",
  "navatala_ml_uniform_f16",
  "kernel:metal:navatala_ml_uniform_f16",
  "abi-r1:metal:navatala_ml_uniform_f16",
  "abi-r1:metal:navatala_ml_uniform_f16",
  3,
  kAbiArgs_metal_navatala_ml_uniform_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_bernoulli_f16[] = {
  { "count", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "p", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_bernoulli_f16 = {
  1,
  "navatala_ml_bernoulli_f16",
  "metal",
  "navatala_ml_bernoulli_f16",
  "kernel:metal:navatala_ml_bernoulli_f16",
  "abi-r1:metal:navatala_ml_bernoulli_f16",
  "abi-r1:metal:navatala_ml_bernoulli_f16",
  4,
  kAbiArgs_metal_navatala_ml_bernoulli_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_normal_f16[] = {
  { "count", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_normal_f16 = {
  1,
  "navatala_ml_normal_f16",
  "metal",
  "navatala_ml_normal_f16",
  "kernel:metal:navatala_ml_normal_f16",
  "abi-r1:metal:navatala_ml_normal_f16",
  "abi-r1:metal:navatala_ml_normal_f16",
  3,
  kAbiArgs_metal_navatala_ml_normal_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_kde_bandwidth_silverman_f32[] = {
  { "stddev", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "bandwidth", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_kde_bandwidth_silverman_f32 = {
  1,
  "navatala_ml_kde_bandwidth_silverman_f32",
  "metal",
  "navatala_ml_kde_bandwidth_silverman_f32",
  "kernel:metal:navatala_ml_kde_bandwidth_silverman_f32",
  "abi-r1:metal:navatala_ml_kde_bandwidth_silverman_f32",
  "abi-r1:metal:navatala_ml_kde_bandwidth_silverman_f32",
  3,
  kAbiArgs_metal_navatala_ml_kde_bandwidth_silverman_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_kde_bandwidth_scott_f32[] = {
  { "stddev", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "bandwidth", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_kde_bandwidth_scott_f32 = {
  1,
  "navatala_ml_kde_bandwidth_scott_f32",
  "metal",
  "navatala_ml_kde_bandwidth_scott_f32",
  "kernel:metal:navatala_ml_kde_bandwidth_scott_f32",
  "abi-r1:metal:navatala_ml_kde_bandwidth_scott_f32",
  "abi-r1:metal:navatala_ml_kde_bandwidth_scott_f32",
  3,
  kAbiArgs_metal_navatala_ml_kde_bandwidth_scott_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_kde_evaluate_gaussian_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "queryPoints", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bandwidth", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "densities", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_kde_evaluate_gaussian_f32 = {
  1,
  "navatala_ml_kde_evaluate_gaussian_f32",
  "metal",
  "navatala_ml_kde_evaluate_gaussian_f32",
  "kernel:metal:navatala_ml_kde_evaluate_gaussian_f32",
  "abi-r1:metal:navatala_ml_kde_evaluate_gaussian_f32",
  "abi-r1:metal:navatala_ml_kde_evaluate_gaussian_f32",
  6,
  kAbiArgs_metal_navatala_ml_kde_evaluate_gaussian_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_kde_evaluate_epanechnikov_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "queryPoints", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bandwidth", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "densities", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_kde_evaluate_epanechnikov_f32 = {
  1,
  "navatala_ml_kde_evaluate_epanechnikov_f32",
  "metal",
  "navatala_ml_kde_evaluate_epanechnikov_f32",
  "kernel:metal:navatala_ml_kde_evaluate_epanechnikov_f32",
  "abi-r1:metal:navatala_ml_kde_evaluate_epanechnikov_f32",
  "abi-r1:metal:navatala_ml_kde_evaluate_epanechnikov_f32",
  6,
  kAbiArgs_metal_navatala_ml_kde_evaluate_epanechnikov_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_kde_find_mode_f32[] = {
  { "densities", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "queryPoints", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "modeValue", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "modeLocation", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_kde_find_mode_f32 = {
  1,
  "navatala_ml_kde_find_mode_f32",
  "metal",
  "navatala_ml_kde_find_mode_f32",
  "kernel:metal:navatala_ml_kde_find_mode_f32",
  "abi-r1:metal:navatala_ml_kde_find_mode_f32",
  "abi-r1:metal:navatala_ml_kde_find_mode_f32",
  5,
  kAbiArgs_metal_navatala_ml_kde_find_mode_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pairwise_euclidean_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "D", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pairwise_euclidean_f32 = {
  1,
  "navatala_ml_pairwise_euclidean_f32",
  "metal",
  "navatala_ml_pairwise_euclidean_f32",
  "kernel:metal:navatala_ml_pairwise_euclidean_f32",
  "abi-r1:metal:navatala_ml_pairwise_euclidean_f32",
  "abi-r1:metal:navatala_ml_pairwise_euclidean_f32",
  6,
  kAbiArgs_metal_navatala_ml_pairwise_euclidean_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pairwise_squared_euclidean_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "D", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pairwise_squared_euclidean_f32 = {
  1,
  "navatala_ml_pairwise_squared_euclidean_f32",
  "metal",
  "navatala_ml_pairwise_squared_euclidean_f32",
  "kernel:metal:navatala_ml_pairwise_squared_euclidean_f32",
  "abi-r1:metal:navatala_ml_pairwise_squared_euclidean_f32",
  "abi-r1:metal:navatala_ml_pairwise_squared_euclidean_f32",
  6,
  kAbiArgs_metal_navatala_ml_pairwise_squared_euclidean_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pairwise_manhattan_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "D", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pairwise_manhattan_f32 = {
  1,
  "navatala_ml_pairwise_manhattan_f32",
  "metal",
  "navatala_ml_pairwise_manhattan_f32",
  "kernel:metal:navatala_ml_pairwise_manhattan_f32",
  "abi-r1:metal:navatala_ml_pairwise_manhattan_f32",
  "abi-r1:metal:navatala_ml_pairwise_manhattan_f32",
  6,
  kAbiArgs_metal_navatala_ml_pairwise_manhattan_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pairwise_cosine_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "D", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pairwise_cosine_f32 = {
  1,
  "navatala_ml_pairwise_cosine_f32",
  "metal",
  "navatala_ml_pairwise_cosine_f32",
  "kernel:metal:navatala_ml_pairwise_cosine_f32",
  "abi-r1:metal:navatala_ml_pairwise_cosine_f32",
  "abi-r1:metal:navatala_ml_pairwise_cosine_f32",
  6,
  kAbiArgs_metal_navatala_ml_pairwise_cosine_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_pairwise_chebyshev_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "D", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_pairwise_chebyshev_f32 = {
  1,
  "navatala_ml_pairwise_chebyshev_f32",
  "metal",
  "navatala_ml_pairwise_chebyshev_f32",
  "kernel:metal:navatala_ml_pairwise_chebyshev_f32",
  "abi-r1:metal:navatala_ml_pairwise_chebyshev_f32",
  "abi-r1:metal:navatala_ml_pairwise_chebyshev_f32",
  6,
  kAbiArgs_metal_navatala_ml_pairwise_chebyshev_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_nearest_neighbor_f32[] = {
  { "D", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_nearest_neighbor_f32 = {
  1,
  "navatala_ml_nearest_neighbor_f32",
  "metal",
  "navatala_ml_nearest_neighbor_f32",
  "kernel:metal:navatala_ml_nearest_neighbor_f32",
  "abi-r1:metal:navatala_ml_nearest_neighbor_f32",
  "abi-r1:metal:navatala_ml_nearest_neighbor_f32",
  5,
  kAbiArgs_metal_navatala_ml_nearest_neighbor_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_init_centroids_random_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "centroids", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_init_centroids_random_f32 = {
  1,
  "navatala_ml_init_centroids_random_f32",
  "metal",
  "navatala_ml_init_centroids_random_f32",
  "kernel:metal:navatala_ml_init_centroids_random_f32",
  "abi-r1:metal:navatala_ml_init_centroids_random_f32",
  "abi-r1:metal:navatala_ml_init_centroids_random_f32",
  6,
  kAbiArgs_metal_navatala_ml_init_centroids_random_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_assign_to_centroids_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "centroids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "assignments", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_assign_to_centroids_f32 = {
  1,
  "navatala_ml_assign_to_centroids_f32",
  "metal",
  "navatala_ml_assign_to_centroids_f32",
  "kernel:metal:navatala_ml_assign_to_centroids_f32",
  "abi-r1:metal:navatala_ml_assign_to_centroids_f32",
  "abi-r1:metal:navatala_ml_assign_to_centroids_f32",
  6,
  kAbiArgs_metal_navatala_ml_assign_to_centroids_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_cluster_sums_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "assignments", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "clusterSums", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusterCounts", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_cluster_sums_f32 = {
  1,
  "navatala_ml_compute_cluster_sums_f32",
  "metal",
  "navatala_ml_compute_cluster_sums_f32",
  "kernel:metal:navatala_ml_compute_cluster_sums_f32",
  "abi-r1:metal:navatala_ml_compute_cluster_sums_f32",
  "abi-r1:metal:navatala_ml_compute_cluster_sums_f32",
  7,
  kAbiArgs_metal_navatala_ml_compute_cluster_sums_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_update_centroids_f32[] = {
  { "clusterSums", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "clusterCounts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "oldCentroids", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "centroids", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_update_centroids_f32 = {
  1,
  "navatala_ml_update_centroids_f32",
  "metal",
  "navatala_ml_update_centroids_f32",
  "kernel:metal:navatala_ml_update_centroids_f32",
  "abi-r1:metal:navatala_ml_update_centroids_f32",
  "abi-r1:metal:navatala_ml_update_centroids_f32",
  6,
  kAbiArgs_metal_navatala_ml_update_centroids_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_inertia_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "centroids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "assignments", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "inertia", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_inertia_f32 = {
  1,
  "navatala_ml_compute_inertia_f32",
  "metal",
  "navatala_ml_compute_inertia_f32",
  "kernel:metal:navatala_ml_compute_inertia_f32",
  "abi-r1:metal:navatala_ml_compute_inertia_f32",
  "abi-r1:metal:navatala_ml_compute_inertia_f32",
  7,
  kAbiArgs_metal_navatala_ml_compute_inertia_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_init_gmm_state_f32[] = {
  { "initialMeans", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "means", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "variances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_init_gmm_state_f32 = {
  1,
  "navatala_ml_init_gmm_state_f32",
  "metal",
  "navatala_ml_init_gmm_state_f32",
  "kernel:metal:navatala_ml_init_gmm_state_f32",
  "abi-r1:metal:navatala_ml_init_gmm_state_f32",
  "abi-r1:metal:navatala_ml_init_gmm_state_f32",
  7,
  kAbiArgs_metal_navatala_ml_init_gmm_state_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_gaussian_log_pdf_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "means", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "variances", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "logPdf", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_gaussian_log_pdf_f32 = {
  1,
  "navatala_ml_gaussian_log_pdf_f32",
  "metal",
  "navatala_ml_gaussian_log_pdf_f32",
  "kernel:metal:navatala_ml_gaussian_log_pdf_f32",
  "abi-r1:metal:navatala_ml_gaussian_log_pdf_f32",
  "abi-r1:metal:navatala_ml_gaussian_log_pdf_f32",
  7,
  kAbiArgs_metal_navatala_ml_gaussian_log_pdf_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_responsibilities_f32[] = {
  { "logPdf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "responsibilities", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_responsibilities_f32 = {
  1,
  "navatala_ml_compute_responsibilities_f32",
  "metal",
  "navatala_ml_compute_responsibilities_f32",
  "kernel:metal:navatala_ml_compute_responsibilities_f32",
  "abi-r1:metal:navatala_ml_compute_responsibilities_f32",
  "abi-r1:metal:navatala_ml_compute_responsibilities_f32",
  5,
  kAbiArgs_metal_navatala_ml_compute_responsibilities_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_update_means_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "responsibilities", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "means", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_update_means_f32 = {
  1,
  "navatala_ml_update_means_f32",
  "metal",
  "navatala_ml_update_means_f32",
  "kernel:metal:navatala_ml_update_means_f32",
  "abi-r1:metal:navatala_ml_update_means_f32",
  "abi-r1:metal:navatala_ml_update_means_f32",
  6,
  kAbiArgs_metal_navatala_ml_update_means_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_update_variances_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "means", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "responsibilities", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "variances", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_update_variances_f32 = {
  1,
  "navatala_ml_update_variances_f32",
  "metal",
  "navatala_ml_update_variances_f32",
  "kernel:metal:navatala_ml_update_variances_f32",
  "abi-r1:metal:navatala_ml_update_variances_f32",
  "abi-r1:metal:navatala_ml_update_variances_f32",
  7,
  kAbiArgs_metal_navatala_ml_update_variances_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_update_weights_f32[] = {
  { "responsibilities", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "weights", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_update_weights_f32 = {
  1,
  "navatala_ml_update_weights_f32",
  "metal",
  "navatala_ml_update_weights_f32",
  "kernel:metal:navatala_ml_update_weights_f32",
  "abi-r1:metal:navatala_ml_update_weights_f32",
  "abi-r1:metal:navatala_ml_update_weights_f32",
  4,
  kAbiArgs_metal_navatala_ml_update_weights_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_log_likelihood_f32[] = {
  { "logPdf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "logLikelihood", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_log_likelihood_f32 = {
  1,
  "navatala_ml_compute_log_likelihood_f32",
  "metal",
  "navatala_ml_compute_log_likelihood_f32",
  "kernel:metal:navatala_ml_compute_log_likelihood_f32",
  "abi-r1:metal:navatala_ml_compute_log_likelihood_f32",
  "abi-r1:metal:navatala_ml_compute_log_likelihood_f32",
  5,
  kAbiArgs_metal_navatala_ml_compute_log_likelihood_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_gram_matrix_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nSamples", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nFeatures", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "G", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_gram_matrix_f32 = {
  1,
  "navatala_ml_compute_gram_matrix_f32",
  "metal",
  "navatala_ml_compute_gram_matrix_f32",
  "kernel:metal:navatala_ml_compute_gram_matrix_f32",
  "abi-r1:metal:navatala_ml_compute_gram_matrix_f32",
  "abi-r1:metal:navatala_ml_compute_gram_matrix_f32",
  4,
  kAbiArgs_metal_navatala_ml_compute_gram_matrix_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_xt_y_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nSamples", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nFeatures", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "Xty", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_xt_y_f32 = {
  1,
  "navatala_ml_compute_xt_y_f32",
  "metal",
  "navatala_ml_compute_xt_y_f32",
  "kernel:metal:navatala_ml_compute_xt_y_f32",
  "abi-r1:metal:navatala_ml_compute_xt_y_f32",
  "abi-r1:metal:navatala_ml_compute_xt_y_f32",
  5,
  kAbiArgs_metal_navatala_ml_compute_xt_y_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_predict_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nSamples", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nFeatures", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "yPred", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_predict_f32 = {
  1,
  "navatala_ml_predict_f32",
  "metal",
  "navatala_ml_predict_f32",
  "kernel:metal:navatala_ml_predict_f32",
  "abi-r1:metal:navatala_ml_predict_f32",
  "abi-r1:metal:navatala_ml_predict_f32",
  5,
  kAbiArgs_metal_navatala_ml_predict_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_residuals_f32[] = {
  { "y", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "yPred", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nSamples", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "residuals", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_residuals_f32 = {
  1,
  "navatala_ml_compute_residuals_f32",
  "metal",
  "navatala_ml_compute_residuals_f32",
  "kernel:metal:navatala_ml_compute_residuals_f32",
  "abi-r1:metal:navatala_ml_compute_residuals_f32",
  "abi-r1:metal:navatala_ml_compute_residuals_f32",
  4,
  kAbiArgs_metal_navatala_ml_compute_residuals_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_r_squared_f32[] = {
  { "y", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "yPred", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "yMean", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nSamples", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rSquared", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_r_squared_f32 = {
  1,
  "navatala_ml_compute_r_squared_f32",
  "metal",
  "navatala_ml_compute_r_squared_f32",
  "kernel:metal:navatala_ml_compute_r_squared_f32",
  "abi-r1:metal:navatala_ml_compute_r_squared_f32",
  "abi-r1:metal:navatala_ml_compute_r_squared_f32",
  5,
  kAbiArgs_metal_navatala_ml_compute_r_squared_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_mean_f32[] = {
  { "y", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nSamples", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "mean", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_mean_f32 = {
  1,
  "navatala_ml_compute_mean_f32",
  "metal",
  "navatala_ml_compute_mean_f32",
  "kernel:metal:navatala_ml_compute_mean_f32",
  "abi-r1:metal:navatala_ml_compute_mean_f32",
  "abi-r1:metal:navatala_ml_compute_mean_f32",
  3,
  kAbiArgs_metal_navatala_ml_compute_mean_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_add_regularization_f32[] = {
  { "G", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambda", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "p", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "G", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_add_regularization_f32 = {
  1,
  "navatala_ml_add_regularization_f32",
  "metal",
  "navatala_ml_add_regularization_f32",
  "kernel:metal:navatala_ml_add_regularization_f32",
  "abi-r1:metal:navatala_ml_add_regularization_f32",
  "abi-r1:metal:navatala_ml_add_regularization_f32",
  4,
  kAbiArgs_metal_navatala_ml_add_regularization_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_ridge_predict_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "p", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "yPred", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_ridge_predict_f32 = {
  1,
  "navatala_ml_ridge_predict_f32",
  "metal",
  "navatala_ml_ridge_predict_f32",
  "kernel:metal:navatala_ml_ridge_predict_f32",
  "abi-r1:metal:navatala_ml_ridge_predict_f32",
  "abi-r1:metal:navatala_ml_ridge_predict_f32",
  5,
  kAbiArgs_metal_navatala_ml_ridge_predict_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_gcv_score_f32[] = {
  { "y", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "yPred", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "traceH", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gcv", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_gcv_score_f32 = {
  1,
  "navatala_ml_compute_gcv_score_f32",
  "metal",
  "navatala_ml_compute_gcv_score_f32",
  "kernel:metal:navatala_ml_compute_gcv_score_f32",
  "abi-r1:metal:navatala_ml_compute_gcv_score_f32",
  "abi-r1:metal:navatala_ml_compute_gcv_score_f32",
  5,
  kAbiArgs_metal_navatala_ml_compute_gcv_score_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_effective_dof_f32[] = {
  { "eigenvalues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambda", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "p", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dof", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_effective_dof_f32 = {
  1,
  "navatala_ml_compute_effective_dof_f32",
  "metal",
  "navatala_ml_compute_effective_dof_f32",
  "kernel:metal:navatala_ml_compute_effective_dof_f32",
  "abi-r1:metal:navatala_ml_compute_effective_dof_f32",
  "abi-r1:metal:navatala_ml_compute_effective_dof_f32",
  4,
  kAbiArgs_metal_navatala_ml_compute_effective_dof_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_init_lasso_state_f32[] = {
  { "y", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nSamples", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nFeatures", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "residuals", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_init_lasso_state_f32 = {
  1,
  "navatala_ml_init_lasso_state_f32",
  "metal",
  "navatala_ml_init_lasso_state_f32",
  "kernel:metal:navatala_ml_init_lasso_state_f32",
  "abi-r1:metal:navatala_ml_init_lasso_state_f32",
  "abi-r1:metal:navatala_ml_init_lasso_state_f32",
  5,
  kAbiArgs_metal_navatala_ml_init_lasso_state_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_partial_residual_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "j", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nSamples", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nFeatures", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "residuals", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_partial_residual_f32 = {
  1,
  "navatala_ml_compute_partial_residual_f32",
  "metal",
  "navatala_ml_compute_partial_residual_f32",
  "kernel:metal:navatala_ml_compute_partial_residual_f32",
  "abi-r1:metal:navatala_ml_compute_partial_residual_f32",
  "abi-r1:metal:navatala_ml_compute_partial_residual_f32",
  7,
  kAbiArgs_metal_navatala_ml_compute_partial_residual_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_soft_threshold_f32[] = {
  { "value", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "threshold", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_soft_threshold_f32 = {
  1,
  "navatala_ml_soft_threshold_f32",
  "metal",
  "navatala_ml_soft_threshold_f32",
  "kernel:metal:navatala_ml_soft_threshold_f32",
  "abi-r1:metal:navatala_ml_soft_threshold_f32",
  "abi-r1:metal:navatala_ml_soft_threshold_f32",
  3,
  kAbiArgs_metal_navatala_ml_soft_threshold_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_coordinate_update_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "residuals", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "j", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "lambda", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nSamples", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nFeatures", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "newBetaJ", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_coordinate_update_f32 = {
  1,
  "navatala_ml_coordinate_update_f32",
  "metal",
  "navatala_ml_coordinate_update_f32",
  "kernel:metal:navatala_ml_coordinate_update_f32",
  "abi-r1:metal:navatala_ml_coordinate_update_f32",
  "abi-r1:metal:navatala_ml_coordinate_update_f32",
  7,
  kAbiArgs_metal_navatala_ml_coordinate_update_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_objective_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambda", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nSamples", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nFeatures", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "objective", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_objective_f32 = {
  1,
  "navatala_ml_compute_objective_f32",
  "metal",
  "navatala_ml_compute_objective_f32",
  "kernel:metal:navatala_ml_compute_objective_f32",
  "abi-r1:metal:navatala_ml_compute_objective_f32",
  "abi-r1:metal:navatala_ml_compute_objective_f32",
  7,
  kAbiArgs_metal_navatala_ml_compute_objective_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_active_set_f32[] = {
  { "beta", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nFeatures", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "activeMask", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_active_set_f32 = {
  1,
  "navatala_ml_compute_active_set_f32",
  "metal",
  "navatala_ml_compute_active_set_f32",
  "kernel:metal:navatala_ml_compute_active_set_f32",
  "abi-r1:metal:navatala_ml_compute_active_set_f32",
  "abi-r1:metal:navatala_ml_compute_active_set_f32",
  3,
  kAbiArgs_metal_navatala_ml_compute_active_set_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_center_data_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nSamples", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nFeatures", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "means", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_center_data_f32 = {
  1,
  "navatala_ml_center_data_f32",
  "metal",
  "navatala_ml_center_data_f32",
  "kernel:metal:navatala_ml_center_data_f32",
  "abi-r1:metal:navatala_ml_center_data_f32",
  "abi-r1:metal:navatala_ml_center_data_f32",
  5,
  kAbiArgs_metal_navatala_ml_center_data_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_covariance_matrix_f32[] = {
  { "Xc", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nSamples", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nFeatures", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "Cov", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_covariance_matrix_f32 = {
  1,
  "navatala_ml_compute_covariance_matrix_f32",
  "metal",
  "navatala_ml_compute_covariance_matrix_f32",
  "kernel:metal:navatala_ml_compute_covariance_matrix_f32",
  "abi-r1:metal:navatala_ml_compute_covariance_matrix_f32",
  "abi-r1:metal:navatala_ml_compute_covariance_matrix_f32",
  4,
  kAbiArgs_metal_navatala_ml_compute_covariance_matrix_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_project_to_principal_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "V", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nSamples", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nFeatures", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nComponents", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "Z", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_project_to_principal_f32 = {
  1,
  "navatala_ml_project_to_principal_f32",
  "metal",
  "navatala_ml_project_to_principal_f32",
  "kernel:metal:navatala_ml_project_to_principal_f32",
  "abi-r1:metal:navatala_ml_project_to_principal_f32",
  "abi-r1:metal:navatala_ml_project_to_principal_f32",
  6,
  kAbiArgs_metal_navatala_ml_project_to_principal_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_explained_variance_ratio_f32[] = {
  { "eigenvalues", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nFeatures", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nComponents", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ratios", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_explained_variance_ratio_f32 = {
  1,
  "navatala_ml_explained_variance_ratio_f32",
  "metal",
  "navatala_ml_explained_variance_ratio_f32",
  "kernel:metal:navatala_ml_explained_variance_ratio_f32",
  "abi-r1:metal:navatala_ml_explained_variance_ratio_f32",
  "abi-r1:metal:navatala_ml_explained_variance_ratio_f32",
  4,
  kAbiArgs_metal_navatala_ml_explained_variance_ratio_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_pairwise_affinity_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sigma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nPoints", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nDims", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "P", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_pairwise_affinity_f32 = {
  1,
  "navatala_ml_compute_pairwise_affinity_f32",
  "metal",
  "navatala_ml_compute_pairwise_affinity_f32",
  "kernel:metal:navatala_ml_compute_pairwise_affinity_f32",
  "abi-r1:metal:navatala_ml_compute_pairwise_affinity_f32",
  "abi-r1:metal:navatala_ml_compute_pairwise_affinity_f32",
  5,
  kAbiArgs_metal_navatala_ml_compute_pairwise_affinity_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_symmetrize_affinity_f32[] = {
  { "P", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nPoints", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_symmetrize_affinity_f32 = {
  1,
  "navatala_ml_symmetrize_affinity_f32",
  "metal",
  "navatala_ml_symmetrize_affinity_f32",
  "kernel:metal:navatala_ml_symmetrize_affinity_f32",
  "abi-r1:metal:navatala_ml_symmetrize_affinity_f32",
  "abi-r1:metal:navatala_ml_symmetrize_affinity_f32",
  2,
  kAbiArgs_metal_navatala_ml_symmetrize_affinity_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_init_embedding_random_f32[] = {
  { "nPoints", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outputDims", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "Y", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_init_embedding_random_f32 = {
  1,
  "navatala_ml_init_embedding_random_f32",
  "metal",
  "navatala_ml_init_embedding_random_f32",
  "kernel:metal:navatala_ml_init_embedding_random_f32",
  "abi-r1:metal:navatala_ml_init_embedding_random_f32",
  "abi-r1:metal:navatala_ml_init_embedding_random_f32",
  4,
  kAbiArgs_metal_navatala_ml_init_embedding_random_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_q_distribution_f32[] = {
  { "Y", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nPoints", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outputDims", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "Q", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sumQ", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_q_distribution_f32 = {
  1,
  "navatala_ml_compute_q_distribution_f32",
  "metal",
  "navatala_ml_compute_q_distribution_f32",
  "kernel:metal:navatala_ml_compute_q_distribution_f32",
  "abi-r1:metal:navatala_ml_compute_q_distribution_f32",
  "abi-r1:metal:navatala_ml_compute_q_distribution_f32",
  5,
  kAbiArgs_metal_navatala_ml_compute_q_distribution_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_gradient_f32[] = {
  { "P", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Q", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Y", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nPoints", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outputDims", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "grad", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_gradient_f32 = {
  1,
  "navatala_ml_compute_gradient_f32",
  "metal",
  "navatala_ml_compute_gradient_f32",
  "kernel:metal:navatala_ml_compute_gradient_f32",
  "abi-r1:metal:navatala_ml_compute_gradient_f32",
  "abi-r1:metal:navatala_ml_compute_gradient_f32",
  6,
  kAbiArgs_metal_navatala_ml_compute_gradient_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_apply_momentum_update_f32[] = {
  { "Y", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Y_prev", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "grad", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lr", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "momentum", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nPoints", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outputDims", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "Y_new", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_apply_momentum_update_f32 = {
  1,
  "navatala_ml_apply_momentum_update_f32",
  "metal",
  "navatala_ml_apply_momentum_update_f32",
  "kernel:metal:navatala_ml_apply_momentum_update_f32",
  "abi-r1:metal:navatala_ml_apply_momentum_update_f32",
  "abi-r1:metal:navatala_ml_apply_momentum_update_f32",
  8,
  kAbiArgs_metal_navatala_ml_apply_momentum_update_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_k_l_divergence_f32[] = {
  { "P", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Q", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nPoints", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kl_divergence", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_k_l_divergence_f32 = {
  1,
  "navatala_ml_compute_k_l_divergence_f32",
  "metal",
  "navatala_ml_compute_k_l_divergence_f32",
  "kernel:metal:navatala_ml_compute_k_l_divergence_f32",
  "abi-r1:metal:navatala_ml_compute_k_l_divergence_f32",
  "abi-r1:metal:navatala_ml_compute_k_l_divergence_f32",
  4,
  kAbiArgs_metal_navatala_ml_compute_k_l_divergence_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_perplexity_f32[] = {
  { "distances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "targetPerplexity", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nPoints", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxIter", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sigma", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_perplexity_f32 = {
  1,
  "navatala_ml_compute_perplexity_f32",
  "metal",
  "navatala_ml_compute_perplexity_f32",
  "kernel:metal:navatala_ml_compute_perplexity_f32",
  "abi-r1:metal:navatala_ml_compute_perplexity_f32",
  "abi-r1:metal:navatala_ml_compute_perplexity_f32",
  5,
  kAbiArgs_metal_navatala_ml_compute_perplexity_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_normalize_embedding_f32[] = {
  { "Y", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nPoints", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outputDims", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "means", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_normalize_embedding_f32 = {
  1,
  "navatala_ml_normalize_embedding_f32",
  "metal",
  "navatala_ml_normalize_embedding_f32",
  "kernel:metal:navatala_ml_normalize_embedding_f32",
  "abi-r1:metal:navatala_ml_normalize_embedding_f32",
  "abi-r1:metal:navatala_ml_normalize_embedding_f32",
  4,
  kAbiArgs_metal_navatala_ml_normalize_embedding_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_exaggerated_p_f32[] = {
  { "P", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nPoints", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_exaggerated_p_f32 = {
  1,
  "navatala_ml_compute_exaggerated_p_f32",
  "metal",
  "navatala_ml_compute_exaggerated_p_f32",
  "kernel:metal:navatala_ml_compute_exaggerated_p_f32",
  "abi-r1:metal:navatala_ml_compute_exaggerated_p_f32",
  "abi-r1:metal:navatala_ml_compute_exaggerated_p_f32",
  3,
  kAbiArgs_metal_navatala_ml_compute_exaggerated_p_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_neighbor_counts_f32[] = {
  { "points", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nPoints", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nDims", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_neighbor_counts_f32 = {
  1,
  "navatala_ml_compute_neighbor_counts_f32",
  "metal",
  "navatala_ml_compute_neighbor_counts_f32",
  "kernel:metal:navatala_ml_compute_neighbor_counts_f32",
  "abi-r1:metal:navatala_ml_compute_neighbor_counts_f32",
  "abi-r1:metal:navatala_ml_compute_neighbor_counts_f32",
  5,
  kAbiArgs_metal_navatala_ml_compute_neighbor_counts_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_identify_core_points_f32[] = {
  { "counts", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "minPts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nPoints", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "isCore", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_identify_core_points_f32 = {
  1,
  "navatala_ml_identify_core_points_f32",
  "metal",
  "navatala_ml_identify_core_points_f32",
  "kernel:metal:navatala_ml_identify_core_points_f32",
  "abi-r1:metal:navatala_ml_identify_core_points_f32",
  "abi-r1:metal:navatala_ml_identify_core_points_f32",
  4,
  kAbiArgs_metal_navatala_ml_identify_core_points_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_identify_core_points_f64[] = {
  { "counts", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "minPts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nPoints", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "isCore", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_identify_core_points_f64 = {
  1,
  "navatala_ml_identify_core_points_f64",
  "metal",
  "navatala_ml_identify_core_points_f64",
  "kernel:metal:navatala_ml_identify_core_points_f64",
  "abi-r1:metal:navatala_ml_identify_core_points_f64",
  "abi-r1:metal:navatala_ml_identify_core_points_f64",
  4,
  kAbiArgs_metal_navatala_ml_identify_core_points_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_init_cluster_labels_f32[] = {
  { "nPoints", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labels", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_init_cluster_labels_f32 = {
  1,
  "navatala_ml_init_cluster_labels_f32",
  "metal",
  "navatala_ml_init_cluster_labels_f32",
  "kernel:metal:navatala_ml_init_cluster_labels_f32",
  "abi-r1:metal:navatala_ml_init_cluster_labels_f32",
  "abi-r1:metal:navatala_ml_init_cluster_labels_f32",
  2,
  kAbiArgs_metal_navatala_ml_init_cluster_labels_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_init_cluster_labels_f64[] = {
  { "nPoints", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labels", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_init_cluster_labels_f64 = {
  1,
  "navatala_ml_init_cluster_labels_f64",
  "metal",
  "navatala_ml_init_cluster_labels_f64",
  "kernel:metal:navatala_ml_init_cluster_labels_f64",
  "abi-r1:metal:navatala_ml_init_cluster_labels_f64",
  "abi-r1:metal:navatala_ml_init_cluster_labels_f64",
  2,
  kAbiArgs_metal_navatala_ml_init_cluster_labels_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_expand_cluster_step_f32[] = {
  { "points", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "labelsIn", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "isCore", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nPoints", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nDims", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labelsOut", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_expand_cluster_step_f32 = {
  1,
  "navatala_ml_expand_cluster_step_f32",
  "metal",
  "navatala_ml_expand_cluster_step_f32",
  "kernel:metal:navatala_ml_expand_cluster_step_f32",
  "abi-r1:metal:navatala_ml_expand_cluster_step_f32",
  "abi-r1:metal:navatala_ml_expand_cluster_step_f32",
  7,
  kAbiArgs_metal_navatala_ml_expand_cluster_step_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_propagate_core_labels_f32[] = {
  { "points", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "labelsIn", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "isCore", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nPoints", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nDims", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labelsOut", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_propagate_core_labels_f32 = {
  1,
  "navatala_ml_propagate_core_labels_f32",
  "metal",
  "navatala_ml_propagate_core_labels_f32",
  "kernel:metal:navatala_ml_propagate_core_labels_f32",
  "abi-r1:metal:navatala_ml_propagate_core_labels_f32",
  "abi-r1:metal:navatala_ml_propagate_core_labels_f32",
  7,
  kAbiArgs_metal_navatala_ml_propagate_core_labels_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_mark_noise_points_f32[] = {
  { "labelsIn", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "isCore", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nPoints", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labelsOut", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_mark_noise_points_f32 = {
  1,
  "navatala_ml_mark_noise_points_f32",
  "metal",
  "navatala_ml_mark_noise_points_f32",
  "kernel:metal:navatala_ml_mark_noise_points_f32",
  "abi-r1:metal:navatala_ml_mark_noise_points_f32",
  "abi-r1:metal:navatala_ml_mark_noise_points_f32",
  4,
  kAbiArgs_metal_navatala_ml_mark_noise_points_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_mark_noise_points_f64[] = {
  { "labelsIn", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "isCore", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nPoints", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labelsOut", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_mark_noise_points_f64 = {
  1,
  "navatala_ml_mark_noise_points_f64",
  "metal",
  "navatala_ml_mark_noise_points_f64",
  "kernel:metal:navatala_ml_mark_noise_points_f64",
  "abi-r1:metal:navatala_ml_mark_noise_points_f64",
  "abi-r1:metal:navatala_ml_mark_noise_points_f64",
  4,
  kAbiArgs_metal_navatala_ml_mark_noise_points_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_count_clusters_f32[] = {
  { "labels", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nPoints", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClusters", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_count_clusters_f32 = {
  1,
  "navatala_ml_count_clusters_f32",
  "metal",
  "navatala_ml_count_clusters_f32",
  "kernel:metal:navatala_ml_count_clusters_f32",
  "abi-r1:metal:navatala_ml_count_clusters_f32",
  "abi-r1:metal:navatala_ml_count_clusters_f32",
  3,
  kAbiArgs_metal_navatala_ml_count_clusters_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_count_clusters_f64[] = {
  { "labels", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nPoints", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClusters", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_count_clusters_f64 = {
  1,
  "navatala_ml_count_clusters_f64",
  "metal",
  "navatala_ml_count_clusters_f64",
  "kernel:metal:navatala_ml_count_clusters_f64",
  "abi-r1:metal:navatala_ml_count_clusters_f64",
  "abi-r1:metal:navatala_ml_count_clusters_f64",
  3,
  kAbiArgs_metal_navatala_ml_count_clusters_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_neighborhood_recall_f32[] = {
  { "origNeighbors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "embedNeighbors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "recall", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_neighborhood_recall_f32 = {
  1,
  "navatala_ml_neighborhood_recall_f32",
  "metal",
  "navatala_ml_neighborhood_recall_f32",
  "kernel:metal:navatala_ml_neighborhood_recall_f32",
  "abi-r1:metal:navatala_ml_neighborhood_recall_f32",
  "abi-r1:metal:navatala_ml_neighborhood_recall_f32",
  5,
  kAbiArgs_metal_navatala_ml_neighborhood_recall_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_trustworthiness_f32[] = {
  { "origNeighbors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "embedNeighbors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "origRanks", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "penalty", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_trustworthiness_f32 = {
  1,
  "navatala_ml_trustworthiness_f32",
  "metal",
  "navatala_ml_trustworthiness_f32",
  "kernel:metal:navatala_ml_trustworthiness_f32",
  "abi-r1:metal:navatala_ml_trustworthiness_f32",
  "abi-r1:metal:navatala_ml_trustworthiness_f32",
  6,
  kAbiArgs_metal_navatala_ml_trustworthiness_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_continuity_f32[] = {
  { "origNeighbors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "embedNeighbors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "embedRanks", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "penalty", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_continuity_f32 = {
  1,
  "navatala_ml_continuity_f32",
  "metal",
  "navatala_ml_continuity_f32",
  "kernel:metal:navatala_ml_continuity_f32",
  "abi-r1:metal:navatala_ml_continuity_f32",
  "abi-r1:metal:navatala_ml_continuity_f32",
  6,
  kAbiArgs_metal_navatala_ml_continuity_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_make_blobs_f32[] = {
  { "centers", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "labels", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "randomNormal", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "stddev", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "data", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_make_blobs_f32 = {
  1,
  "navatala_ml_make_blobs_f32",
  "metal",
  "navatala_ml_make_blobs_f32",
  "kernel:metal:navatala_ml_make_blobs_f32",
  "abi-r1:metal:navatala_ml_make_blobs_f32",
  "abi-r1:metal:navatala_ml_make_blobs_f32",
  7,
  kAbiArgs_metal_navatala_ml_make_blobs_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_make_blobs_anisotropic_f32[] = {
  { "centers", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scales", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "labels", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "randomNormal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "data", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_make_blobs_anisotropic_f32 = {
  1,
  "navatala_ml_make_blobs_anisotropic_f32",
  "metal",
  "navatala_ml_make_blobs_anisotropic_f32",
  "kernel:metal:navatala_ml_make_blobs_anisotropic_f32",
  "abi-r1:metal:navatala_ml_make_blobs_anisotropic_f32",
  "abi-r1:metal:navatala_ml_make_blobs_anisotropic_f32",
  7,
  kAbiArgs_metal_navatala_ml_make_blobs_anisotropic_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_make_moons_f32[] = {
  { "labels", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "randomUniform", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "randomNormal", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "noise", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "data", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_make_moons_f32 = {
  1,
  "navatala_ml_make_moons_f32",
  "metal",
  "navatala_ml_make_moons_f32",
  "kernel:metal:navatala_ml_make_moons_f32",
  "abi-r1:metal:navatala_ml_make_moons_f32",
  "abi-r1:metal:navatala_ml_make_moons_f32",
  6,
  kAbiArgs_metal_navatala_ml_make_moons_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_make_circles_f32[] = {
  { "labels", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "randomUniform", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "randomNormal", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "noise", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "factor", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "data", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_make_circles_f32 = {
  1,
  "navatala_ml_make_circles_f32",
  "metal",
  "navatala_ml_make_circles_f32",
  "kernel:metal:navatala_ml_make_circles_f32",
  "abi-r1:metal:navatala_ml_make_circles_f32",
  "abi-r1:metal:navatala_ml_make_circles_f32",
  7,
  kAbiArgs_metal_navatala_ml_make_circles_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_make_regression_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "w", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bias", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "noise", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_make_regression_f32 = {
  1,
  "navatala_ml_make_regression_f32",
  "metal",
  "navatala_ml_make_regression_f32",
  "kernel:metal:navatala_ml_make_regression_f32",
  "abi-r1:metal:navatala_ml_make_regression_f32",
  "abi-r1:metal:navatala_ml_make_regression_f32",
  7,
  kAbiArgs_metal_navatala_ml_make_regression_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_make_classification_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "w", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bias", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labels", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_make_classification_f32 = {
  1,
  "navatala_ml_make_classification_f32",
  "metal",
  "navatala_ml_make_classification_f32",
  "kernel:metal:navatala_ml_make_classification_f32",
  "abi-r1:metal:navatala_ml_make_classification_f32",
  "abi-r1:metal:navatala_ml_make_classification_f32",
  6,
  kAbiArgs_metal_navatala_ml_make_classification_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_make_sparse_uncorrelated_f32[] = {
  { "randomNormal", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sparseMask", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "X", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_make_sparse_uncorrelated_f32 = {
  1,
  "navatala_ml_make_sparse_uncorrelated_f32",
  "metal",
  "navatala_ml_make_sparse_uncorrelated_f32",
  "kernel:metal:navatala_ml_make_sparse_uncorrelated_f32",
  "abi-r1:metal:navatala_ml_make_sparse_uncorrelated_f32",
  "abi-r1:metal:navatala_ml_make_sparse_uncorrelated_f32",
  6,
  kAbiArgs_metal_navatala_ml_make_sparse_uncorrelated_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_knn_brute_force_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "data", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "q", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_knn_brute_force_f32 = {
  1,
  "navatala_ml_knn_brute_force_f32",
  "metal",
  "navatala_ml_knn_brute_force_f32",
  "kernel:metal:navatala_ml_knn_brute_force_f32",
  "abi-r1:metal:navatala_ml_knn_brute_force_f32",
  "abi-r1:metal:navatala_ml_knn_brute_force_f32",
  8,
  kAbiArgs_metal_navatala_ml_knn_brute_force_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_knn_brute_force_euclidean_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "data", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "q", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_knn_brute_force_euclidean_f32 = {
  1,
  "navatala_ml_knn_brute_force_euclidean_f32",
  "metal",
  "navatala_ml_knn_brute_force_euclidean_f32",
  "kernel:metal:navatala_ml_knn_brute_force_euclidean_f32",
  "abi-r1:metal:navatala_ml_knn_brute_force_euclidean_f32",
  "abi-r1:metal:navatala_ml_knn_brute_force_euclidean_f32",
  8,
  kAbiArgs_metal_navatala_ml_knn_brute_force_euclidean_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_knn_brute_force_cosine_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "data", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "q", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_knn_brute_force_cosine_f32 = {
  1,
  "navatala_ml_knn_brute_force_cosine_f32",
  "metal",
  "navatala_ml_knn_brute_force_cosine_f32",
  "kernel:metal:navatala_ml_knn_brute_force_cosine_f32",
  "abi-r1:metal:navatala_ml_knn_brute_force_cosine_f32",
  "abi-r1:metal:navatala_ml_knn_brute_force_cosine_f32",
  8,
  kAbiArgs_metal_navatala_ml_knn_brute_force_cosine_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_knn_brute_force_manhattan_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "data", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "q", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_knn_brute_force_manhattan_f32 = {
  1,
  "navatala_ml_knn_brute_force_manhattan_f32",
  "metal",
  "navatala_ml_knn_brute_force_manhattan_f32",
  "kernel:metal:navatala_ml_knn_brute_force_manhattan_f32",
  "abi-r1:metal:navatala_ml_knn_brute_force_manhattan_f32",
  "abi-r1:metal:navatala_ml_knn_brute_force_manhattan_f32",
  8,
  kAbiArgs_metal_navatala_ml_knn_brute_force_manhattan_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_knn_with_radius_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "data", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "q", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "d", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "radius", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_knn_with_radius_f32 = {
  1,
  "navatala_ml_knn_with_radius_f32",
  "metal",
  "navatala_ml_knn_with_radius_f32",
  "kernel:metal:navatala_ml_knn_with_radius_f32",
  "abi-r1:metal:navatala_ml_knn_with_radius_f32",
  "abi-r1:metal:navatala_ml_knn_with_radius_f32",
  10,
  kAbiArgs_metal_navatala_ml_knn_with_radius_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_knn_classify_f32[] = {
  { "neighborIndices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "labels", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "q", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClasses", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_knn_classify_f32 = {
  1,
  "navatala_ml_knn_classify_f32",
  "metal",
  "navatala_ml_knn_classify_f32",
  "kernel:metal:navatala_ml_knn_classify_f32",
  "abi-r1:metal:navatala_ml_knn_classify_f32",
  "abi-r1:metal:navatala_ml_knn_classify_f32",
  6,
  kAbiArgs_metal_navatala_ml_knn_classify_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_knn_classify_f64[] = {
  { "neighborIndices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "labels", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "q", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClasses", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_knn_classify_f64 = {
  1,
  "navatala_ml_knn_classify_f64",
  "metal",
  "navatala_ml_knn_classify_f64",
  "kernel:metal:navatala_ml_knn_classify_f64",
  "abi-r1:metal:navatala_ml_knn_classify_f64",
  "abi-r1:metal:navatala_ml_knn_classify_f64",
  6,
  kAbiArgs_metal_navatala_ml_knn_classify_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_knn_classify_weighted_f32[] = {
  { "neighborIndices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighborDistances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "labels", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "q", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClasses", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_knn_classify_weighted_f32 = {
  1,
  "navatala_ml_knn_classify_weighted_f32",
  "metal",
  "navatala_ml_knn_classify_weighted_f32",
  "kernel:metal:navatala_ml_knn_classify_weighted_f32",
  "abi-r1:metal:navatala_ml_knn_classify_weighted_f32",
  "abi-r1:metal:navatala_ml_knn_classify_weighted_f32",
  7,
  kAbiArgs_metal_navatala_ml_knn_classify_weighted_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_knn_classify_probs_f32[] = {
  { "neighborIndices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "labels", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "q", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClasses", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "probs", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_knn_classify_probs_f32 = {
  1,
  "navatala_ml_knn_classify_probs_f32",
  "metal",
  "navatala_ml_knn_classify_probs_f32",
  "kernel:metal:navatala_ml_knn_classify_probs_f32",
  "abi-r1:metal:navatala_ml_knn_classify_probs_f32",
  "abi-r1:metal:navatala_ml_knn_classify_probs_f32",
  6,
  kAbiArgs_metal_navatala_ml_knn_classify_probs_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_vote_majority_f32[] = {
  { "neighborLabels", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "q", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClasses", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_vote_majority_f32 = {
  1,
  "navatala_ml_vote_majority_f32",
  "metal",
  "navatala_ml_vote_majority_f32",
  "kernel:metal:navatala_ml_vote_majority_f32",
  "abi-r1:metal:navatala_ml_vote_majority_f32",
  "abi-r1:metal:navatala_ml_vote_majority_f32",
  5,
  kAbiArgs_metal_navatala_ml_vote_majority_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_vote_majority_f64[] = {
  { "neighborLabels", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "q", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numClasses", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_vote_majority_f64 = {
  1,
  "navatala_ml_vote_majority_f64",
  "metal",
  "navatala_ml_vote_majority_f64",
  "kernel:metal:navatala_ml_vote_majority_f64",
  "abi-r1:metal:navatala_ml_vote_majority_f64",
  "abi-r1:metal:navatala_ml_vote_majority_f64",
  5,
  kAbiArgs_metal_navatala_ml_vote_majority_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_knn_regress_f32[] = {
  { "neighborIndices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "targetValues", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "q", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_knn_regress_f32 = {
  1,
  "navatala_ml_knn_regress_f32",
  "metal",
  "navatala_ml_knn_regress_f32",
  "kernel:metal:navatala_ml_knn_regress_f32",
  "abi-r1:metal:navatala_ml_knn_regress_f32",
  "abi-r1:metal:navatala_ml_knn_regress_f32",
  5,
  kAbiArgs_metal_navatala_ml_knn_regress_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_knn_regress_weighted_f32[] = {
  { "neighborIndices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighborDistances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "targetValues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "q", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_knn_regress_weighted_f32 = {
  1,
  "navatala_ml_knn_regress_weighted_f32",
  "metal",
  "navatala_ml_knn_regress_weighted_f32",
  "kernel:metal:navatala_ml_knn_regress_weighted_f32",
  "abi-r1:metal:navatala_ml_knn_regress_weighted_f32",
  "abi-r1:metal:navatala_ml_knn_regress_weighted_f32",
  6,
  kAbiArgs_metal_navatala_ml_knn_regress_weighted_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_knn_regress_median_f32[] = {
  { "neighborIndices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "targetValues", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "q", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_knn_regress_median_f32 = {
  1,
  "navatala_ml_knn_regress_median_f32",
  "metal",
  "navatala_ml_knn_regress_median_f32",
  "kernel:metal:navatala_ml_knn_regress_median_f32",
  "abi-r1:metal:navatala_ml_knn_regress_median_f32",
  "abi-r1:metal:navatala_ml_knn_regress_median_f32",
  5,
  kAbiArgs_metal_navatala_ml_knn_regress_median_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_traverse_tree_classify_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "feature_indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thresholds", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "left_children", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "right_children", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "leaf_value_indices", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "leaf_values", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_features", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_classes", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_traverse_tree_classify_f32 = {
  1,
  "navatala_ml_traverse_tree_classify_f32",
  "metal",
  "navatala_ml_traverse_tree_classify_f32",
  "kernel:metal:navatala_ml_traverse_tree_classify_f32",
  "abi-r1:metal:navatala_ml_traverse_tree_classify_f32",
  "abi-r1:metal:navatala_ml_traverse_tree_classify_f32",
  11,
  kAbiArgs_metal_navatala_ml_traverse_tree_classify_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_traverse_tree_regress_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "feature_indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thresholds", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "left_children", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "right_children", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "leaf_value_indices", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "leaf_values", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_features", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_traverse_tree_regress_f32 = {
  1,
  "navatala_ml_traverse_tree_regress_f32",
  "metal",
  "navatala_ml_traverse_tree_regress_f32",
  "kernel:metal:navatala_ml_traverse_tree_regress_f32",
  "abi-r1:metal:navatala_ml_traverse_tree_regress_f32",
  "abi-r1:metal:navatala_ml_traverse_tree_regress_f32",
  10,
  kAbiArgs_metal_navatala_ml_traverse_tree_regress_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_gini_impurity_f32[] = {
  { "labels", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sample_indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_node_samples", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_classes", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gini", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "class_counts", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_gini_impurity_f32 = {
  1,
  "navatala_ml_compute_gini_impurity_f32",
  "metal",
  "navatala_ml_compute_gini_impurity_f32",
  "kernel:metal:navatala_ml_compute_gini_impurity_f32",
  "abi-r1:metal:navatala_ml_compute_gini_impurity_f32",
  "abi-r1:metal:navatala_ml_compute_gini_impurity_f32",
  6,
  kAbiArgs_metal_navatala_ml_compute_gini_impurity_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_m_s_e_f32[] = {
  { "targets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sample_indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_node_samples", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "mse", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "mean_val", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_m_s_e_f32 = {
  1,
  "navatala_ml_compute_m_s_e_f32",
  "metal",
  "navatala_ml_compute_m_s_e_f32",
  "kernel:metal:navatala_ml_compute_m_s_e_f32",
  "abi-r1:metal:navatala_ml_compute_m_s_e_f32",
  "abi-r1:metal:navatala_ml_compute_m_s_e_f32",
  5,
  kAbiArgs_metal_navatala_ml_compute_m_s_e_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_split_samples_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sample_indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_node_samples", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_features", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "split_feature", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "threshold", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "left_indices", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "right_indices", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "left_count", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "right_count", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_split_samples_f32 = {
  1,
  "navatala_ml_split_samples_f32",
  "metal",
  "navatala_ml_split_samples_f32",
  "kernel:metal:navatala_ml_split_samples_f32",
  "abi-r1:metal:navatala_ml_split_samples_f32",
  "abi-r1:metal:navatala_ml_split_samples_f32",
  10,
  kAbiArgs_metal_navatala_ml_split_samples_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_leaf_class_probs_f32[] = {
  { "labels", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sample_indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_leaf_samples", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_classes", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "probs", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_leaf_class_probs_f32 = {
  1,
  "navatala_ml_compute_leaf_class_probs_f32",
  "metal",
  "navatala_ml_compute_leaf_class_probs_f32",
  "kernel:metal:navatala_ml_compute_leaf_class_probs_f32",
  "abi-r1:metal:navatala_ml_compute_leaf_class_probs_f32",
  "abi-r1:metal:navatala_ml_compute_leaf_class_probs_f32",
  6,
  kAbiArgs_metal_navatala_ml_compute_leaf_class_probs_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_forest_classify_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "feature_indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thresholds", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "left_children", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "right_children", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "leaf_value_indices", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "forest_leaves", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tree_offsets", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "leaf_offsets", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_features", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_trees", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_classes", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vote_counts", 14, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_forest_classify_f32 = {
  1,
  "navatala_ml_forest_classify_f32",
  "metal",
  "navatala_ml_forest_classify_f32",
  "kernel:metal:navatala_ml_forest_classify_f32",
  "abi-r1:metal:navatala_ml_forest_classify_f32",
  "abi-r1:metal:navatala_ml_forest_classify_f32",
  15,
  kAbiArgs_metal_navatala_ml_forest_classify_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_forest_regress_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "feature_indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thresholds", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "left_children", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "right_children", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "leaf_value_indices", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "forest_leaves", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tree_offsets", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "leaf_offsets", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_features", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_trees", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 12, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_forest_regress_f32 = {
  1,
  "navatala_ml_forest_regress_f32",
  "metal",
  "navatala_ml_forest_regress_f32",
  "kernel:metal:navatala_ml_forest_regress_f32",
  "abi-r1:metal:navatala_ml_forest_regress_f32",
  "abi-r1:metal:navatala_ml_forest_regress_f32",
  13,
  kAbiArgs_metal_navatala_ml_forest_regress_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_aggregate_importances_f32[] = {
  { "tree_importances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_trees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_features", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "importances", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_aggregate_importances_f32 = {
  1,
  "navatala_ml_aggregate_importances_f32",
  "metal",
  "navatala_ml_aggregate_importances_f32",
  "kernel:metal:navatala_ml_aggregate_importances_f32",
  "abi-r1:metal:navatala_ml_aggregate_importances_f32",
  "abi-r1:metal:navatala_ml_aggregate_importances_f32",
  4,
  kAbiArgs_metal_navatala_ml_aggregate_importances_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_fil_infer_regression_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "feature_indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thresholds", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "left_children", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "right_children", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tree_offsets", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bias", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_samples", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_features", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_trees", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_fil_infer_regression_f32 = {
  1,
  "navatala_ml_fil_infer_regression_f32",
  "metal",
  "navatala_ml_fil_infer_regression_f32",
  "kernel:metal:navatala_ml_fil_infer_regression_f32",
  "abi-r1:metal:navatala_ml_fil_infer_regression_f32",
  "abi-r1:metal:navatala_ml_fil_infer_regression_f32",
  11,
  kAbiArgs_metal_navatala_ml_fil_infer_regression_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_fil_infer_classification_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "feature_indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thresholds", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "left_children", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "right_children", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tree_offsets", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_features", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_trees", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_classes", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vote_counts", 11, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_fil_infer_classification_f32 = {
  1,
  "navatala_ml_fil_infer_classification_f32",
  "metal",
  "navatala_ml_fil_infer_classification_f32",
  "kernel:metal:navatala_ml_fil_infer_classification_f32",
  "abi-r1:metal:navatala_ml_fil_infer_classification_f32",
  "abi-r1:metal:navatala_ml_fil_infer_classification_f32",
  12,
  kAbiArgs_metal_navatala_ml_fil_infer_classification_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_svm_predict_classify_f32[] = {
  { "support_vectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dual_coefs", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "X", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bias", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gamma", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_samples", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_sv", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_features", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_svm_predict_classify_f32 = {
  1,
  "navatala_ml_svm_predict_classify_f32",
  "metal",
  "navatala_ml_svm_predict_classify_f32",
  "kernel:metal:navatala_ml_svm_predict_classify_f32",
  "abi-r1:metal:navatala_ml_svm_predict_classify_f32",
  "abi-r1:metal:navatala_ml_svm_predict_classify_f32",
  9,
  kAbiArgs_metal_navatala_ml_svm_predict_classify_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_svm_predict_regress_f32[] = {
  { "support_vectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dual_coefs", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "X", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bias", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gamma", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_samples", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_sv", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_features", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_svm_predict_regress_f32 = {
  1,
  "navatala_ml_svm_predict_regress_f32",
  "metal",
  "navatala_ml_svm_predict_regress_f32",
  "kernel:metal:navatala_ml_svm_predict_regress_f32",
  "abi-r1:metal:navatala_ml_svm_predict_regress_f32",
  "abi-r1:metal:navatala_ml_svm_predict_regress_f32",
  9,
  kAbiArgs_metal_navatala_ml_svm_predict_regress_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_decision_function_f32[] = {
  { "support_vectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dual_coefs", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "X", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bias", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gamma", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_samples", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_sv", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_features", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "decision_values", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_decision_function_f32 = {
  1,
  "navatala_ml_compute_decision_function_f32",
  "metal",
  "navatala_ml_compute_decision_function_f32",
  "kernel:metal:navatala_ml_compute_decision_function_f32",
  "abi-r1:metal:navatala_ml_compute_decision_function_f32",
  "abi-r1:metal:navatala_ml_compute_decision_function_f32",
  9,
  kAbiArgs_metal_navatala_ml_compute_decision_function_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_r_b_f_kernel_matrix_f32[] = {
  { "X1", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "X2", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n1", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n2", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_features", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "K", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_r_b_f_kernel_matrix_f32 = {
  1,
  "navatala_ml_compute_r_b_f_kernel_matrix_f32",
  "metal",
  "navatala_ml_compute_r_b_f_kernel_matrix_f32",
  "kernel:metal:navatala_ml_compute_r_b_f_kernel_matrix_f32",
  "abi-r1:metal:navatala_ml_compute_r_b_f_kernel_matrix_f32",
  "abi-r1:metal:navatala_ml_compute_r_b_f_kernel_matrix_f32",
  7,
  kAbiArgs_metal_navatala_ml_compute_r_b_f_kernel_matrix_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_linear_kernel_matrix_f32[] = {
  { "X1", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "X2", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n1", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n2", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_features", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "K", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_linear_kernel_matrix_f32 = {
  1,
  "navatala_ml_compute_linear_kernel_matrix_f32",
  "metal",
  "navatala_ml_compute_linear_kernel_matrix_f32",
  "kernel:metal:navatala_ml_compute_linear_kernel_matrix_f32",
  "abi-r1:metal:navatala_ml_compute_linear_kernel_matrix_f32",
  "abi-r1:metal:navatala_ml_compute_linear_kernel_matrix_f32",
  6,
  kAbiArgs_metal_navatala_ml_compute_linear_kernel_matrix_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_polynomial_kernel_matrix_f32[] = {
  { "X1", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "X2", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coef0", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n1", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n2", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_features", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "K", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_polynomial_kernel_matrix_f32 = {
  1,
  "navatala_ml_compute_polynomial_kernel_matrix_f32",
  "metal",
  "navatala_ml_compute_polynomial_kernel_matrix_f32",
  "kernel:metal:navatala_ml_compute_polynomial_kernel_matrix_f32",
  "abi-r1:metal:navatala_ml_compute_polynomial_kernel_matrix_f32",
  "abi-r1:metal:navatala_ml_compute_polynomial_kernel_matrix_f32",
  9,
  kAbiArgs_metal_navatala_ml_compute_polynomial_kernel_matrix_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_core_distances_f32[] = {
  { "distances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "min_samples", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "core_dists", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_core_distances_f32 = {
  1,
  "navatala_ml_compute_core_distances_f32",
  "metal",
  "navatala_ml_compute_core_distances_f32",
  "kernel:metal:navatala_ml_compute_core_distances_f32",
  "abi-r1:metal:navatala_ml_compute_core_distances_f32",
  "abi-r1:metal:navatala_ml_compute_core_distances_f32",
  4,
  kAbiArgs_metal_navatala_ml_compute_core_distances_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_mutual_reachability_f32[] = {
  { "distances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "core_dists", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "mutual_reach", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_mutual_reachability_f32 = {
  1,
  "navatala_ml_compute_mutual_reachability_f32",
  "metal",
  "navatala_ml_compute_mutual_reachability_f32",
  "kernel:metal:navatala_ml_compute_mutual_reachability_f32",
  "abi-r1:metal:navatala_ml_compute_mutual_reachability_f32",
  "abi-r1:metal:navatala_ml_compute_mutual_reachability_f32",
  4,
  kAbiArgs_metal_navatala_ml_compute_mutual_reachability_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_extract_cluster_labels_f32[] = {
  { "selected_clusters", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "point_cluster_map", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cluster_parent_map", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_clusters", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labels", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_extract_cluster_labels_f32 = {
  1,
  "navatala_ml_extract_cluster_labels_f32",
  "metal",
  "navatala_ml_extract_cluster_labels_f32",
  "kernel:metal:navatala_ml_extract_cluster_labels_f32",
  "abi-r1:metal:navatala_ml_extract_cluster_labels_f32",
  "abi-r1:metal:navatala_ml_extract_cluster_labels_f32",
  6,
  kAbiArgs_metal_navatala_ml_extract_cluster_labels_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_extract_cluster_labels_f64[] = {
  { "selected_clusters", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "point_cluster_map", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cluster_parent_map", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_clusters", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labels", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_extract_cluster_labels_f64 = {
  1,
  "navatala_ml_extract_cluster_labels_f64",
  "metal",
  "navatala_ml_extract_cluster_labels_f64",
  "kernel:metal:navatala_ml_extract_cluster_labels_f64",
  "abi-r1:metal:navatala_ml_extract_cluster_labels_f64",
  "abi-r1:metal:navatala_ml_extract_cluster_labels_f64",
  6,
  kAbiArgs_metal_navatala_ml_extract_cluster_labels_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_probabilities_f32[] = {
  { "labels", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lambdas", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cluster_lambdas_max", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_clusters", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "probabilities", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_probabilities_f32 = {
  1,
  "navatala_ml_compute_probabilities_f32",
  "metal",
  "navatala_ml_compute_probabilities_f32",
  "kernel:metal:navatala_ml_compute_probabilities_f32",
  "abi-r1:metal:navatala_ml_compute_probabilities_f32",
  "abi-r1:metal:navatala_ml_compute_probabilities_f32",
  6,
  kAbiArgs_metal_navatala_ml_compute_probabilities_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_cluster_stability_f32[] = {
  { "point_lambdas", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cluster_ids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_edges", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_clusters", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stabilities", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_cluster_stability_f32 = {
  1,
  "navatala_ml_compute_cluster_stability_f32",
  "metal",
  "navatala_ml_compute_cluster_stability_f32",
  "kernel:metal:navatala_ml_compute_cluster_stability_f32",
  "abi-r1:metal:navatala_ml_compute_cluster_stability_f32",
  "abi-r1:metal:navatala_ml_compute_cluster_stability_f32",
  5,
  kAbiArgs_metal_navatala_ml_compute_cluster_stability_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_initialize_labels_f32[] = {
  { "n_samples", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labels", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_initialize_labels_f32 = {
  1,
  "navatala_ml_initialize_labels_f32",
  "metal",
  "navatala_ml_initialize_labels_f32",
  "kernel:metal:navatala_ml_initialize_labels_f32",
  "abi-r1:metal:navatala_ml_initialize_labels_f32",
  "abi-r1:metal:navatala_ml_initialize_labels_f32",
  2,
  kAbiArgs_metal_navatala_ml_initialize_labels_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_initialize_labels_f64[] = {
  { "n_samples", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labels", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_initialize_labels_f64 = {
  1,
  "navatala_ml_initialize_labels_f64",
  "metal",
  "navatala_ml_initialize_labels_f64",
  "kernel:metal:navatala_ml_initialize_labels_f64",
  "abi-r1:metal:navatala_ml_initialize_labels_f64",
  "abi-r1:metal:navatala_ml_initialize_labels_f64",
  2,
  kAbiArgs_metal_navatala_ml_initialize_labels_f64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_fuzzy_simplicial_set_f32[] = {
  { "knn_indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "knn_dists", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_neighbors", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "local_connectivity", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "membership", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rho", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sigma", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_fuzzy_simplicial_set_f32 = {
  1,
  "navatala_ml_compute_fuzzy_simplicial_set_f32",
  "metal",
  "navatala_ml_compute_fuzzy_simplicial_set_f32",
  "kernel:metal:navatala_ml_compute_fuzzy_simplicial_set_f32",
  "abi-r1:metal:navatala_ml_compute_fuzzy_simplicial_set_f32",
  "abi-r1:metal:navatala_ml_compute_fuzzy_simplicial_set_f32",
  8,
  kAbiArgs_metal_navatala_ml_compute_fuzzy_simplicial_set_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_attraction_f32[] = {
  { "embedding", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edge_from", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edge_to", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edge_weights", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_edges", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_components", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "a_param", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "b_param", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradients", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_attraction_f32 = {
  1,
  "navatala_ml_compute_attraction_f32",
  "metal",
  "navatala_ml_compute_attraction_f32",
  "kernel:metal:navatala_ml_compute_attraction_f32",
  "abi-r1:metal:navatala_ml_compute_attraction_f32",
  "abi-r1:metal:navatala_ml_compute_attraction_f32",
  9,
  kAbiArgs_metal_navatala_ml_compute_attraction_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_repulsion_f32[] = {
  { "embedding", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "negative_samples", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_neg", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_components", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "a_param", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "b_param", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gamma", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradients", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_repulsion_f32 = {
  1,
  "navatala_ml_compute_repulsion_f32",
  "metal",
  "navatala_ml_compute_repulsion_f32",
  "kernel:metal:navatala_ml_compute_repulsion_f32",
  "abi-r1:metal:navatala_ml_compute_repulsion_f32",
  "abi-r1:metal:navatala_ml_compute_repulsion_f32",
  9,
  kAbiArgs_metal_navatala_ml_compute_repulsion_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_optimize_layout_f32[] = {
  { "embedding", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradients", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_components", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "embedding_out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_optimize_layout_f32 = {
  1,
  "navatala_ml_optimize_layout_f32",
  "metal",
  "navatala_ml_optimize_layout_f32",
  "kernel:metal:navatala_ml_optimize_layout_f32",
  "abi-r1:metal:navatala_ml_optimize_layout_f32",
  "abi-r1:metal:navatala_ml_optimize_layout_f32",
  6,
  kAbiArgs_metal_navatala_ml_optimize_layout_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_initialize_embedding_f32[] = {
  { "seed", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_samples", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_components", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "spread", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "embedding", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_initialize_embedding_f32 = {
  1,
  "navatala_ml_initialize_embedding_f32",
  "metal",
  "navatala_ml_initialize_embedding_f32",
  "kernel:metal:navatala_ml_initialize_embedding_f32",
  "abi-r1:metal:navatala_ml_initialize_embedding_f32",
  "abi-r1:metal:navatala_ml_initialize_embedding_f32",
  5,
  kAbiArgs_metal_navatala_ml_initialize_embedding_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_clear_gradients_f32[] = {
  { "n_samples", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_components", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradients", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_clear_gradients_f32 = {
  1,
  "navatala_ml_clear_gradients_f32",
  "metal",
  "navatala_ml_clear_gradients_f32",
  "kernel:metal:navatala_ml_clear_gradients_f32",
  "abi-r1:metal:navatala_ml_clear_gradients_f32",
  "abi-r1:metal:navatala_ml_clear_gradients_f32",
  3,
  kAbiArgs_metal_navatala_ml_clear_gradients_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_differencing_f32[] = {
  { "y", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_obs", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batch_size", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y_diff", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_differencing_f32 = {
  1,
  "navatala_ml_differencing_f32",
  "metal",
  "navatala_ml_differencing_f32",
  "kernel:metal:navatala_ml_differencing_f32",
  "abi-r1:metal:navatala_ml_differencing_f32",
  "abi-r1:metal:navatala_ml_differencing_f32",
  4,
  kAbiArgs_metal_navatala_ml_differencing_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_holt_winters_forecast_f32[] = {
  { "level", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "trend", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "season", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "period", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batch_size", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "horizon", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "forecasts", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_holt_winters_forecast_f32 = {
  1,
  "navatala_ml_holt_winters_forecast_f32",
  "metal",
  "navatala_ml_holt_winters_forecast_f32",
  "kernel:metal:navatala_ml_holt_winters_forecast_f32",
  "abi-r1:metal:navatala_ml_holt_winters_forecast_f32",
  "abi-r1:metal:navatala_ml_holt_winters_forecast_f32",
  7,
  kAbiArgs_metal_navatala_ml_holt_winters_forecast_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_ar_predict_f32[] = {
  { "y", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phi", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "intercept", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "p", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batch_size", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_ar_predict_f32 = {
  1,
  "navatala_ml_ar_predict_f32",
  "metal",
  "navatala_ml_ar_predict_f32",
  "kernel:metal:navatala_ml_ar_predict_f32",
  "abi-r1:metal:navatala_ml_ar_predict_f32",
  "abi-r1:metal:navatala_ml_ar_predict_f32",
  6,
  kAbiArgs_metal_navatala_ml_ar_predict_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_shap_values_f32[] = {
  { "predictions", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coalitions", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nSamples", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nFeatures", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "shapValues", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_shap_values_f32 = {
  1,
  "navatala_ml_shap_values_f32",
  "metal",
  "navatala_ml_shap_values_f32",
  "kernel:metal:navatala_ml_shap_values_f32",
  "abi-r1:metal:navatala_ml_shap_values_f32",
  "abi-r1:metal:navatala_ml_shap_values_f32",
  6,
  kAbiArgs_metal_navatala_ml_shap_values_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_coalition_marginal_f32[] = {
  { "predWith", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "predWithout", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nSamples", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "marginals", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_coalition_marginal_f32 = {
  1,
  "navatala_ml_coalition_marginal_f32",
  "metal",
  "navatala_ml_coalition_marginal_f32",
  "kernel:metal:navatala_ml_coalition_marginal_f32",
  "abi-r1:metal:navatala_ml_coalition_marginal_f32",
  "abi-r1:metal:navatala_ml_coalition_marginal_f32",
  4,
  kAbiArgs_metal_navatala_ml_coalition_marginal_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_feature_interaction_f32[] = {
  { "jointContrib", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "shapValues", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nFeatures", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "interactions", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_feature_interaction_f32 = {
  1,
  "navatala_ml_feature_interaction_f32",
  "metal",
  "navatala_ml_feature_interaction_f32",
  "kernel:metal:navatala_ml_feature_interaction_f32",
  "abi-r1:metal:navatala_ml_feature_interaction_f32",
  "abi-r1:metal:navatala_ml_feature_interaction_f32",
  4,
  kAbiArgs_metal_navatala_ml_feature_interaction_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_correlations_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "residual", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nCols", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "correlations", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_correlations_f32 = {
  1,
  "navatala_ml_compute_correlations_f32",
  "metal",
  "navatala_ml_compute_correlations_f32",
  "kernel:metal:navatala_ml_compute_correlations_f32",
  "abi-r1:metal:navatala_ml_compute_correlations_f32",
  "abi-r1:metal:navatala_ml_compute_correlations_f32",
  5,
  kAbiArgs_metal_navatala_ml_compute_correlations_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_find_max_correlation_f32[] = {
  { "correlations", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "activeMask", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxIndex", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxValue", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_find_max_correlation_f32 = {
  1,
  "navatala_ml_find_max_correlation_f32",
  "metal",
  "navatala_ml_find_max_correlation_f32",
  "kernel:metal:navatala_ml_find_max_correlation_f32",
  "abi-r1:metal:navatala_ml_find_max_correlation_f32",
  "abi-r1:metal:navatala_ml_find_max_correlation_f32",
  5,
  kAbiArgs_metal_navatala_ml_find_max_correlation_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_update_coefs_f32[] = {
  { "equiangular", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "activeMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCols", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_update_coefs_f32 = {
  1,
  "navatala_ml_update_coefs_f32",
  "metal",
  "navatala_ml_update_coefs_f32",
  "kernel:metal:navatala_ml_update_coefs_f32",
  "abi-r1:metal:navatala_ml_update_coefs_f32",
  "abi-r1:metal:navatala_ml_update_coefs_f32",
  5,
  kAbiArgs_metal_navatala_ml_update_coefs_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_compute_equiangular_f32[] = {
  { "correlations", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "activeMask", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "equiangular", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_compute_equiangular_f32 = {
  1,
  "navatala_ml_compute_equiangular_f32",
  "metal",
  "navatala_ml_compute_equiangular_f32",
  "kernel:metal:navatala_ml_compute_equiangular_f32",
  "abi-r1:metal:navatala_ml_compute_equiangular_f32",
  "abi-r1:metal:navatala_ml_compute_equiangular_f32",
  4,
  kAbiArgs_metal_navatala_ml_compute_equiangular_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_evaluate_fitness_f32[] = {
  { "predictions", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "targets", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "popSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nSamples", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "fitness", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_evaluate_fitness_f32 = {
  1,
  "navatala_ml_evaluate_fitness_f32",
  "metal",
  "navatala_ml_evaluate_fitness_f32",
  "kernel:metal:navatala_ml_evaluate_fitness_f32",
  "abi-r1:metal:navatala_ml_evaluate_fitness_f32",
  "abi-r1:metal:navatala_ml_evaluate_fitness_f32",
  5,
  kAbiArgs_metal_navatala_ml_evaluate_fitness_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_mutate_population_f32[] = {
  { "population", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rngState", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mutationRate", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "mutationStrength", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalGenes", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "mutated", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rngStateOut", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_mutate_population_f32 = {
  1,
  "navatala_ml_mutate_population_f32",
  "metal",
  "navatala_ml_mutate_population_f32",
  "kernel:metal:navatala_ml_mutate_population_f32",
  "abi-r1:metal:navatala_ml_mutate_population_f32",
  "abi-r1:metal:navatala_ml_mutate_population_f32",
  7,
  kAbiArgs_metal_navatala_ml_mutate_population_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_crossover_population_f32[] = {
  { "parent1", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "parent2", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "crossoverPoints", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nPairs", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indivSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "offspring1", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offspring2", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_crossover_population_f32 = {
  1,
  "navatala_ml_crossover_population_f32",
  "metal",
  "navatala_ml_crossover_population_f32",
  "kernel:metal:navatala_ml_crossover_population_f32",
  "abi-r1:metal:navatala_ml_crossover_population_f32",
  "abi-r1:metal:navatala_ml_crossover_population_f32",
  7,
  kAbiArgs_metal_navatala_ml_crossover_population_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_select_tournament_f32[] = {
  { "fitness", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tournamentIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nTournaments", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "tournamentSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "selectedIndices", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_select_tournament_f32 = {
  1,
  "navatala_ml_select_tournament_f32",
  "metal",
  "navatala_ml_select_tournament_f32",
  "kernel:metal:navatala_ml_select_tournament_f32",
  "abi-r1:metal:navatala_ml_select_tournament_f32",
  "abi-r1:metal:navatala_ml_select_tournament_f32",
  5,
  kAbiArgs_metal_navatala_ml_select_tournament_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_batch_traverse_trees_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "feature_indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thresholds", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "left_children", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "right_children", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tree_offsets", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_features", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_trees", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "leaf_indices", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_batch_traverse_trees_f32 = {
  1,
  "navatala_ml_batch_traverse_trees_f32",
  "metal",
  "navatala_ml_batch_traverse_trees_f32",
  "kernel:metal:navatala_ml_batch_traverse_trees_f32",
  "abi-r1:metal:navatala_ml_batch_traverse_trees_f32",
  "abi-r1:metal:navatala_ml_batch_traverse_trees_f32",
  10,
  kAbiArgs_metal_navatala_ml_batch_traverse_trees_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_collect_leaf_values_f32[] = {
  { "leaf_indices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thresholds", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tree_offsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_trees", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "leaf_values", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_collect_leaf_values_f32 = {
  1,
  "navatala_ml_collect_leaf_values_f32",
  "metal",
  "navatala_ml_collect_leaf_values_f32",
  "kernel:metal:navatala_ml_collect_leaf_values_f32",
  "abi-r1:metal:navatala_ml_collect_leaf_values_f32",
  "abi-r1:metal:navatala_ml_collect_leaf_values_f32",
  6,
  kAbiArgs_metal_navatala_ml_collect_leaf_values_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_aggregate_tree_outputs_f32[] = {
  { "leaf_values", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_trees", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "predictions", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_aggregate_tree_outputs_f32 = {
  1,
  "navatala_ml_aggregate_tree_outputs_f32",
  "metal",
  "navatala_ml_aggregate_tree_outputs_f32",
  "kernel:metal:navatala_ml_aggregate_tree_outputs_f32",
  "abi-r1:metal:navatala_ml_aggregate_tree_outputs_f32",
  "abi-r1:metal:navatala_ml_aggregate_tree_outputs_f32",
  4,
  kAbiArgs_metal_navatala_ml_aggregate_tree_outputs_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_traverse_with_path_f32[] = {
  { "X", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "feature_indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thresholds", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "left_children", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "right_children", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tree_offsets", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_samples", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_features", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_trees", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_depth", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "paths", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "path_lengths", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_traverse_with_path_f32 = {
  1,
  "navatala_ml_traverse_with_path_f32",
  "metal",
  "navatala_ml_traverse_with_path_f32",
  "kernel:metal:navatala_ml_traverse_with_path_f32",
  "abi-r1:metal:navatala_ml_traverse_with_path_f32",
  "abi-r1:metal:navatala_ml_traverse_with_path_f32",
  12,
  kAbiArgs_metal_navatala_ml_traverse_with_path_f32
};

bool tryGetKernelAbiManifest_metal_ml(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_ml_init_ema_f32") {
    out = &kAbiManifest_metal_navatala_ml_init_ema_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_init_ema_from_span_f32") {
    out = &kAbiManifest_metal_navatala_ml_init_ema_from_span_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_ema_update_f32") {
    out = &kAbiManifest_metal_navatala_ml_ema_update_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_ema_extract_f32") {
    out = &kAbiManifest_metal_navatala_ml_ema_extract_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_ema_reset_f32") {
    out = &kAbiManifest_metal_navatala_ml_ema_reset_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_init_dema_f32") {
    out = &kAbiManifest_metal_navatala_ml_init_dema_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_dema_update_f32") {
    out = &kAbiManifest_metal_navatala_ml_dema_update_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_dema_extract_f32") {
    out = &kAbiManifest_metal_navatala_ml_dema_extract_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_maxpool_f32") {
    out = &kAbiManifest_metal_navatala_ml_maxpool_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_avgpool_exclude_pad_f32") {
    out = &kAbiManifest_metal_navatala_ml_avgpool_exclude_pad_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_avgpool_include_pad_f32") {
    out = &kAbiManifest_metal_navatala_ml_avgpool_include_pad_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_resize_nearest_f32") {
    out = &kAbiManifest_metal_navatala_ml_resize_nearest_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_resize_bilinear_f32") {
    out = &kAbiManifest_metal_navatala_ml_resize_bilinear_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_maxpool_f16") {
    out = &kAbiManifest_metal_navatala_ml_maxpool_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_avgpool_exclude_pad_f16") {
    out = &kAbiManifest_metal_navatala_ml_avgpool_exclude_pad_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_avgpool_include_pad_f16") {
    out = &kAbiManifest_metal_navatala_ml_avgpool_include_pad_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_resize_nearest_f16") {
    out = &kAbiManifest_metal_navatala_ml_resize_nearest_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_resize_bilinear_f16") {
    out = &kAbiManifest_metal_navatala_ml_resize_bilinear_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reshape_f32") {
    out = &kAbiManifest_metal_navatala_ml_reshape_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_slice_f32") {
    out = &kAbiManifest_metal_navatala_ml_slice_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_transpose_f32") {
    out = &kAbiManifest_metal_navatala_ml_transpose_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_concatenate2_f32") {
    out = &kAbiManifest_metal_navatala_ml_concatenate2_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reshape_f16") {
    out = &kAbiManifest_metal_navatala_ml_reshape_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_slice_f16") {
    out = &kAbiManifest_metal_navatala_ml_slice_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_transpose_f16") {
    out = &kAbiManifest_metal_navatala_ml_transpose_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_concatenate2_f16") {
    out = &kAbiManifest_metal_navatala_ml_concatenate2_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_softmax_f32") {
    out = &kAbiManifest_metal_navatala_ml_softmax_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_softmax_f16") {
    out = &kAbiManifest_metal_navatala_ml_softmax_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_layernorm_f32") {
    out = &kAbiManifest_metal_navatala_ml_layernorm_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_spatial_batchnorm_f32") {
    out = &kAbiManifest_metal_navatala_ml_spatial_batchnorm_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_instancenorm_f32") {
    out = &kAbiManifest_metal_navatala_ml_instancenorm_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_batchnorm_inference_f32") {
    out = &kAbiManifest_metal_navatala_ml_batchnorm_inference_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_layernorm_backward_f32") {
    out = &kAbiManifest_metal_navatala_ml_layernorm_backward_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_layernorm_f16") {
    out = &kAbiManifest_metal_navatala_ml_layernorm_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_spatial_batchnorm_f16") {
    out = &kAbiManifest_metal_navatala_ml_spatial_batchnorm_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_instancenorm_f16") {
    out = &kAbiManifest_metal_navatala_ml_instancenorm_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_batchnorm_inference_f16") {
    out = &kAbiManifest_metal_navatala_ml_batchnorm_inference_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_layernorm_backward_f16") {
    out = &kAbiManifest_metal_navatala_ml_layernorm_backward_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_uniform_f32") {
    out = &kAbiManifest_metal_navatala_ml_uniform_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_bernoulli_f32") {
    out = &kAbiManifest_metal_navatala_ml_bernoulli_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_normal_f32") {
    out = &kAbiManifest_metal_navatala_ml_normal_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_uniform_f16") {
    out = &kAbiManifest_metal_navatala_ml_uniform_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_bernoulli_f16") {
    out = &kAbiManifest_metal_navatala_ml_bernoulli_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_normal_f16") {
    out = &kAbiManifest_metal_navatala_ml_normal_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_kde_bandwidth_silverman_f32") {
    out = &kAbiManifest_metal_navatala_ml_kde_bandwidth_silverman_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_kde_bandwidth_scott_f32") {
    out = &kAbiManifest_metal_navatala_ml_kde_bandwidth_scott_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_kde_evaluate_gaussian_f32") {
    out = &kAbiManifest_metal_navatala_ml_kde_evaluate_gaussian_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_kde_evaluate_epanechnikov_f32") {
    out = &kAbiManifest_metal_navatala_ml_kde_evaluate_epanechnikov_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_kde_find_mode_f32") {
    out = &kAbiManifest_metal_navatala_ml_kde_find_mode_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pairwise_euclidean_f32") {
    out = &kAbiManifest_metal_navatala_ml_pairwise_euclidean_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pairwise_squared_euclidean_f32") {
    out = &kAbiManifest_metal_navatala_ml_pairwise_squared_euclidean_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pairwise_manhattan_f32") {
    out = &kAbiManifest_metal_navatala_ml_pairwise_manhattan_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pairwise_cosine_f32") {
    out = &kAbiManifest_metal_navatala_ml_pairwise_cosine_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pairwise_chebyshev_f32") {
    out = &kAbiManifest_metal_navatala_ml_pairwise_chebyshev_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_nearest_neighbor_f32") {
    out = &kAbiManifest_metal_navatala_ml_nearest_neighbor_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_init_centroids_random_f32") {
    out = &kAbiManifest_metal_navatala_ml_init_centroids_random_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_assign_to_centroids_f32") {
    out = &kAbiManifest_metal_navatala_ml_assign_to_centroids_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_cluster_sums_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_cluster_sums_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_update_centroids_f32") {
    out = &kAbiManifest_metal_navatala_ml_update_centroids_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_inertia_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_inertia_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_init_gmm_state_f32") {
    out = &kAbiManifest_metal_navatala_ml_init_gmm_state_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_gaussian_log_pdf_f32") {
    out = &kAbiManifest_metal_navatala_ml_gaussian_log_pdf_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_responsibilities_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_responsibilities_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_update_means_f32") {
    out = &kAbiManifest_metal_navatala_ml_update_means_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_update_variances_f32") {
    out = &kAbiManifest_metal_navatala_ml_update_variances_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_update_weights_f32") {
    out = &kAbiManifest_metal_navatala_ml_update_weights_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_log_likelihood_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_log_likelihood_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_gram_matrix_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_gram_matrix_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_xt_y_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_xt_y_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_predict_f32") {
    out = &kAbiManifest_metal_navatala_ml_predict_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_residuals_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_residuals_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_r_squared_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_r_squared_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_mean_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_mean_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_add_regularization_f32") {
    out = &kAbiManifest_metal_navatala_ml_add_regularization_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_ridge_predict_f32") {
    out = &kAbiManifest_metal_navatala_ml_ridge_predict_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_gcv_score_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_gcv_score_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_effective_dof_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_effective_dof_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_init_lasso_state_f32") {
    out = &kAbiManifest_metal_navatala_ml_init_lasso_state_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_partial_residual_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_partial_residual_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_soft_threshold_f32") {
    out = &kAbiManifest_metal_navatala_ml_soft_threshold_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_coordinate_update_f32") {
    out = &kAbiManifest_metal_navatala_ml_coordinate_update_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_objective_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_objective_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_active_set_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_active_set_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_center_data_f32") {
    out = &kAbiManifest_metal_navatala_ml_center_data_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_covariance_matrix_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_covariance_matrix_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_project_to_principal_f32") {
    out = &kAbiManifest_metal_navatala_ml_project_to_principal_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_explained_variance_ratio_f32") {
    out = &kAbiManifest_metal_navatala_ml_explained_variance_ratio_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_pairwise_affinity_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_pairwise_affinity_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_symmetrize_affinity_f32") {
    out = &kAbiManifest_metal_navatala_ml_symmetrize_affinity_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_init_embedding_random_f32") {
    out = &kAbiManifest_metal_navatala_ml_init_embedding_random_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_q_distribution_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_q_distribution_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_gradient_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_gradient_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_apply_momentum_update_f32") {
    out = &kAbiManifest_metal_navatala_ml_apply_momentum_update_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_k_l_divergence_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_k_l_divergence_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_perplexity_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_perplexity_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_normalize_embedding_f32") {
    out = &kAbiManifest_metal_navatala_ml_normalize_embedding_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_exaggerated_p_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_exaggerated_p_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_neighbor_counts_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_neighbor_counts_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_identify_core_points_f32") {
    out = &kAbiManifest_metal_navatala_ml_identify_core_points_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_identify_core_points_f64") {
    out = &kAbiManifest_metal_navatala_ml_identify_core_points_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_init_cluster_labels_f32") {
    out = &kAbiManifest_metal_navatala_ml_init_cluster_labels_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_init_cluster_labels_f64") {
    out = &kAbiManifest_metal_navatala_ml_init_cluster_labels_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_expand_cluster_step_f32") {
    out = &kAbiManifest_metal_navatala_ml_expand_cluster_step_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_propagate_core_labels_f32") {
    out = &kAbiManifest_metal_navatala_ml_propagate_core_labels_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_mark_noise_points_f32") {
    out = &kAbiManifest_metal_navatala_ml_mark_noise_points_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_mark_noise_points_f64") {
    out = &kAbiManifest_metal_navatala_ml_mark_noise_points_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_count_clusters_f32") {
    out = &kAbiManifest_metal_navatala_ml_count_clusters_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_count_clusters_f64") {
    out = &kAbiManifest_metal_navatala_ml_count_clusters_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_neighborhood_recall_f32") {
    out = &kAbiManifest_metal_navatala_ml_neighborhood_recall_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_trustworthiness_f32") {
    out = &kAbiManifest_metal_navatala_ml_trustworthiness_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_continuity_f32") {
    out = &kAbiManifest_metal_navatala_ml_continuity_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_make_blobs_f32") {
    out = &kAbiManifest_metal_navatala_ml_make_blobs_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_make_blobs_anisotropic_f32") {
    out = &kAbiManifest_metal_navatala_ml_make_blobs_anisotropic_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_make_moons_f32") {
    out = &kAbiManifest_metal_navatala_ml_make_moons_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_make_circles_f32") {
    out = &kAbiManifest_metal_navatala_ml_make_circles_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_make_regression_f32") {
    out = &kAbiManifest_metal_navatala_ml_make_regression_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_make_classification_f32") {
    out = &kAbiManifest_metal_navatala_ml_make_classification_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_make_sparse_uncorrelated_f32") {
    out = &kAbiManifest_metal_navatala_ml_make_sparse_uncorrelated_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_brute_force_f32") {
    out = &kAbiManifest_metal_navatala_ml_knn_brute_force_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_brute_force_euclidean_f32") {
    out = &kAbiManifest_metal_navatala_ml_knn_brute_force_euclidean_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_brute_force_cosine_f32") {
    out = &kAbiManifest_metal_navatala_ml_knn_brute_force_cosine_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_brute_force_manhattan_f32") {
    out = &kAbiManifest_metal_navatala_ml_knn_brute_force_manhattan_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_with_radius_f32") {
    out = &kAbiManifest_metal_navatala_ml_knn_with_radius_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_classify_f32") {
    out = &kAbiManifest_metal_navatala_ml_knn_classify_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_classify_f64") {
    out = &kAbiManifest_metal_navatala_ml_knn_classify_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_classify_weighted_f32") {
    out = &kAbiManifest_metal_navatala_ml_knn_classify_weighted_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_classify_probs_f32") {
    out = &kAbiManifest_metal_navatala_ml_knn_classify_probs_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_vote_majority_f32") {
    out = &kAbiManifest_metal_navatala_ml_vote_majority_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_vote_majority_f64") {
    out = &kAbiManifest_metal_navatala_ml_vote_majority_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_regress_f32") {
    out = &kAbiManifest_metal_navatala_ml_knn_regress_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_regress_weighted_f32") {
    out = &kAbiManifest_metal_navatala_ml_knn_regress_weighted_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_regress_median_f32") {
    out = &kAbiManifest_metal_navatala_ml_knn_regress_median_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_traverse_tree_classify_f32") {
    out = &kAbiManifest_metal_navatala_ml_traverse_tree_classify_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_traverse_tree_regress_f32") {
    out = &kAbiManifest_metal_navatala_ml_traverse_tree_regress_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_gini_impurity_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_gini_impurity_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_m_s_e_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_m_s_e_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_split_samples_f32") {
    out = &kAbiManifest_metal_navatala_ml_split_samples_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_leaf_class_probs_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_leaf_class_probs_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_forest_classify_f32") {
    out = &kAbiManifest_metal_navatala_ml_forest_classify_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_forest_regress_f32") {
    out = &kAbiManifest_metal_navatala_ml_forest_regress_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_aggregate_importances_f32") {
    out = &kAbiManifest_metal_navatala_ml_aggregate_importances_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_fil_infer_regression_f32") {
    out = &kAbiManifest_metal_navatala_ml_fil_infer_regression_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_fil_infer_classification_f32") {
    out = &kAbiManifest_metal_navatala_ml_fil_infer_classification_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_svm_predict_classify_f32") {
    out = &kAbiManifest_metal_navatala_ml_svm_predict_classify_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_svm_predict_regress_f32") {
    out = &kAbiManifest_metal_navatala_ml_svm_predict_regress_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_decision_function_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_decision_function_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_r_b_f_kernel_matrix_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_r_b_f_kernel_matrix_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_linear_kernel_matrix_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_linear_kernel_matrix_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_polynomial_kernel_matrix_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_polynomial_kernel_matrix_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_core_distances_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_core_distances_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_mutual_reachability_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_mutual_reachability_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_extract_cluster_labels_f32") {
    out = &kAbiManifest_metal_navatala_ml_extract_cluster_labels_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_extract_cluster_labels_f64") {
    out = &kAbiManifest_metal_navatala_ml_extract_cluster_labels_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_probabilities_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_probabilities_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_cluster_stability_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_cluster_stability_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_initialize_labels_f32") {
    out = &kAbiManifest_metal_navatala_ml_initialize_labels_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_initialize_labels_f64") {
    out = &kAbiManifest_metal_navatala_ml_initialize_labels_f64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_fuzzy_simplicial_set_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_fuzzy_simplicial_set_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_attraction_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_attraction_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_repulsion_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_repulsion_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_optimize_layout_f32") {
    out = &kAbiManifest_metal_navatala_ml_optimize_layout_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_initialize_embedding_f32") {
    out = &kAbiManifest_metal_navatala_ml_initialize_embedding_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_clear_gradients_f32") {
    out = &kAbiManifest_metal_navatala_ml_clear_gradients_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_differencing_f32") {
    out = &kAbiManifest_metal_navatala_ml_differencing_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_holt_winters_forecast_f32") {
    out = &kAbiManifest_metal_navatala_ml_holt_winters_forecast_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_ar_predict_f32") {
    out = &kAbiManifest_metal_navatala_ml_ar_predict_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_shap_values_f32") {
    out = &kAbiManifest_metal_navatala_ml_shap_values_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_coalition_marginal_f32") {
    out = &kAbiManifest_metal_navatala_ml_coalition_marginal_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_feature_interaction_f32") {
    out = &kAbiManifest_metal_navatala_ml_feature_interaction_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_correlations_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_correlations_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_find_max_correlation_f32") {
    out = &kAbiManifest_metal_navatala_ml_find_max_correlation_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_update_coefs_f32") {
    out = &kAbiManifest_metal_navatala_ml_update_coefs_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_equiangular_f32") {
    out = &kAbiManifest_metal_navatala_ml_compute_equiangular_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_evaluate_fitness_f32") {
    out = &kAbiManifest_metal_navatala_ml_evaluate_fitness_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_mutate_population_f32") {
    out = &kAbiManifest_metal_navatala_ml_mutate_population_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_crossover_population_f32") {
    out = &kAbiManifest_metal_navatala_ml_crossover_population_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_select_tournament_f32") {
    out = &kAbiManifest_metal_navatala_ml_select_tournament_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_batch_traverse_trees_f32") {
    out = &kAbiManifest_metal_navatala_ml_batch_traverse_trees_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_collect_leaf_values_f32") {
    out = &kAbiManifest_metal_navatala_ml_collect_leaf_values_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_aggregate_tree_outputs_f32") {
    out = &kAbiManifest_metal_navatala_ml_aggregate_tree_outputs_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_traverse_with_path_f32") {
    out = &kAbiManifest_metal_navatala_ml_traverse_with_path_f32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_ml(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_ml_init_ema_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_init_ema_f32";
    std::string_view sv(k_metal_navatala_ml_init_ema_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_init_ema_from_span_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_init_ema_from_span_f32";
    std::string_view sv(k_metal_navatala_ml_init_ema_from_span_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_ema_update_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_ema_update_f32";
    std::string_view sv(k_metal_navatala_ml_ema_update_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_ema_extract_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_ema_extract_f32";
    std::string_view sv(k_metal_navatala_ml_ema_extract_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_ema_reset_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_ema_reset_f32";
    std::string_view sv(k_metal_navatala_ml_ema_reset_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_init_dema_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_init_dema_f32";
    std::string_view sv(k_metal_navatala_ml_init_dema_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_dema_update_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_dema_update_f32";
    std::string_view sv(k_metal_navatala_ml_dema_update_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_dema_extract_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_dema_extract_f32";
    std::string_view sv(k_metal_navatala_ml_dema_extract_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_maxpool_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_maxpool_f32";
    std::string_view sv(k_metal_navatala_ml_maxpool_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_avgpool_exclude_pad_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_avgpool_exclude_pad_f32";
    std::string_view sv(k_metal_navatala_ml_avgpool_exclude_pad_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_avgpool_include_pad_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_avgpool_include_pad_f32";
    std::string_view sv(k_metal_navatala_ml_avgpool_include_pad_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_resize_nearest_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_resize_nearest_f32";
    std::string_view sv(k_metal_navatala_ml_resize_nearest_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_resize_bilinear_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_resize_bilinear_f32";
    std::string_view sv(k_metal_navatala_ml_resize_bilinear_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_maxpool_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_maxpool_f16";
    std::string_view sv(k_metal_navatala_ml_maxpool_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_avgpool_exclude_pad_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_avgpool_exclude_pad_f16";
    std::string_view sv(k_metal_navatala_ml_avgpool_exclude_pad_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_avgpool_include_pad_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_avgpool_include_pad_f16";
    std::string_view sv(k_metal_navatala_ml_avgpool_include_pad_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_resize_nearest_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_resize_nearest_f16";
    std::string_view sv(k_metal_navatala_ml_resize_nearest_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_resize_bilinear_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_resize_bilinear_f16";
    std::string_view sv(k_metal_navatala_ml_resize_bilinear_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reshape_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reshape_f32";
    std::string_view sv(k_metal_navatala_ml_reshape_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_slice_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_slice_f32";
    std::string_view sv(k_metal_navatala_ml_slice_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_transpose_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_transpose_f32";
    std::string_view sv(k_metal_navatala_ml_transpose_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_concatenate2_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_concatenate2_f32";
    std::string_view sv(k_metal_navatala_ml_concatenate2_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reshape_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reshape_f16";
    std::string_view sv(k_metal_navatala_ml_reshape_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_slice_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_slice_f16";
    std::string_view sv(k_metal_navatala_ml_slice_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_transpose_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_transpose_f16";
    std::string_view sv(k_metal_navatala_ml_transpose_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_concatenate2_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_concatenate2_f16";
    std::string_view sv(k_metal_navatala_ml_concatenate2_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_softmax_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_softmax_f32";
    std::string_view sv(k_metal_navatala_ml_softmax_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_softmax_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_softmax_f16";
    std::string_view sv(k_metal_navatala_ml_softmax_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_layernorm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_layernorm_f32";
    std::string_view sv(k_metal_navatala_ml_layernorm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_spatial_batchnorm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_spatial_batchnorm_f32";
    std::string_view sv(k_metal_navatala_ml_spatial_batchnorm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_instancenorm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_instancenorm_f32";
    std::string_view sv(k_metal_navatala_ml_instancenorm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_batchnorm_inference_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_batchnorm_inference_f32";
    std::string_view sv(k_metal_navatala_ml_batchnorm_inference_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_layernorm_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_layernorm_backward_f32";
    std::string_view sv(k_metal_navatala_ml_layernorm_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_layernorm_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_layernorm_f16";
    std::string_view sv(k_metal_navatala_ml_layernorm_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_spatial_batchnorm_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_spatial_batchnorm_f16";
    std::string_view sv(k_metal_navatala_ml_spatial_batchnorm_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_instancenorm_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_instancenorm_f16";
    std::string_view sv(k_metal_navatala_ml_instancenorm_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_batchnorm_inference_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_batchnorm_inference_f16";
    std::string_view sv(k_metal_navatala_ml_batchnorm_inference_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_layernorm_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_layernorm_backward_f16";
    std::string_view sv(k_metal_navatala_ml_layernorm_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_uniform_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_uniform_f32";
    std::string_view sv(k_metal_navatala_ml_uniform_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_bernoulli_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_bernoulli_f32";
    std::string_view sv(k_metal_navatala_ml_bernoulli_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_normal_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_normal_f32";
    std::string_view sv(k_metal_navatala_ml_normal_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_uniform_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_uniform_f16";
    std::string_view sv(k_metal_navatala_ml_uniform_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_bernoulli_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_bernoulli_f16";
    std::string_view sv(k_metal_navatala_ml_bernoulli_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_normal_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_normal_f16";
    std::string_view sv(k_metal_navatala_ml_normal_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_kde_bandwidth_silverman_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_kde_bandwidth_silverman_f32";
    std::string_view sv(k_metal_navatala_ml_kde_bandwidth_silverman_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_kde_bandwidth_scott_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_kde_bandwidth_scott_f32";
    std::string_view sv(k_metal_navatala_ml_kde_bandwidth_scott_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_kde_evaluate_gaussian_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_kde_evaluate_gaussian_f32";
    std::string_view sv(k_metal_navatala_ml_kde_evaluate_gaussian_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_kde_evaluate_epanechnikov_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_kde_evaluate_epanechnikov_f32";
    std::string_view sv(k_metal_navatala_ml_kde_evaluate_epanechnikov_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_kde_find_mode_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_kde_find_mode_f32";
    std::string_view sv(k_metal_navatala_ml_kde_find_mode_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pairwise_euclidean_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pairwise_euclidean_f32";
    std::string_view sv(k_metal_navatala_ml_pairwise_euclidean_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pairwise_squared_euclidean_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pairwise_squared_euclidean_f32";
    std::string_view sv(k_metal_navatala_ml_pairwise_squared_euclidean_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pairwise_manhattan_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pairwise_manhattan_f32";
    std::string_view sv(k_metal_navatala_ml_pairwise_manhattan_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pairwise_cosine_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pairwise_cosine_f32";
    std::string_view sv(k_metal_navatala_ml_pairwise_cosine_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_pairwise_chebyshev_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_pairwise_chebyshev_f32";
    std::string_view sv(k_metal_navatala_ml_pairwise_chebyshev_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_nearest_neighbor_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_nearest_neighbor_f32";
    std::string_view sv(k_metal_navatala_ml_nearest_neighbor_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_init_centroids_random_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_init_centroids_random_f32";
    std::string_view sv(k_metal_navatala_ml_init_centroids_random_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_assign_to_centroids_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_assign_to_centroids_f32";
    std::string_view sv(k_metal_navatala_ml_assign_to_centroids_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_cluster_sums_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_cluster_sums_f32";
    std::string_view sv(k_metal_navatala_ml_compute_cluster_sums_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_update_centroids_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_update_centroids_f32";
    std::string_view sv(k_metal_navatala_ml_update_centroids_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_inertia_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_inertia_f32";
    std::string_view sv(k_metal_navatala_ml_compute_inertia_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_init_gmm_state_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_init_gmm_state_f32";
    std::string_view sv(k_metal_navatala_ml_init_gmm_state_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_gaussian_log_pdf_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_gaussian_log_pdf_f32";
    std::string_view sv(k_metal_navatala_ml_gaussian_log_pdf_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_responsibilities_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_responsibilities_f32";
    std::string_view sv(k_metal_navatala_ml_compute_responsibilities_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_update_means_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_update_means_f32";
    std::string_view sv(k_metal_navatala_ml_update_means_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_update_variances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_update_variances_f32";
    std::string_view sv(k_metal_navatala_ml_update_variances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_update_weights_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_update_weights_f32";
    std::string_view sv(k_metal_navatala_ml_update_weights_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_log_likelihood_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_log_likelihood_f32";
    std::string_view sv(k_metal_navatala_ml_compute_log_likelihood_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_gram_matrix_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_gram_matrix_f32";
    std::string_view sv(k_metal_navatala_ml_compute_gram_matrix_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_xt_y_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_xt_y_f32";
    std::string_view sv(k_metal_navatala_ml_compute_xt_y_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_predict_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_predict_f32";
    std::string_view sv(k_metal_navatala_ml_predict_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_residuals_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_residuals_f32";
    std::string_view sv(k_metal_navatala_ml_compute_residuals_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_r_squared_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_r_squared_f32";
    std::string_view sv(k_metal_navatala_ml_compute_r_squared_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_mean_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_mean_f32";
    std::string_view sv(k_metal_navatala_ml_compute_mean_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_add_regularization_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_add_regularization_f32";
    std::string_view sv(k_metal_navatala_ml_add_regularization_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_ridge_predict_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_ridge_predict_f32";
    std::string_view sv(k_metal_navatala_ml_ridge_predict_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_gcv_score_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_gcv_score_f32";
    std::string_view sv(k_metal_navatala_ml_compute_gcv_score_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_effective_dof_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_effective_dof_f32";
    std::string_view sv(k_metal_navatala_ml_compute_effective_dof_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_init_lasso_state_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_init_lasso_state_f32";
    std::string_view sv(k_metal_navatala_ml_init_lasso_state_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_partial_residual_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_partial_residual_f32";
    std::string_view sv(k_metal_navatala_ml_compute_partial_residual_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_soft_threshold_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_soft_threshold_f32";
    std::string_view sv(k_metal_navatala_ml_soft_threshold_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_coordinate_update_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_coordinate_update_f32";
    std::string_view sv(k_metal_navatala_ml_coordinate_update_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_objective_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_objective_f32";
    std::string_view sv(k_metal_navatala_ml_compute_objective_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_active_set_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_active_set_f32";
    std::string_view sv(k_metal_navatala_ml_compute_active_set_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_center_data_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_center_data_f32";
    std::string_view sv(k_metal_navatala_ml_center_data_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_covariance_matrix_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_covariance_matrix_f32";
    std::string_view sv(k_metal_navatala_ml_compute_covariance_matrix_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_project_to_principal_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_project_to_principal_f32";
    std::string_view sv(k_metal_navatala_ml_project_to_principal_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_explained_variance_ratio_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_explained_variance_ratio_f32";
    std::string_view sv(k_metal_navatala_ml_explained_variance_ratio_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_pairwise_affinity_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_pairwise_affinity_f32";
    std::string_view sv(k_metal_navatala_ml_compute_pairwise_affinity_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_symmetrize_affinity_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_symmetrize_affinity_f32";
    std::string_view sv(k_metal_navatala_ml_symmetrize_affinity_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_init_embedding_random_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_init_embedding_random_f32";
    std::string_view sv(k_metal_navatala_ml_init_embedding_random_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_q_distribution_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_q_distribution_f32";
    std::string_view sv(k_metal_navatala_ml_compute_q_distribution_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_gradient_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_gradient_f32";
    std::string_view sv(k_metal_navatala_ml_compute_gradient_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_apply_momentum_update_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_apply_momentum_update_f32";
    std::string_view sv(k_metal_navatala_ml_apply_momentum_update_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_k_l_divergence_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_k_l_divergence_f32";
    std::string_view sv(k_metal_navatala_ml_compute_k_l_divergence_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_perplexity_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_perplexity_f32";
    std::string_view sv(k_metal_navatala_ml_compute_perplexity_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_normalize_embedding_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_normalize_embedding_f32";
    std::string_view sv(k_metal_navatala_ml_normalize_embedding_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_exaggerated_p_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_exaggerated_p_f32";
    std::string_view sv(k_metal_navatala_ml_compute_exaggerated_p_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_neighbor_counts_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_neighbor_counts_f32";
    std::string_view sv(k_metal_navatala_ml_compute_neighbor_counts_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_identify_core_points_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_identify_core_points_f32";
    std::string_view sv(k_metal_navatala_ml_identify_core_points_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_identify_core_points_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_identify_core_points_f64";
    std::string_view sv(k_metal_navatala_ml_identify_core_points_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_init_cluster_labels_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_init_cluster_labels_f32";
    std::string_view sv(k_metal_navatala_ml_init_cluster_labels_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_init_cluster_labels_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_init_cluster_labels_f64";
    std::string_view sv(k_metal_navatala_ml_init_cluster_labels_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_expand_cluster_step_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_expand_cluster_step_f32";
    std::string_view sv(k_metal_navatala_ml_expand_cluster_step_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_propagate_core_labels_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_propagate_core_labels_f32";
    std::string_view sv(k_metal_navatala_ml_propagate_core_labels_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_mark_noise_points_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_mark_noise_points_f32";
    std::string_view sv(k_metal_navatala_ml_mark_noise_points_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_mark_noise_points_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_mark_noise_points_f64";
    std::string_view sv(k_metal_navatala_ml_mark_noise_points_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_count_clusters_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_count_clusters_f32";
    std::string_view sv(k_metal_navatala_ml_count_clusters_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_count_clusters_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_count_clusters_f64";
    std::string_view sv(k_metal_navatala_ml_count_clusters_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_neighborhood_recall_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_neighborhood_recall_f32";
    std::string_view sv(k_metal_navatala_ml_neighborhood_recall_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_trustworthiness_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_trustworthiness_f32";
    std::string_view sv(k_metal_navatala_ml_trustworthiness_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_continuity_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_continuity_f32";
    std::string_view sv(k_metal_navatala_ml_continuity_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_make_blobs_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_make_blobs_f32";
    std::string_view sv(k_metal_navatala_ml_make_blobs_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_make_blobs_anisotropic_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_make_blobs_anisotropic_f32";
    std::string_view sv(k_metal_navatala_ml_make_blobs_anisotropic_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_make_moons_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_make_moons_f32";
    std::string_view sv(k_metal_navatala_ml_make_moons_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_make_circles_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_make_circles_f32";
    std::string_view sv(k_metal_navatala_ml_make_circles_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_make_regression_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_make_regression_f32";
    std::string_view sv(k_metal_navatala_ml_make_regression_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_make_classification_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_make_classification_f32";
    std::string_view sv(k_metal_navatala_ml_make_classification_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_make_sparse_uncorrelated_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_make_sparse_uncorrelated_f32";
    std::string_view sv(k_metal_navatala_ml_make_sparse_uncorrelated_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_brute_force_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_knn_brute_force_f32";
    std::string_view sv(k_metal_navatala_ml_knn_brute_force_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_brute_force_euclidean_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_knn_brute_force_euclidean_f32";
    std::string_view sv(k_metal_navatala_ml_knn_brute_force_euclidean_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_brute_force_cosine_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_knn_brute_force_cosine_f32";
    std::string_view sv(k_metal_navatala_ml_knn_brute_force_cosine_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_brute_force_manhattan_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_knn_brute_force_manhattan_f32";
    std::string_view sv(k_metal_navatala_ml_knn_brute_force_manhattan_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_with_radius_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_knn_with_radius_f32";
    std::string_view sv(k_metal_navatala_ml_knn_with_radius_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_classify_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_knn_classify_f32";
    std::string_view sv(k_metal_navatala_ml_knn_classify_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_classify_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_knn_classify_f64";
    std::string_view sv(k_metal_navatala_ml_knn_classify_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_classify_weighted_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_knn_classify_weighted_f32";
    std::string_view sv(k_metal_navatala_ml_knn_classify_weighted_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_classify_probs_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_knn_classify_probs_f32";
    std::string_view sv(k_metal_navatala_ml_knn_classify_probs_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_vote_majority_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_vote_majority_f32";
    std::string_view sv(k_metal_navatala_ml_vote_majority_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_vote_majority_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_vote_majority_f64";
    std::string_view sv(k_metal_navatala_ml_vote_majority_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_regress_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_knn_regress_f32";
    std::string_view sv(k_metal_navatala_ml_knn_regress_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_regress_weighted_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_knn_regress_weighted_f32";
    std::string_view sv(k_metal_navatala_ml_knn_regress_weighted_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_knn_regress_median_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_knn_regress_median_f32";
    std::string_view sv(k_metal_navatala_ml_knn_regress_median_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_traverse_tree_classify_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_traverse_tree_classify_f32";
    std::string_view sv(k_metal_navatala_ml_traverse_tree_classify_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_traverse_tree_regress_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_traverse_tree_regress_f32";
    std::string_view sv(k_metal_navatala_ml_traverse_tree_regress_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_gini_impurity_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_gini_impurity_f32";
    std::string_view sv(k_metal_navatala_ml_compute_gini_impurity_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_m_s_e_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_m_s_e_f32";
    std::string_view sv(k_metal_navatala_ml_compute_m_s_e_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_split_samples_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_split_samples_f32";
    std::string_view sv(k_metal_navatala_ml_split_samples_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_leaf_class_probs_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_leaf_class_probs_f32";
    std::string_view sv(k_metal_navatala_ml_compute_leaf_class_probs_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_forest_classify_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_forest_classify_f32";
    std::string_view sv(k_metal_navatala_ml_forest_classify_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_forest_regress_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_forest_regress_f32";
    std::string_view sv(k_metal_navatala_ml_forest_regress_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_aggregate_importances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_aggregate_importances_f32";
    std::string_view sv(k_metal_navatala_ml_aggregate_importances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_fil_infer_regression_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_fil_infer_regression_f32";
    std::string_view sv(k_metal_navatala_ml_fil_infer_regression_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_fil_infer_classification_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_fil_infer_classification_f32";
    std::string_view sv(k_metal_navatala_ml_fil_infer_classification_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_svm_predict_classify_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_svm_predict_classify_f32";
    std::string_view sv(k_metal_navatala_ml_svm_predict_classify_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_svm_predict_regress_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_svm_predict_regress_f32";
    std::string_view sv(k_metal_navatala_ml_svm_predict_regress_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_decision_function_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_decision_function_f32";
    std::string_view sv(k_metal_navatala_ml_compute_decision_function_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_r_b_f_kernel_matrix_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_r_b_f_kernel_matrix_f32";
    std::string_view sv(k_metal_navatala_ml_compute_r_b_f_kernel_matrix_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_linear_kernel_matrix_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_linear_kernel_matrix_f32";
    std::string_view sv(k_metal_navatala_ml_compute_linear_kernel_matrix_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_polynomial_kernel_matrix_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_polynomial_kernel_matrix_f32";
    std::string_view sv(k_metal_navatala_ml_compute_polynomial_kernel_matrix_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_core_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_core_distances_f32";
    std::string_view sv(k_metal_navatala_ml_compute_core_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_mutual_reachability_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_mutual_reachability_f32";
    std::string_view sv(k_metal_navatala_ml_compute_mutual_reachability_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_extract_cluster_labels_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_extract_cluster_labels_f32";
    std::string_view sv(k_metal_navatala_ml_extract_cluster_labels_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_extract_cluster_labels_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_extract_cluster_labels_f64";
    std::string_view sv(k_metal_navatala_ml_extract_cluster_labels_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_probabilities_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_probabilities_f32";
    std::string_view sv(k_metal_navatala_ml_compute_probabilities_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_cluster_stability_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_cluster_stability_f32";
    std::string_view sv(k_metal_navatala_ml_compute_cluster_stability_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_initialize_labels_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_initialize_labels_f32";
    std::string_view sv(k_metal_navatala_ml_initialize_labels_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_initialize_labels_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_initialize_labels_f64";
    std::string_view sv(k_metal_navatala_ml_initialize_labels_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_fuzzy_simplicial_set_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_fuzzy_simplicial_set_f32";
    std::string_view sv(k_metal_navatala_ml_compute_fuzzy_simplicial_set_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_attraction_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_attraction_f32";
    std::string_view sv(k_metal_navatala_ml_compute_attraction_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_repulsion_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_repulsion_f32";
    std::string_view sv(k_metal_navatala_ml_compute_repulsion_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_optimize_layout_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_optimize_layout_f32";
    std::string_view sv(k_metal_navatala_ml_optimize_layout_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_initialize_embedding_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_initialize_embedding_f32";
    std::string_view sv(k_metal_navatala_ml_initialize_embedding_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_clear_gradients_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_clear_gradients_f32";
    std::string_view sv(k_metal_navatala_ml_clear_gradients_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_differencing_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_differencing_f32";
    std::string_view sv(k_metal_navatala_ml_differencing_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_holt_winters_forecast_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_holt_winters_forecast_f32";
    std::string_view sv(k_metal_navatala_ml_holt_winters_forecast_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_ar_predict_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_ar_predict_f32";
    std::string_view sv(k_metal_navatala_ml_ar_predict_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_shap_values_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_shap_values_f32";
    std::string_view sv(k_metal_navatala_ml_shap_values_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_coalition_marginal_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_coalition_marginal_f32";
    std::string_view sv(k_metal_navatala_ml_coalition_marginal_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_feature_interaction_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_feature_interaction_f32";
    std::string_view sv(k_metal_navatala_ml_feature_interaction_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_correlations_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_correlations_f32";
    std::string_view sv(k_metal_navatala_ml_compute_correlations_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_find_max_correlation_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_find_max_correlation_f32";
    std::string_view sv(k_metal_navatala_ml_find_max_correlation_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_update_coefs_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_update_coefs_f32";
    std::string_view sv(k_metal_navatala_ml_update_coefs_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_compute_equiangular_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_compute_equiangular_f32";
    std::string_view sv(k_metal_navatala_ml_compute_equiangular_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_evaluate_fitness_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_evaluate_fitness_f32";
    std::string_view sv(k_metal_navatala_ml_evaluate_fitness_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_mutate_population_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_mutate_population_f32";
    std::string_view sv(k_metal_navatala_ml_mutate_population_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_crossover_population_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_crossover_population_f32";
    std::string_view sv(k_metal_navatala_ml_crossover_population_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_select_tournament_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_select_tournament_f32";
    std::string_view sv(k_metal_navatala_ml_select_tournament_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_batch_traverse_trees_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_batch_traverse_trees_f32";
    std::string_view sv(k_metal_navatala_ml_batch_traverse_trees_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_collect_leaf_values_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_collect_leaf_values_f32";
    std::string_view sv(k_metal_navatala_ml_collect_leaf_values_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_aggregate_tree_outputs_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_aggregate_tree_outputs_f32";
    std::string_view sv(k_metal_navatala_ml_aggregate_tree_outputs_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_traverse_with_path_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_traverse_with_path_f32";
    std::string_view sv(k_metal_navatala_ml_traverse_with_path_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

