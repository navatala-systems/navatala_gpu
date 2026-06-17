// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_ml_reduction` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_ml_reduction_sum_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reduction_sum_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = raw;
      gsAcc = (gsAcc + v);
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_reduction_prod_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reduction_prod_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0x3f800000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = raw;
      gsAcc = (gsAcc * v);
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine * other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_reduction_min_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reduction_min_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0x7f7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = raw;
      gsAcc = (((gsAcc < v)) ? (gsAcc) : (v));
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine < other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_reduction_max_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reduction_max_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0xff7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = raw;
      gsAcc = (((gsAcc > v)) ? (gsAcc) : (v));
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_reduction_amax_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reduction_amax_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0xff7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = abs(raw);
      gsAcc = (((gsAcc > v)) ? (gsAcc) : (v));
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_reduction_norm1_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reduction_norm1_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = abs(raw);
      gsAcc = (gsAcc + v);
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_reduction_avg_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reduction_avg_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = raw;
      gsAcc = (gsAcc + v);
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = (reduced / nF);
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_reduction_norm2_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reduction_norm2_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = (raw * raw);
      gsAcc = (gsAcc + v);
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = sqrt(reduced);
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_reduction_sum_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reduction_sum_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (gsAcc + v);
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_reduction_prod_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reduction_prod_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0x3f800000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (gsAcc * v);
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine * other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_reduction_min_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reduction_min_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0x7f7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (((gsAcc < v)) ? (gsAcc) : (v));
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine < other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_reduction_max_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reduction_max_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0xff7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (((gsAcc > v)) ? (gsAcc) : (v));
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_reduction_amax_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reduction_amax_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0xff7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = abs(((float)(raw)));
      gsAcc = (((gsAcc > v)) ? (gsAcc) : (v));
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_reduction_norm1_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reduction_norm1_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = abs(((float)(raw)));
      gsAcc = (gsAcc + v);
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_reduction_avg_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reduction_avg_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (gsAcc + v);
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = (reduced / nF);
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_metal_navatala_ml_reduction_norm2_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_reduction_norm2_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = (((float)(raw)) * ((float)(raw)));
      gsAcc = (gsAcc + v);
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = sqrt(reduced);
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reduction_sum_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reduction_sum_f32 = {
  1,
  "navatala_ml_reduction_sum_f32",
  "metal",
  "navatala_ml_reduction_sum_f32",
  "kernel:metal:navatala_ml_reduction_sum_f32",
  "abi-r1:metal:navatala_ml_reduction_sum_f32",
  "abi-r1:metal:navatala_ml_reduction_sum_f32",
  3,
  kAbiArgs_metal_navatala_ml_reduction_sum_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reduction_prod_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reduction_prod_f32 = {
  1,
  "navatala_ml_reduction_prod_f32",
  "metal",
  "navatala_ml_reduction_prod_f32",
  "kernel:metal:navatala_ml_reduction_prod_f32",
  "abi-r1:metal:navatala_ml_reduction_prod_f32",
  "abi-r1:metal:navatala_ml_reduction_prod_f32",
  3,
  kAbiArgs_metal_navatala_ml_reduction_prod_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reduction_min_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reduction_min_f32 = {
  1,
  "navatala_ml_reduction_min_f32",
  "metal",
  "navatala_ml_reduction_min_f32",
  "kernel:metal:navatala_ml_reduction_min_f32",
  "abi-r1:metal:navatala_ml_reduction_min_f32",
  "abi-r1:metal:navatala_ml_reduction_min_f32",
  3,
  kAbiArgs_metal_navatala_ml_reduction_min_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reduction_max_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reduction_max_f32 = {
  1,
  "navatala_ml_reduction_max_f32",
  "metal",
  "navatala_ml_reduction_max_f32",
  "kernel:metal:navatala_ml_reduction_max_f32",
  "abi-r1:metal:navatala_ml_reduction_max_f32",
  "abi-r1:metal:navatala_ml_reduction_max_f32",
  3,
  kAbiArgs_metal_navatala_ml_reduction_max_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reduction_amax_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reduction_amax_f32 = {
  1,
  "navatala_ml_reduction_amax_f32",
  "metal",
  "navatala_ml_reduction_amax_f32",
  "kernel:metal:navatala_ml_reduction_amax_f32",
  "abi-r1:metal:navatala_ml_reduction_amax_f32",
  "abi-r1:metal:navatala_ml_reduction_amax_f32",
  3,
  kAbiArgs_metal_navatala_ml_reduction_amax_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reduction_norm1_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reduction_norm1_f32 = {
  1,
  "navatala_ml_reduction_norm1_f32",
  "metal",
  "navatala_ml_reduction_norm1_f32",
  "kernel:metal:navatala_ml_reduction_norm1_f32",
  "abi-r1:metal:navatala_ml_reduction_norm1_f32",
  "abi-r1:metal:navatala_ml_reduction_norm1_f32",
  3,
  kAbiArgs_metal_navatala_ml_reduction_norm1_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reduction_avg_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reduction_avg_f32 = {
  1,
  "navatala_ml_reduction_avg_f32",
  "metal",
  "navatala_ml_reduction_avg_f32",
  "kernel:metal:navatala_ml_reduction_avg_f32",
  "abi-r1:metal:navatala_ml_reduction_avg_f32",
  "abi-r1:metal:navatala_ml_reduction_avg_f32",
  3,
  kAbiArgs_metal_navatala_ml_reduction_avg_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reduction_norm2_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reduction_norm2_f32 = {
  1,
  "navatala_ml_reduction_norm2_f32",
  "metal",
  "navatala_ml_reduction_norm2_f32",
  "kernel:metal:navatala_ml_reduction_norm2_f32",
  "abi-r1:metal:navatala_ml_reduction_norm2_f32",
  "abi-r1:metal:navatala_ml_reduction_norm2_f32",
  3,
  kAbiArgs_metal_navatala_ml_reduction_norm2_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reduction_sum_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reduction_sum_f16 = {
  1,
  "navatala_ml_reduction_sum_f16",
  "metal",
  "navatala_ml_reduction_sum_f16",
  "kernel:metal:navatala_ml_reduction_sum_f16",
  "abi-r1:metal:navatala_ml_reduction_sum_f16",
  "abi-r1:metal:navatala_ml_reduction_sum_f16",
  3,
  kAbiArgs_metal_navatala_ml_reduction_sum_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reduction_prod_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reduction_prod_f16 = {
  1,
  "navatala_ml_reduction_prod_f16",
  "metal",
  "navatala_ml_reduction_prod_f16",
  "kernel:metal:navatala_ml_reduction_prod_f16",
  "abi-r1:metal:navatala_ml_reduction_prod_f16",
  "abi-r1:metal:navatala_ml_reduction_prod_f16",
  3,
  kAbiArgs_metal_navatala_ml_reduction_prod_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reduction_min_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reduction_min_f16 = {
  1,
  "navatala_ml_reduction_min_f16",
  "metal",
  "navatala_ml_reduction_min_f16",
  "kernel:metal:navatala_ml_reduction_min_f16",
  "abi-r1:metal:navatala_ml_reduction_min_f16",
  "abi-r1:metal:navatala_ml_reduction_min_f16",
  3,
  kAbiArgs_metal_navatala_ml_reduction_min_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reduction_max_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reduction_max_f16 = {
  1,
  "navatala_ml_reduction_max_f16",
  "metal",
  "navatala_ml_reduction_max_f16",
  "kernel:metal:navatala_ml_reduction_max_f16",
  "abi-r1:metal:navatala_ml_reduction_max_f16",
  "abi-r1:metal:navatala_ml_reduction_max_f16",
  3,
  kAbiArgs_metal_navatala_ml_reduction_max_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reduction_amax_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reduction_amax_f16 = {
  1,
  "navatala_ml_reduction_amax_f16",
  "metal",
  "navatala_ml_reduction_amax_f16",
  "kernel:metal:navatala_ml_reduction_amax_f16",
  "abi-r1:metal:navatala_ml_reduction_amax_f16",
  "abi-r1:metal:navatala_ml_reduction_amax_f16",
  3,
  kAbiArgs_metal_navatala_ml_reduction_amax_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reduction_norm1_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reduction_norm1_f16 = {
  1,
  "navatala_ml_reduction_norm1_f16",
  "metal",
  "navatala_ml_reduction_norm1_f16",
  "kernel:metal:navatala_ml_reduction_norm1_f16",
  "abi-r1:metal:navatala_ml_reduction_norm1_f16",
  "abi-r1:metal:navatala_ml_reduction_norm1_f16",
  3,
  kAbiArgs_metal_navatala_ml_reduction_norm1_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reduction_avg_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reduction_avg_f16 = {
  1,
  "navatala_ml_reduction_avg_f16",
  "metal",
  "navatala_ml_reduction_avg_f16",
  "kernel:metal:navatala_ml_reduction_avg_f16",
  "abi-r1:metal:navatala_ml_reduction_avg_f16",
  "abi-r1:metal:navatala_ml_reduction_avg_f16",
  3,
  kAbiArgs_metal_navatala_ml_reduction_avg_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_ml_reduction_norm2_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_ml_reduction_norm2_f16 = {
  1,
  "navatala_ml_reduction_norm2_f16",
  "metal",
  "navatala_ml_reduction_norm2_f16",
  "kernel:metal:navatala_ml_reduction_norm2_f16",
  "abi-r1:metal:navatala_ml_reduction_norm2_f16",
  "abi-r1:metal:navatala_ml_reduction_norm2_f16",
  3,
  kAbiArgs_metal_navatala_ml_reduction_norm2_f16
};

bool tryGetKernelAbiManifest_metal_ml_reduction(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_ml_reduction_sum_f32") {
    out = &kAbiManifest_metal_navatala_ml_reduction_sum_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_prod_f32") {
    out = &kAbiManifest_metal_navatala_ml_reduction_prod_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_min_f32") {
    out = &kAbiManifest_metal_navatala_ml_reduction_min_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_max_f32") {
    out = &kAbiManifest_metal_navatala_ml_reduction_max_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_amax_f32") {
    out = &kAbiManifest_metal_navatala_ml_reduction_amax_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_norm1_f32") {
    out = &kAbiManifest_metal_navatala_ml_reduction_norm1_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_avg_f32") {
    out = &kAbiManifest_metal_navatala_ml_reduction_avg_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_norm2_f32") {
    out = &kAbiManifest_metal_navatala_ml_reduction_norm2_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_sum_f16") {
    out = &kAbiManifest_metal_navatala_ml_reduction_sum_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_prod_f16") {
    out = &kAbiManifest_metal_navatala_ml_reduction_prod_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_min_f16") {
    out = &kAbiManifest_metal_navatala_ml_reduction_min_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_max_f16") {
    out = &kAbiManifest_metal_navatala_ml_reduction_max_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_amax_f16") {
    out = &kAbiManifest_metal_navatala_ml_reduction_amax_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_norm1_f16") {
    out = &kAbiManifest_metal_navatala_ml_reduction_norm1_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_avg_f16") {
    out = &kAbiManifest_metal_navatala_ml_reduction_avg_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_norm2_f16") {
    out = &kAbiManifest_metal_navatala_ml_reduction_norm2_f16;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_ml_reduction(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_ml_reduction_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reduction_sum_f32";
    std::string_view sv(k_metal_navatala_ml_reduction_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_prod_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reduction_prod_f32";
    std::string_view sv(k_metal_navatala_ml_reduction_prod_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_min_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reduction_min_f32";
    std::string_view sv(k_metal_navatala_ml_reduction_min_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reduction_max_f32";
    std::string_view sv(k_metal_navatala_ml_reduction_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_amax_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reduction_amax_f32";
    std::string_view sv(k_metal_navatala_ml_reduction_amax_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_norm1_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reduction_norm1_f32";
    std::string_view sv(k_metal_navatala_ml_reduction_norm1_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_avg_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reduction_avg_f32";
    std::string_view sv(k_metal_navatala_ml_reduction_avg_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_norm2_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reduction_norm2_f32";
    std::string_view sv(k_metal_navatala_ml_reduction_norm2_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_sum_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reduction_sum_f16";
    std::string_view sv(k_metal_navatala_ml_reduction_sum_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_prod_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reduction_prod_f16";
    std::string_view sv(k_metal_navatala_ml_reduction_prod_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_min_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reduction_min_f16";
    std::string_view sv(k_metal_navatala_ml_reduction_min_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_max_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reduction_max_f16";
    std::string_view sv(k_metal_navatala_ml_reduction_max_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_amax_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reduction_amax_f16";
    std::string_view sv(k_metal_navatala_ml_reduction_amax_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_norm1_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reduction_norm1_f16";
    std::string_view sv(k_metal_navatala_ml_reduction_norm1_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_avg_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reduction_avg_f16";
    std::string_view sv(k_metal_navatala_ml_reduction_avg_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_ml_reduction_norm2_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_ml_reduction_norm2_f16";
    std::string_view sv(k_metal_navatala_ml_reduction_norm2_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

