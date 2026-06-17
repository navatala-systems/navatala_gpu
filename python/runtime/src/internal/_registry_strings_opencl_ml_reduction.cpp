// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `opencl_ml_reduction` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_opencl_navatala_ml_reduction_sum_f32 = R"kernel(
__kernel void navatala_ml_reduction_sum_f32(__global const float* _input, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = raw;
      gsAcc = (gsAcc + v);
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_opencl_navatala_ml_reduction_prod_f32 = R"kernel(
__kernel void navatala_ml_reduction_prod_f32(__global const float* _input, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0x3f800000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = raw;
      gsAcc = (gsAcc * v);
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine * other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_opencl_navatala_ml_reduction_min_f32 = R"kernel(
__kernel void navatala_ml_reduction_min_f32(__global const float* _input, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0x7f7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = raw;
      gsAcc = (((gsAcc < v)) ? (gsAcc) : (v));
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine < other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_opencl_navatala_ml_reduction_max_f32 = R"kernel(
__kernel void navatala_ml_reduction_max_f32(__global const float* _input, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0xff7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = raw;
      gsAcc = (((gsAcc > v)) ? (gsAcc) : (v));
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_opencl_navatala_ml_reduction_amax_f32 = R"kernel(
__kernel void navatala_ml_reduction_amax_f32(__global const float* _input, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0xff7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = fabs(raw);
      gsAcc = (((gsAcc > v)) ? (gsAcc) : (v));
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_opencl_navatala_ml_reduction_norm1_f32 = R"kernel(
__kernel void navatala_ml_reduction_norm1_f32(__global const float* _input, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = fabs(raw);
      gsAcc = (gsAcc + v);
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_opencl_navatala_ml_reduction_avg_f32 = R"kernel(
__kernel void navatala_ml_reduction_avg_f32(__global const float* _input, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = raw;
      gsAcc = (gsAcc + v);
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = (reduced / nF);
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_opencl_navatala_ml_reduction_norm2_f32 = R"kernel(
__kernel void navatala_ml_reduction_norm2_f32(__global const float* _input, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = (raw * raw);
      gsAcc = (gsAcc + v);
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = sqrt(reduced);
    float outV = finalF;
    result[0] = outV;
  }
}

)kernel";
const char* k_opencl_navatala_ml_reduction_sum_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_reduction_sum_f16(__global const half* _input, __global const uint* count, __global half* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (gsAcc + v);
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_opencl_navatala_ml_reduction_prod_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_reduction_prod_f16(__global const half* _input, __global const uint* count, __global half* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0x3f800000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (gsAcc * v);
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine * other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_opencl_navatala_ml_reduction_min_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_reduction_min_f16(__global const half* _input, __global const uint* count, __global half* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0x7f7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (((gsAcc < v)) ? (gsAcc) : (v));
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine < other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_opencl_navatala_ml_reduction_max_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_reduction_max_f16(__global const half* _input, __global const uint* count, __global half* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0xff7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (((gsAcc > v)) ? (gsAcc) : (v));
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_opencl_navatala_ml_reduction_amax_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_reduction_amax_f16(__global const half* _input, __global const uint* count, __global half* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0xff7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = fabs(((float)(raw)));
      gsAcc = (((gsAcc > v)) ? (gsAcc) : (v));
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_opencl_navatala_ml_reduction_norm1_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_reduction_norm1_f16(__global const half* _input, __global const uint* count, __global half* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = fabs(((float)(raw)));
      gsAcc = (gsAcc + v);
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_opencl_navatala_ml_reduction_avg_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_reduction_avg_f16(__global const half* _input, __global const uint* count, __global half* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (gsAcc + v);
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = (reduced / nF);
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}

)kernel";
const char* k_opencl_navatala_ml_reduction_norm2_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_reduction_norm2_f16(__global const half* _input, __global const uint* count, __global half* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = (((float)(raw)) * ((float)(raw)));
      gsAcc = (gsAcc + v);
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    redStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = sqrt(reduced);
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_reduction_sum_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_reduction_sum_f32 = {
  1,
  "navatala_ml_reduction_sum_f32",
  "opencl",
  "navatala_ml_reduction_sum_f32",
  "kernel:opencl:navatala_ml_reduction_sum_f32",
  "abi-r1:opencl:navatala_ml_reduction_sum_f32",
  "abi-r1:opencl:navatala_ml_reduction_sum_f32",
  3,
  kAbiArgs_opencl_navatala_ml_reduction_sum_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_reduction_prod_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_reduction_prod_f32 = {
  1,
  "navatala_ml_reduction_prod_f32",
  "opencl",
  "navatala_ml_reduction_prod_f32",
  "kernel:opencl:navatala_ml_reduction_prod_f32",
  "abi-r1:opencl:navatala_ml_reduction_prod_f32",
  "abi-r1:opencl:navatala_ml_reduction_prod_f32",
  3,
  kAbiArgs_opencl_navatala_ml_reduction_prod_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_reduction_min_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_reduction_min_f32 = {
  1,
  "navatala_ml_reduction_min_f32",
  "opencl",
  "navatala_ml_reduction_min_f32",
  "kernel:opencl:navatala_ml_reduction_min_f32",
  "abi-r1:opencl:navatala_ml_reduction_min_f32",
  "abi-r1:opencl:navatala_ml_reduction_min_f32",
  3,
  kAbiArgs_opencl_navatala_ml_reduction_min_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_reduction_max_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_reduction_max_f32 = {
  1,
  "navatala_ml_reduction_max_f32",
  "opencl",
  "navatala_ml_reduction_max_f32",
  "kernel:opencl:navatala_ml_reduction_max_f32",
  "abi-r1:opencl:navatala_ml_reduction_max_f32",
  "abi-r1:opencl:navatala_ml_reduction_max_f32",
  3,
  kAbiArgs_opencl_navatala_ml_reduction_max_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_reduction_amax_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_reduction_amax_f32 = {
  1,
  "navatala_ml_reduction_amax_f32",
  "opencl",
  "navatala_ml_reduction_amax_f32",
  "kernel:opencl:navatala_ml_reduction_amax_f32",
  "abi-r1:opencl:navatala_ml_reduction_amax_f32",
  "abi-r1:opencl:navatala_ml_reduction_amax_f32",
  3,
  kAbiArgs_opencl_navatala_ml_reduction_amax_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_reduction_norm1_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_reduction_norm1_f32 = {
  1,
  "navatala_ml_reduction_norm1_f32",
  "opencl",
  "navatala_ml_reduction_norm1_f32",
  "kernel:opencl:navatala_ml_reduction_norm1_f32",
  "abi-r1:opencl:navatala_ml_reduction_norm1_f32",
  "abi-r1:opencl:navatala_ml_reduction_norm1_f32",
  3,
  kAbiArgs_opencl_navatala_ml_reduction_norm1_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_reduction_avg_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_reduction_avg_f32 = {
  1,
  "navatala_ml_reduction_avg_f32",
  "opencl",
  "navatala_ml_reduction_avg_f32",
  "kernel:opencl:navatala_ml_reduction_avg_f32",
  "abi-r1:opencl:navatala_ml_reduction_avg_f32",
  "abi-r1:opencl:navatala_ml_reduction_avg_f32",
  3,
  kAbiArgs_opencl_navatala_ml_reduction_avg_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_reduction_norm2_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_reduction_norm2_f32 = {
  1,
  "navatala_ml_reduction_norm2_f32",
  "opencl",
  "navatala_ml_reduction_norm2_f32",
  "kernel:opencl:navatala_ml_reduction_norm2_f32",
  "abi-r1:opencl:navatala_ml_reduction_norm2_f32",
  "abi-r1:opencl:navatala_ml_reduction_norm2_f32",
  3,
  kAbiArgs_opencl_navatala_ml_reduction_norm2_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_reduction_sum_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_reduction_sum_f16 = {
  1,
  "navatala_ml_reduction_sum_f16",
  "opencl",
  "navatala_ml_reduction_sum_f16",
  "kernel:opencl:navatala_ml_reduction_sum_f16",
  "abi-r1:opencl:navatala_ml_reduction_sum_f16",
  "abi-r1:opencl:navatala_ml_reduction_sum_f16",
  3,
  kAbiArgs_opencl_navatala_ml_reduction_sum_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_reduction_prod_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_reduction_prod_f16 = {
  1,
  "navatala_ml_reduction_prod_f16",
  "opencl",
  "navatala_ml_reduction_prod_f16",
  "kernel:opencl:navatala_ml_reduction_prod_f16",
  "abi-r1:opencl:navatala_ml_reduction_prod_f16",
  "abi-r1:opencl:navatala_ml_reduction_prod_f16",
  3,
  kAbiArgs_opencl_navatala_ml_reduction_prod_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_reduction_min_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_reduction_min_f16 = {
  1,
  "navatala_ml_reduction_min_f16",
  "opencl",
  "navatala_ml_reduction_min_f16",
  "kernel:opencl:navatala_ml_reduction_min_f16",
  "abi-r1:opencl:navatala_ml_reduction_min_f16",
  "abi-r1:opencl:navatala_ml_reduction_min_f16",
  3,
  kAbiArgs_opencl_navatala_ml_reduction_min_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_reduction_max_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_reduction_max_f16 = {
  1,
  "navatala_ml_reduction_max_f16",
  "opencl",
  "navatala_ml_reduction_max_f16",
  "kernel:opencl:navatala_ml_reduction_max_f16",
  "abi-r1:opencl:navatala_ml_reduction_max_f16",
  "abi-r1:opencl:navatala_ml_reduction_max_f16",
  3,
  kAbiArgs_opencl_navatala_ml_reduction_max_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_reduction_amax_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_reduction_amax_f16 = {
  1,
  "navatala_ml_reduction_amax_f16",
  "opencl",
  "navatala_ml_reduction_amax_f16",
  "kernel:opencl:navatala_ml_reduction_amax_f16",
  "abi-r1:opencl:navatala_ml_reduction_amax_f16",
  "abi-r1:opencl:navatala_ml_reduction_amax_f16",
  3,
  kAbiArgs_opencl_navatala_ml_reduction_amax_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_reduction_norm1_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_reduction_norm1_f16 = {
  1,
  "navatala_ml_reduction_norm1_f16",
  "opencl",
  "navatala_ml_reduction_norm1_f16",
  "kernel:opencl:navatala_ml_reduction_norm1_f16",
  "abi-r1:opencl:navatala_ml_reduction_norm1_f16",
  "abi-r1:opencl:navatala_ml_reduction_norm1_f16",
  3,
  kAbiArgs_opencl_navatala_ml_reduction_norm1_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_reduction_avg_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_reduction_avg_f16 = {
  1,
  "navatala_ml_reduction_avg_f16",
  "opencl",
  "navatala_ml_reduction_avg_f16",
  "kernel:opencl:navatala_ml_reduction_avg_f16",
  "abi-r1:opencl:navatala_ml_reduction_avg_f16",
  "abi-r1:opencl:navatala_ml_reduction_avg_f16",
  3,
  kAbiArgs_opencl_navatala_ml_reduction_avg_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_ml_reduction_norm2_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 2, 2, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_ml_reduction_norm2_f16 = {
  1,
  "navatala_ml_reduction_norm2_f16",
  "opencl",
  "navatala_ml_reduction_norm2_f16",
  "kernel:opencl:navatala_ml_reduction_norm2_f16",
  "abi-r1:opencl:navatala_ml_reduction_norm2_f16",
  "abi-r1:opencl:navatala_ml_reduction_norm2_f16",
  3,
  kAbiArgs_opencl_navatala_ml_reduction_norm2_f16
};

bool tryGetKernelAbiManifest_opencl_ml_reduction(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_sum_f32") {
    out = &kAbiManifest_opencl_navatala_ml_reduction_sum_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_prod_f32") {
    out = &kAbiManifest_opencl_navatala_ml_reduction_prod_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_min_f32") {
    out = &kAbiManifest_opencl_navatala_ml_reduction_min_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_max_f32") {
    out = &kAbiManifest_opencl_navatala_ml_reduction_max_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_amax_f32") {
    out = &kAbiManifest_opencl_navatala_ml_reduction_amax_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_norm1_f32") {
    out = &kAbiManifest_opencl_navatala_ml_reduction_norm1_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_avg_f32") {
    out = &kAbiManifest_opencl_navatala_ml_reduction_avg_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_norm2_f32") {
    out = &kAbiManifest_opencl_navatala_ml_reduction_norm2_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_sum_f16") {
    out = &kAbiManifest_opencl_navatala_ml_reduction_sum_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_prod_f16") {
    out = &kAbiManifest_opencl_navatala_ml_reduction_prod_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_min_f16") {
    out = &kAbiManifest_opencl_navatala_ml_reduction_min_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_max_f16") {
    out = &kAbiManifest_opencl_navatala_ml_reduction_max_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_amax_f16") {
    out = &kAbiManifest_opencl_navatala_ml_reduction_amax_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_norm1_f16") {
    out = &kAbiManifest_opencl_navatala_ml_reduction_norm1_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_avg_f16") {
    out = &kAbiManifest_opencl_navatala_ml_reduction_avg_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_norm2_f16") {
    out = &kAbiManifest_opencl_navatala_ml_reduction_norm2_f16;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_opencl_ml_reduction(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_reduction_sum_f32";
    std::string_view sv(k_opencl_navatala_ml_reduction_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_prod_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_reduction_prod_f32";
    std::string_view sv(k_opencl_navatala_ml_reduction_prod_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_min_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_reduction_min_f32";
    std::string_view sv(k_opencl_navatala_ml_reduction_min_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_reduction_max_f32";
    std::string_view sv(k_opencl_navatala_ml_reduction_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_amax_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_reduction_amax_f32";
    std::string_view sv(k_opencl_navatala_ml_reduction_amax_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_norm1_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_reduction_norm1_f32";
    std::string_view sv(k_opencl_navatala_ml_reduction_norm1_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_avg_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_reduction_avg_f32";
    std::string_view sv(k_opencl_navatala_ml_reduction_avg_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_norm2_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_reduction_norm2_f32";
    std::string_view sv(k_opencl_navatala_ml_reduction_norm2_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_sum_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_reduction_sum_f16";
    std::string_view sv(k_opencl_navatala_ml_reduction_sum_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_prod_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_reduction_prod_f16";
    std::string_view sv(k_opencl_navatala_ml_reduction_prod_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_min_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_reduction_min_f16";
    std::string_view sv(k_opencl_navatala_ml_reduction_min_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_max_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_reduction_max_f16";
    std::string_view sv(k_opencl_navatala_ml_reduction_max_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_amax_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_reduction_amax_f16";
    std::string_view sv(k_opencl_navatala_ml_reduction_amax_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_norm1_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_reduction_norm1_f16";
    std::string_view sv(k_opencl_navatala_ml_reduction_norm1_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_avg_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_reduction_avg_f16";
    std::string_view sv(k_opencl_navatala_ml_reduction_avg_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_ml_reduction_norm2_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_ml_reduction_norm2_f16";
    std::string_view sv(k_opencl_navatala_ml_reduction_norm2_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

