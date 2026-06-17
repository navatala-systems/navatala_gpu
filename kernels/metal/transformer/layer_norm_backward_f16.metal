// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
