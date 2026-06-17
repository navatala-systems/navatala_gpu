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
    if ((lid < obj32RedStrideVal)) {
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
  if ((lid == 0u)) {
    float totalMse = sdataMse[0];
    float totalL1 = sdataL1[0];
    float halfMse = (as_type<float>(0x3f000000u) * totalMse);
    float lambdaL1 = (lambdaVal * totalL1);
    float obj = (halfMse + lambdaL1);
    objective[0] = obj;
  }
}
