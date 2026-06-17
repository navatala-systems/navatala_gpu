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
    if ((lid < stride3)) {
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
  if ((lid == 0u)) {
    float ssRes = sdataRes[0];
    float ssTot = sdataTot[0];
    float ratio = (ssRes / ssTot);
    float r2 = (as_type<float>(0x3f800000u) - ratio);
    rSquared[0] = r2;
  }
}
