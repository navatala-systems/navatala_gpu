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
