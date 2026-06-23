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

kernel void navatala_dataframe_arg_min_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device const uint* tiePolicy [[buffer(2)]], device float* minValue [[buffer(3)]], device uint* minIndex [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float svals[256];
  threadgroup uint sidxs[256];
  float otherVal = as_type<float>(0x00000000u);
  uint otherIdx = 0u;
  float myVal = as_type<float>(0x00000000u);
  uint myIdx = 0u;
  bool otherLess = false;
  bool equalVals = false;
  bool otherIdxSmaller = false;
  bool takeOtherFirst = false;
  bool takeOther = false;
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  uint policy = tiePolicy[0];
  bool useFirst = (policy == 0u);
  float val = _input[gid];
  if (inBounds) {
    svals[lid] = val;
    sidxs[lid] = gid;
  } else {
    svals[lid] = as_type<float>(0x7f7fffffu);
    sidxs[lid] = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint aminf32_reductionStride = 128u;
  for (int aminf32_reductionStep = 0; aminf32_reductionStep < (int)(8); ++aminf32_reductionStep) {
    uint aminf32_stride = aminf32_reductionStride;
    if (lid < aminf32_stride) {
      otherVal = svals[(lid + aminf32_stride)];
      otherIdx = sidxs[(lid + aminf32_stride)];
      myVal = svals[lid];
      myIdx = sidxs[lid];
      otherLess = (otherVal < myVal);
      equalVals = (otherVal == myVal);
      otherIdxSmaller = (otherIdx < myIdx);
      takeOtherFirst = (otherLess || (equalVals && otherIdxSmaller));
      takeOther = ((useFirst) ? (takeOtherFirst) : (otherLess));
      if (takeOther) {
        svals[lid] = otherVal;
        sidxs[lid] = otherIdx;
      }
    }
    uint aminf32_strideToHalve = aminf32_reductionStride;
    uint aminf32_nextStride = (aminf32_strideToHalve >> 1u);
    aminf32_reductionStride = aminf32_nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    minValue[0] = svals[0];
    minIndex[0] = sidxs[0];
  }
}
