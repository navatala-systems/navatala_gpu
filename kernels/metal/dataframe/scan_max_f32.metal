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

kernel void navatala_dataframe_scan_max_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device const uint* scanMode [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[512];
  uint n = count[0u];
  uint mode = scanMode[0u];
  bool isExclusive = (mode == 1u);
  uint ai = lid;
  uint bi = (lid + 256u);
  bool inBoundsA = (ai < n);
  bool inBoundsB = (bi < n);
  float valA = ((inBoundsA) ? (_input[ai]) : (as_type<float>(0xff7fffffu)));
  float valB = ((inBoundsB) ? (_input[bi]) : (as_type<float>(0xff7fffffu)));
  sdata[ai] = valA;
  sdata[bi] = valB;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int offset = 0; offset < (int)(1); ++offset) {
    uint offsetU32 = ((uint)(offset));
    uint upMask = ((offsetU32 * 2u) - 1u);
    bool upShouldWork = ((lid & upMask) == 0u);
    if (upShouldWork) {
      uint upAi2 = ((offsetU32 * ((lid * 2u) + 1u)) + (offsetU32 - 1u));
      uint upBi2 = (upAi2 + offsetU32);
      float upVa = sdata[upAi2];
      float upVb = sdata[upBi2];
      float upMaxVal = (((upVa > upVb)) ? (upVa) : (upVb));
      sdata[upBi2] = upMaxVal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    sdata[511u] = as_type<float>(0xff7fffffu);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int offset = 0; offset < (int)(256); ++offset) {
    uint downOffsetU32 = ((uint)(offset));
    uint downMask = ((downOffsetU32 * 2u) - 1u);
    bool downShouldWork = ((lid & downMask) == 0u);
    if (downShouldWork) {
      uint downAi2 = ((downOffsetU32 * ((lid * 2u) + 1u)) + (downOffsetU32 - 1u));
      uint downBi2 = (downAi2 + downOffsetU32);
      float downT = sdata[downAi2];
      float downVb = sdata[downBi2];
      sdata[downAi2] = downVb;
      float downMaxVal = (((downT > downVb)) ? (downT) : (downVb));
      sdata[downBi2] = downMaxVal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (inBoundsA) {
    float exclusiveResultA = sdata[ai];
    float inclusiveResultA = (((exclusiveResultA > valA)) ? (exclusiveResultA) : (valA));
    float resultA = ((isExclusive) ? (exclusiveResultA) : (inclusiveResultA));
    _output[ai] = resultA;
  }
  if (inBoundsB) {
    float exclusiveResultB = sdata[bi];
    float inclusiveResultB = (((exclusiveResultB > valB)) ? (exclusiveResultB) : (valB));
    float resultB = ((isExclusive) ? (exclusiveResultB) : (inclusiveResultB));
    _output[bi] = resultB;
  }
}
