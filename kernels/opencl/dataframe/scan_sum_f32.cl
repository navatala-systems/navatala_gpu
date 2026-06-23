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

__kernel void navatala_dataframe_scan_sum_f32(__global const float* _input, __global const uint* count, __global const uint* scanMode, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[512];
  uint n = count[(uint)(0u)];
  uint mode = scanMode[(uint)(0u)];
  bool isExclusive = (mode == (uint)(1u));
  uint ai = lid;
  uint bi = (lid + (uint)(256u));
  bool inBoundsA = (ai < n);
  bool inBoundsB = (bi < n);
  float valA = ((inBoundsA) ? (_input[ai]) : (as_float(0x00000000u)));
  float valB = ((inBoundsB) ? (_input[bi]) : (as_float(0x00000000u)));
  sdata[ai] = valA;
  sdata[bi] = valB;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int offset = 0; offset < (int)(1); ++offset) {
    uint offsetU32 = ((uint)(offset));
    uint upMask = ((offsetU32 * (uint)(2u)) - (uint)(1u));
    bool upShouldWork = ((lid & upMask) == (uint)(0u));
    if (upShouldWork) {
      uint upAi2 = ((offsetU32 * ((lid * (uint)(2u)) + (uint)(1u))) + (offsetU32 - (uint)(1u)));
      uint upBi2 = (upAi2 + offsetU32);
      float upVa = sdata[upAi2];
      float upVb = sdata[upBi2];
      sdata[upBi2] = (upVa + upVb);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    float total = sdata[(uint)(511u)];
    sdata[(uint)(511u)] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int offset = 0; offset < (int)(256); ++offset) {
    uint downOffsetU32 = ((uint)(offset));
    uint downMask = ((downOffsetU32 * (uint)(2u)) - (uint)(1u));
    bool downShouldWork = ((lid & downMask) == (uint)(0u));
    if (downShouldWork) {
      uint downAi2 = ((downOffsetU32 * ((lid * (uint)(2u)) + (uint)(1u))) + (downOffsetU32 - (uint)(1u)));
      uint downBi2 = (downAi2 + downOffsetU32);
      float downT = sdata[downAi2];
      float downVb = sdata[downBi2];
      sdata[downAi2] = downVb;
      sdata[downBi2] = (downT + downVb);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (inBoundsA) {
    float exclusiveResultA = sdata[ai];
    float inclusiveResultA = (exclusiveResultA + valA);
    float resultA = ((isExclusive) ? (exclusiveResultA) : (inclusiveResultA));
    _output[ai] = resultA;
  }
  if (inBoundsB) {
    float exclusiveResultB = sdata[bi];
    float inclusiveResultB = (exclusiveResultB + valB);
    float resultB = ((isExclusive) ? (exclusiveResultB) : (inclusiveResultB));
    _output[bi] = resultB;
  }
}
