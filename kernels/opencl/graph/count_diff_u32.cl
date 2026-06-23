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

__kernel void navatala_graph_count_diff_u32(__global const uint* a, __global const uint* b, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsAcc = as_float(0x00000000u);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * (uint)(256u)));
    if (idx < countVal) {
      uint av = a[idx];
      uint bv = b[idx];
      float diff = (((av == bv)) ? (as_float(0x00000000u)) : (as_float(0x3f800000u)));
      gsAcc = (gsAcc + diff);
    }
  }
  sdata[lid] = gsAcc;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint redStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if (lid < stride) {
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
  if (lid == (uint)(0u)) {
    result[0] = sdata[0];
  }
}
