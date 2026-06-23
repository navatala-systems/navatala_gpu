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

__kernel void navatala_dataframe_arg_min_f32(__global const float* _input, __global const uint* count, __global const uint* tiePolicy, __global float* minValue, __global uint* minIndex) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float svals[256];
  __local uint sidxs[256];
  float otherVal = as_float(0x00000000u);
  uint otherIdx = (uint)(0u);
  float myVal = as_float(0x00000000u);
  uint myIdx = (uint)(0u);
  bool otherLess = false;
  bool equalVals = false;
  bool otherIdxSmaller = false;
  bool takeOtherFirst = false;
  bool takeOther = false;
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  uint policy = tiePolicy[0];
  bool useFirst = (policy == (uint)(0u));
  float val = _input[gid];
  if (inBounds) {
    svals[lid] = val;
    sidxs[lid] = gid;
  } else {
    svals[lid] = as_float(0x7f7fffffu);
    sidxs[lid] = (uint)(0u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint aminf32_reductionStride = (uint)(128u);
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
    uint aminf32_nextStride = (aminf32_strideToHalve >> (uint)(1u));
    aminf32_reductionStride = aminf32_nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    minValue[0] = svals[0];
    minIndex[0] = sidxs[0];
  }
}
