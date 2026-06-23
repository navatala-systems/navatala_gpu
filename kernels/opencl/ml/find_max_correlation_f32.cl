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

__kernel void navatala_ml_find_max_correlation_f32(__global const float* correlations, __global const uint* activeMask, __global const uint* nCols, __global uint* maxIndex, __global float* maxValue) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float svals[256];
  __local uint sidxs[256];
  float otherVal = as_float(0x00000000u);
  uint otherIdx = (uint)(0u);
  float myVal = as_float(0x00000000u);
  uint myIdx = (uint)(0u);
  bool takeOther = false;
  uint d = nCols[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    uint mask = activeMask[gid];
    bool isActive = (mask == (uint)(1u));
    float corr = correlations[gid];
    float absCorr = fabs(corr);
    float val = ((isActive) ? (absCorr) : (as_float(0xff7fffffu)));
    svals[lid] = val;
    sidxs[lid] = gid;
  } else {
    svals[lid] = as_float(0xff7fffffu);
    sidxs[lid] = (uint)(0u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint maxCorrF32RedStride = (uint)(128u);
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
    uint maxCorrF32NextStride = (maxCorrF32StrideToHalve >> (uint)(1u));
    maxCorrF32RedStride = maxCorrF32NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    maxValue[0] = svals[0];
    maxIndex[0] = sidxs[0];
  }
}
