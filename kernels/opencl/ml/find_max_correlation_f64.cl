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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_ml_find_max_correlation_f64(__global const double* correlations, __global const uint* activeMask, __global const uint* nCols, __global uint* maxIndex, __global double* maxValue) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double svals[256];
  __local uint sidxs[256];
  double otherVal = as_double(0x0000000000000000ul);
  uint otherIdx = (uint)(0u);
  double myVal = as_double(0x0000000000000000ul);
  uint myIdx = (uint)(0u);
  bool takeOther = false;
  uint d = nCols[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    uint mask = activeMask[gid];
    bool isActive = (mask == (uint)(1u));
    double corr = correlations[gid];
    double absCorr = fabs(corr);
    double val = ((isActive) ? (absCorr) : (as_double(0xffeffffffffffffful)));
    svals[lid] = val;
    sidxs[lid] = gid;
  } else {
    svals[lid] = as_double(0xffeffffffffffffful);
    sidxs[lid] = (uint)(0u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint maxCorrF64RedStride = (uint)(128u);
  for (int maxCorrF64RedStep = 0; maxCorrF64RedStep < (int)(8); ++maxCorrF64RedStep) {
    uint maxCorrF64Stride = maxCorrF64RedStride;
    if (lid < maxCorrF64Stride) {
      otherVal = svals[(lid + maxCorrF64Stride)];
      otherIdx = sidxs[(lid + maxCorrF64Stride)];
      myVal = svals[lid];
      myIdx = sidxs[lid];
      takeOther = (otherVal > myVal);
      if (takeOther) {
        svals[lid] = otherVal;
        sidxs[lid] = otherIdx;
      }
    }
    uint maxCorrF64StrideToHalve = maxCorrF64RedStride;
    uint maxCorrF64NextStride = (maxCorrF64StrideToHalve >> (uint)(1u));
    maxCorrF64RedStride = maxCorrF64NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    maxValue[0] = svals[0];
    maxIndex[0] = sidxs[0];
  }
}
