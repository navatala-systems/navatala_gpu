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
__kernel void navatala_dataframe_arg_min_f64(__global const double* _input, __global const uint* count, __global const uint* tiePolicy, __global double* minValue, __global uint* minIndex) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double svals[256];
  __local uint sidxs[256];
  double otherVal = as_double(0x0000000000000000ul);
  uint otherIdx = (uint)(0u);
  double myVal = as_double(0x0000000000000000ul);
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
  double val = _input[gid];
  if (inBounds) {
    svals[lid] = val;
    sidxs[lid] = gid;
  } else {
    svals[lid] = as_double(0x7feffffffffffffful);
    sidxs[lid] = (uint)(0u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint aminf64_reductionStride = (uint)(128u);
  for (int aminf64_reductionStep = 0; aminf64_reductionStep < (int)(8); ++aminf64_reductionStep) {
    uint aminf64_stride = aminf64_reductionStride;
    if ((lid < aminf64_stride)) {
      otherVal = svals[(lid + aminf64_stride)];
      otherIdx = sidxs[(lid + aminf64_stride)];
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
    uint aminf64_strideToHalve = aminf64_reductionStride;
    uint aminf64_nextStride = (aminf64_strideToHalve >> (uint)(1u));
    aminf64_reductionStride = aminf64_nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    minValue[0] = svals[0];
    minIndex[0] = sidxs[0];
  }
}
