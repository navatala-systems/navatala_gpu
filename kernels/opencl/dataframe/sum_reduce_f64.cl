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
__kernel void navatala_dataframe_sum_reduce_f64(__global const double* _input, __global const uint* count, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double sdata[256];
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  double val = _input[gid];
  if (inBounds) {
    sdata[lid] = val;
  } else {
    sdata[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint reductionStride = (uint)(128u);
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    uint stride = reductionStride;
    if (lid < stride) {
      uint partnerIdx = (lid + stride);
      double myVal = sdata[lid];
      double partnerVal = sdata[partnerIdx];
      sdata[lid] = (myVal + partnerVal);
    }
    uint strideToHalve = reductionStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    reductionStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  uint zeroU32 = ((uint)(0));
  if (lid == zeroU32) {
    result[0] = sdata[0];
  }
}
