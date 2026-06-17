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
__kernel void navatala_dataframe_entropy_f64(__global const double* p, __global const uint* n, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double sdata[256];
  uint nVal = n[(uint)(0u)];
  bool inBounds = (gid < nVal);
  double pVal = p[gid];
  bool pPositive = (pVal > as_double(0x0000000000000000ul));
  double logP = log(pVal);
  double pLogP = (pVal * logP);
  double contrib = ((pPositive) ? ((as_double(0xbff0000000000000ul) * pLogP)) : (as_double(0x0000000000000000ul)));
  if (inBounds) {
    sdata[lid] = contrib;
  } else {
    sdata[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint e64_reductionStride = (uint)(128u);
  for (int e64_reductionStep = 0; e64_reductionStep < (int)(8); ++e64_reductionStep) {
    uint e64_stride = e64_reductionStride;
    if ((lid < e64_stride)) {
      double e64_other = sdata[(lid + e64_stride)];
      double e64_mine = sdata[lid];
      double e64_sum = (e64_mine + e64_other);
      sdata[lid] = e64_sum;
    }
    uint e64_strideToHalve = e64_reductionStride;
    uint e64_nextStride = (e64_strideToHalve >> (uint)(1u));
    e64_reductionStride = e64_nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    result[(uint)(0u)] = sdata[(uint)(0u)];
  }
}
