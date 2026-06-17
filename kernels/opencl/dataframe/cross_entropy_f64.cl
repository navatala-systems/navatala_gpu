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
__kernel void navatala_dataframe_cross_entropy_f64(__global const double* p, __global const double* q, __global const uint* n, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double sdata[256];
  uint nVal = n[(uint)(0u)];
  bool inBounds = (gid < nVal);
  double pVal = p[gid];
  double qVal = q[gid];
  bool pPositive = (pVal > as_double(0x0000000000000000ul));
  double logQ = log(qVal);
  double pLogQ = (pVal * logQ);
  double contrib = ((pPositive) ? ((as_double(0xbff0000000000000ul) * pLogQ)) : (as_double(0x0000000000000000ul)));
  if (inBounds) {
    sdata[lid] = contrib;
  } else {
    sdata[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint ce64_reductionStride = (uint)(128u);
  for (int ce64_reductionStep = 0; ce64_reductionStep < (int)(8); ++ce64_reductionStep) {
    uint ce64_stride = ce64_reductionStride;
    if ((lid < ce64_stride)) {
      double ce64_other = sdata[(lid + ce64_stride)];
      double ce64_mine = sdata[lid];
      double ce64_sum = (ce64_mine + ce64_other);
      sdata[lid] = ce64_sum;
    }
    uint ce64_strideToHalve = ce64_reductionStride;
    uint ce64_nextStride = (ce64_strideToHalve >> (uint)(1u));
    ce64_reductionStride = ce64_nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    result[(uint)(0u)] = sdata[(uint)(0u)];
  }
}
