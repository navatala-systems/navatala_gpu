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
__kernel void navatala_dataframe_kl_divergence_f64(__global const double* p, __global const double* q, __global const uint* n, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double sdata[256];
  uint nVal = n[(uint)(0u)];
  bool inBounds = (gid < nVal);
  double pVal = p[gid];
  double qVal = q[gid];
  bool pPositive = (pVal > as_double(0x0000000000000000ul));
  double logP = log(pVal);
  double logQ = log(qVal);
  double logRatio = (logP - logQ);
  double pLogRatio = (pVal * logRatio);
  double contrib = ((pPositive) ? (pLogRatio) : (as_double(0x0000000000000000ul)));
  if (inBounds) {
    sdata[lid] = contrib;
  } else {
    sdata[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint kl64_reductionStride = (uint)(128u);
  for (int kl64_reductionStep = 0; kl64_reductionStep < (int)(8); ++kl64_reductionStep) {
    uint kl64_stride = kl64_reductionStride;
    if (lid < kl64_stride) {
      double kl64_other = sdata[(lid + kl64_stride)];
      double kl64_mine = sdata[lid];
      double kl64_sum = (kl64_mine + kl64_other);
      sdata[lid] = kl64_sum;
    }
    uint kl64_strideToHalve = kl64_reductionStride;
    uint kl64_nextStride = (kl64_strideToHalve >> (uint)(1u));
    kl64_reductionStride = kl64_nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    result[(uint)(0u)] = sdata[(uint)(0u)];
  }
}
