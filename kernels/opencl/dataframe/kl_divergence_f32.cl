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

__kernel void navatala_dataframe_kl_divergence_f32(__global const float* p, __global const float* q, __global const uint* n, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint nVal = n[(uint)(0u)];
  bool inBounds = (gid < nVal);
  float pVal = p[gid];
  float qVal = q[gid];
  bool pPositive = (pVal > as_float(0x00000000u));
  float logP = log(pVal);
  float logQ = log(qVal);
  float logRatio = (logP - logQ);
  float pLogRatio = (pVal * logRatio);
  float contrib = ((pPositive) ? (pLogRatio) : (as_float(0x00000000u)));
  if (inBounds) {
    sdata[lid] = contrib;
  } else {
    sdata[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint kl32_reductionStride = (uint)(128u);
  for (int kl32_reductionStep = 0; kl32_reductionStep < (int)(8); ++kl32_reductionStep) {
    uint kl32_stride = kl32_reductionStride;
    if ((lid < kl32_stride)) {
      float kl32_other = sdata[(lid + kl32_stride)];
      float kl32_mine = sdata[lid];
      float kl32_sum = (kl32_mine + kl32_other);
      sdata[lid] = kl32_sum;
    }
    uint kl32_strideToHalve = kl32_reductionStride;
    uint kl32_nextStride = (kl32_strideToHalve >> (uint)(1u));
    kl32_reductionStride = kl32_nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    result[(uint)(0u)] = sdata[(uint)(0u)];
  }
}
