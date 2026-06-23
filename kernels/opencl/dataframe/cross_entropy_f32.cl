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

__kernel void navatala_dataframe_cross_entropy_f32(__global const float* p, __global const float* q, __global const uint* n, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint nVal = n[(uint)(0u)];
  bool inBounds = (gid < nVal);
  float pVal = p[gid];
  float qVal = q[gid];
  bool pPositive = (pVal > as_float(0x00000000u));
  float logQ = log(qVal);
  float pLogQ = (pVal * logQ);
  float contrib = ((pPositive) ? ((as_float(0xbf800000u) * pLogQ)) : (as_float(0x00000000u)));
  if (inBounds) {
    sdata[lid] = contrib;
  } else {
    sdata[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint ce32_reductionStride = (uint)(128u);
  for (int ce32_reductionStep = 0; ce32_reductionStep < (int)(8); ++ce32_reductionStep) {
    uint ce32_stride = ce32_reductionStride;
    if (lid < ce32_stride) {
      float ce32_other = sdata[(lid + ce32_stride)];
      float ce32_mine = sdata[lid];
      float ce32_sum = (ce32_mine + ce32_other);
      sdata[lid] = ce32_sum;
    }
    uint ce32_strideToHalve = ce32_reductionStride;
    uint ce32_nextStride = (ce32_strideToHalve >> (uint)(1u));
    ce32_reductionStride = ce32_nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    result[(uint)(0u)] = sdata[(uint)(0u)];
  }
}
