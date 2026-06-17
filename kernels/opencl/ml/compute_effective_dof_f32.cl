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

__kernel void navatala_ml_compute_effective_dof_f32(__global const float* eigenvalues, __global const float* lambda, __global const uint* p, __global float* dof) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint pVal = p[0];
  float lambdaVal = lambda[0];
  bool inBounds = (gid < pVal);
  if (inBounds) {
    float eigVal = eigenvalues[gid];
    float denominator = (eigVal + lambdaVal);
    float contribution = (eigVal / denominator);
    sdata[lid] = contribution;
  } else {
    sdata[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint dof32_reductionStride = (uint)(128u);
  for (int dof32_reductionStep = 0; dof32_reductionStep < (int)(8); ++dof32_reductionStep) {
    uint dof32_stride = dof32_reductionStride;
    if ((lid < dof32_stride)) {
      float dof32_other = sdata[(lid + dof32_stride)];
      float dof32_mine = sdata[lid];
      float dof32_sum = (dof32_mine + dof32_other);
      sdata[lid] = dof32_sum;
    }
    uint dof32_strideToHalve = dof32_reductionStride;
    uint dof32_nextStride = (dof32_strideToHalve >> (uint)(1u));
    dof32_reductionStride = dof32_nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float dofVal = sdata[0];
    dof[0] = dofVal;
  }
}
