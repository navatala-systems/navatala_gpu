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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_effective_dof_f32(device const float* eigenvalues [[buffer(0)]], device const float* lambda [[buffer(1)]], device const uint* p [[buffer(2)]], device float* dof [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint pVal = p[0];
  float lambdaVal = lambda[0];
  bool inBounds = (gid < pVal);
  if (inBounds) {
    float eigVal = eigenvalues[gid];
    float denominator = (eigVal + lambdaVal);
    float contribution = (eigVal / denominator);
    sdata[lid] = contribution;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint dof32_reductionStride = 128u;
  for (int dof32_reductionStep = 0; dof32_reductionStep < (int)(8); ++dof32_reductionStep) {
    uint dof32_stride = dof32_reductionStride;
    if (lid < dof32_stride) {
      float dof32_other = sdata[(lid + dof32_stride)];
      float dof32_mine = sdata[lid];
      float dof32_sum = (dof32_mine + dof32_other);
      sdata[lid] = dof32_sum;
    }
    uint dof32_strideToHalve = dof32_reductionStride;
    uint dof32_nextStride = (dof32_strideToHalve >> 1u);
    dof32_reductionStride = dof32_nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float dofVal = sdata[0];
    dof[0] = dofVal;
  }
}
