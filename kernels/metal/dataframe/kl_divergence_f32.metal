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

kernel void navatala_dataframe_kl_divergence_f32(device const float* p [[buffer(0)]], device const float* q [[buffer(1)]], device const uint* n [[buffer(2)]], device float* result [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint nVal = n[0u];
  bool inBounds = (gid < nVal);
  float pVal = p[gid];
  float qVal = q[gid];
  bool pPositive = (pVal > as_type<float>(0x00000000u));
  float logP = log(pVal);
  float logQ = log(qVal);
  float logRatio = (logP - logQ);
  float pLogRatio = (pVal * logRatio);
  float contrib = ((pPositive) ? (pLogRatio) : (as_type<float>(0x00000000u)));
  if (inBounds) {
    sdata[lid] = contrib;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint kl32_reductionStride = 128u;
  for (int kl32_reductionStep = 0; kl32_reductionStep < (int)(8); ++kl32_reductionStep) {
    uint kl32_stride = kl32_reductionStride;
    if ((lid < kl32_stride)) {
      float kl32_other = sdata[(lid + kl32_stride)];
      float kl32_mine = sdata[lid];
      float kl32_sum = (kl32_mine + kl32_other);
      sdata[lid] = kl32_sum;
    }
    uint kl32_strideToHalve = kl32_reductionStride;
    uint kl32_nextStride = (kl32_strideToHalve >> 1u);
    kl32_reductionStride = kl32_nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    result[0u] = sdata[0u];
  }
}
