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

kernel void navatala_ml_compute_k_l_divergence_f32(device const float* P [[buffer(0)]], device const float* Q [[buffer(1)]], device const uint* nPoints [[buffer(2)]], device float* kl_divergence [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  threadgroup float sdata[256];
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint n = nPoints[0];
  uint totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    float pij = P[gid];
    float qij = Q[gid];
    float minVal = as_type<float>(0x33d6bf95u);
    bool pSafe = (pij > minVal);
    bool qSafe = (qij > minVal);
    bool bothSafe = (pSafe && qSafe);
    if (bothSafe) {
      float ratio = (pij / qij);
      float logRatio = log(ratio);
      float klTerm = (pij * logRatio);
      sdata[lid] = klTerm;
    } else {
      sdata[lid] = as_type<float>(0x00000000u);
    }
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint klF32ReductionStride = 128u;
  for (int klF32ReductionStep = 0; klF32ReductionStep < (int)(8); ++klF32ReductionStep) {
    uint klF32Stride = klF32ReductionStride;
    if (lid < klF32Stride) {
      float klF32Other = sdata[(lid + klF32Stride)];
      float klF32Mine = sdata[lid];
      float klF32Sum = (klF32Mine + klF32Other);
      sdata[lid] = klF32Sum;
    }
    uint klF32StrideToHalve = klF32ReductionStride;
    uint klF32NextStride = (klF32StrideToHalve >> 1u);
    klF32ReductionStride = klF32NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float klVal = sdata[0];
    kl_divergence[0] = klVal;
  }
}
