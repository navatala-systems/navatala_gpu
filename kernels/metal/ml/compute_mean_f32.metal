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

kernel void navatala_ml_compute_mean_f32(device const float* y [[buffer(0)]], device const uint* nSamples [[buffer(1)]], device float* mean [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint n = nSamples[0];
  bool inBounds = (gid < n);
  float val = y[gid];
  if (inBounds) {
    sdata[lid] = val;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint meanF32Stride = 128u;
  for (int reductionStep5 = 0; reductionStep5 < (int)(8); ++reductionStep5) {
    uint stride5 = meanF32Stride;
    if (lid < stride5) {
      float other = sdata[(lid + stride5)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    uint strideToHalve5 = meanF32Stride;
    uint nextStride5 = (strideToHalve5 >> 1u);
    meanF32Stride = nextStride5;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float totalSum = sdata[0];
    float nFloat = ((float)(n));
    float meanVal = (totalSum / nFloat);
    mean[0] = meanVal;
  }
}
