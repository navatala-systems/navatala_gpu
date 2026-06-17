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

kernel void navatala_ml_compute_xt_y_f32(device const float* X [[buffer(0)]], device const float* y [[buffer(1)]], device const uint* nSamples [[buffer(2)]], device const uint* nFeatures [[buffer(3)]], device float* Xty [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint wgid = ((uint)(int(__tgid.x)));
  threadgroup float sdata[256];
  uint n = nSamples[0];
  uint p = nFeatures[0];
  uint featureIdx = wgid;
  bool featureInBounds = (featureIdx < p);
  if (featureInBounds) {
    uint sampleIdx = lid;
    bool sampleInBounds = (sampleIdx < n);
    uint xIdx = ((sampleIdx * p) + featureIdx);
    if (sampleInBounds) {
      float xVal = X[xIdx];
      float yVal = y[sampleIdx];
      float product = (xVal * yVal);
      sdata[lid] = product;
    } else {
      sdata[lid] = as_type<float>(0x00000000u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint xtyF32Stride = 128u;
    for (int reductionStep1 = 0; reductionStep1 < (int)(8); ++reductionStep1) {
      uint stride1 = xtyF32Stride;
      if ((lid < stride1)) {
        float other = sdata[(lid + stride1)];
        float mine = sdata[lid];
        float sum = (mine + other);
        sdata[lid] = sum;
      }
      uint strideToHalve1 = xtyF32Stride;
      uint nextStride1 = (strideToHalve1 >> 1u);
      xtyF32Stride = nextStride1;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if ((lid == 0u)) {
      float totalSum = sdata[0];
      Xty[featureIdx] = totalSum;
    }
  }
}
