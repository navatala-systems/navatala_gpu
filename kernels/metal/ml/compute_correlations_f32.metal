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

kernel void navatala_ml_compute_correlations_f32(device const float* X [[buffer(0)]], device const float* residual [[buffer(1)]], device const uint* nRows [[buffer(2)]], device const uint* nCols [[buffer(3)]], device float* correlations [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nRows[0];
  uint d = nCols[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    uint colOffset = (gid * n);
    float accum = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(n); ++i) {
      uint iU32 = ((uint)(i));
      uint xIdx = (iU32 + colOffset);
      float xVal = X[xIdx];
      float rVal = residual[iU32];
      float prod = (xVal * rVal);
      float currentAccum = accum;
      float newAccum = (currentAccum + prod);
      accum = newAccum;
    }
    float corr = accum;
    correlations[gid] = corr;
  }
}
