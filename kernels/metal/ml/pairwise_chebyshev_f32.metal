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

kernel void navatala_ml_pairwise_chebyshev_f32(device const float* X [[buffer(0)]], device const float* Y [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* m [[buffer(3)]], device const uint* d [[buffer(4)]], device float* D [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint mVal = m[0];
  uint dVal = d[0];
  uint totalElems = (nVal * mVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / mVal);
    uint j = (gid % mVal);
    uint xBase = (i * dVal);
    uint yBase = (j * dVal);
    float maxDiff = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(dVal); ++k) {
      uint kU32 = ((uint)(k));
      uint xIdx = (xBase + kU32);
      uint yIdx = (yBase + kU32);
      float xVal = X[xIdx];
      float yVal = Y[yIdx];
      float diff = (xVal - yVal);
      float absDiff = abs(diff);
      float currentMax = maxDiff;
      bool isGreater = (absDiff > currentMax);
      float newMax = ((isGreater) ? (absDiff) : (currentMax));
      maxDiff = newMax;
    }
    float finalMax = maxDiff;
    D[gid] = finalMax;
  }
}
