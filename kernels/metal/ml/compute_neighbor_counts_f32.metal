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

kernel void navatala_ml_compute_neighbor_counts_f32(device const float* points [[buffer(0)]], device const float* epsilon [[buffer(1)]], device const uint* nPoints [[buffer(2)]], device const uint* nDims [[buffer(3)]], device uint* counts [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  uint d = nDims[0];
  float eps = epsilon[0];
  float epsSq = (eps * eps);
  bool inBounds = (gid < n);
  if (inBounds) {
    uint iBase = (gid * d);
    uint countAccum = 0u;
    for (int j = 0; j < (int)(n); ++j) {
      uint jU32 = ((uint)(j));
      uint jBase = (jU32 * d);
      float distSqAccum = as_type<float>(0x00000000u);
      for (int k = 0; k < (int)(d); ++k) {
        uint kU32 = ((uint)(k));
        uint iIdx = (iBase + kU32);
        uint jIdx = (jBase + kU32);
        float iVal = points[iIdx];
        float jVal = points[jIdx];
        float diff = (iVal - jVal);
        float diffSq = (diff * diff);
        float currentDistSq = distSqAccum;
        float newDistSq = (currentDistSq + diffSq);
        distSqAccum = newDistSq;
      }
      float finalDistSq = distSqAccum;
      if (finalDistSq < epsSq) {
        uint currentCount = countAccum;
        uint newCount = (currentCount + 1u);
        countAccum = newCount;
      }
    }
    uint finalCount = countAccum;
    counts[gid] = finalCount;
  }
}
