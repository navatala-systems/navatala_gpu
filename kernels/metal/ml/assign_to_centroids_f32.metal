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

kernel void navatala_ml_assign_to_centroids_f32(device const float* data [[buffer(0)]], device const float* centroids [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* d [[buffer(4)]], device uint* assignments [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint kVal = k[0];
  uint dVal = d[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint dataBase = (gid * dVal);
    uint bestClusterAccum = 0u;
    float bestDistAccum = as_type<float>(0x7f7fffffu);
    float distSumAccum = as_type<float>(0x00000000u);
    for (int c = 0; c < (int)(kVal); ++c) {
      uint cU32 = ((uint)(c));
      uint centroidBase = (cU32 * dVal);
      distSumAccum = as_type<float>(0x00000000u);
      for (int j = 0; j < (int)(dVal); ++j) {
        uint jU32 = ((uint)(j));
        uint dataIdx = (dataBase + jU32);
        uint centIdx = (centroidBase + jU32);
        float dataVal = data[dataIdx];
        float centVal = centroids[centIdx];
        float diff = (dataVal - centVal);
        float diffSq = (diff * diff);
        float currDistSum = distSumAccum;
        float newDistSum = (currDistSum + diffSq);
        distSumAccum = newDistSum;
      }
      float distSum = distSumAccum;
      float bestDist = bestDistAccum;
      if ((distSum < bestDist)) {
        bestDistAccum = distSum;
        bestClusterAccum = cU32;
      }
    }
    uint bestCluster = bestClusterAccum;
    assignments[gid] = bestCluster;
  }
}
