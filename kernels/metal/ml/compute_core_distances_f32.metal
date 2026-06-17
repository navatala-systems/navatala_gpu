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

kernel void navatala_ml_compute_core_distances_f32(device const float* distances [[buffer(0)]], device const uint* n_samples [[buffer(1)]], device const uint* min_samples [[buffer(2)]], device float* core_dists [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint minSamples = min_samples[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint rowBase = (gid * nSamples);
    float kthDist = as_type<float>(0x7149f2cau);
    uint countSmaller = 0u;
    for (int iter = 0; iter < (int)(nSamples); ++iter) {
      uint iterU32 = ((uint)(iter));
      uint distIdx = (rowBase + iterU32);
      float candidateDist = distances[distIdx];
      bool notSelf = (candidateDist > as_type<float>(0x00000000u));
      if (notSelf) {
        countSmaller = 0u;
        for (int j = 0; j < (int)(nSamples); ++j) {
          uint jU32 = ((uint)(j));
          uint jDistIdx = (rowBase + jU32);
          float jDist = distances[jDistIdx];
          bool isPositive = (jDist > as_type<float>(0x00000000u));
          bool isLeCandidate = (jDist <= candidateDist);
          bool shouldCount = (isPositive && isLeCandidate);
          if (shouldCount) {
            uint currCount = countSmaller;
            uint newCount = (currCount + 1u);
            countSmaller = newCount;
          }
        }
        uint finalCount = countSmaller;
        bool isKthOrMore = (finalCount >= minSamples);
        if (isKthOrMore) {
          float currKth = kthDist;
          bool isBetter = (candidateDist < currKth);
          if (isBetter) {
            kthDist = candidateDist;
          }
        }
      }
    }
    float finalCoreDist = kthDist;
    core_dists[gid] = finalCoreDist;
  }
}
