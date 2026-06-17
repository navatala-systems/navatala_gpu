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

kernel void navatala_ml_neighborhood_recall_f32(device const uint* origNeighbors [[buffer(0)]], device const uint* embedNeighbors [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* k [[buffer(3)]], device float* recall [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0u];
  uint kVal = k[0u];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint rowBase = (gid * kVal);
    uint intersectCountAccum = 0u;
    uint foundAccum = 0u;
    for (int origIdx = 0; origIdx < (int)(kVal); ++origIdx) {
      uint origIdxU32 = ((uint)(origIdx));
      uint origNeighborIdx = (rowBase + origIdxU32);
      uint origNeighbor = origNeighbors[origNeighborIdx];
      foundAccum = 0u;
      for (int embedIdx = 0; embedIdx < (int)(kVal); ++embedIdx) {
        uint embedIdxU32 = ((uint)(embedIdx));
        uint embedNeighborIdx = (rowBase + embedIdxU32);
        uint embedNeighbor = embedNeighbors[embedNeighborIdx];
        bool isMatch = (origNeighbor == embedNeighbor);
        uint currentFound = foundAccum;
        uint matchAsU32 = ((isMatch) ? (1u) : (0u));
        uint newFound = (currentFound | matchAsU32);
        foundAccum = newFound;
      }
      uint finalFound = foundAccum;
      bool wasFound = (finalFound > 0u);
      if (wasFound) {
        uint currentCount = intersectCountAccum;
        uint newCount = (currentCount + 1u);
        intersectCountAccum = newCount;
      }
    }
    uint finalIntersectCount = intersectCountAccum;
    float kFloat = ((float)(kVal));
    float countFloat = ((float)(finalIntersectCount));
    float recallVal = (countFloat / kFloat);
    recall[gid] = recallVal;
  }
}
