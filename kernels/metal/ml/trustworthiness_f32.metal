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

kernel void navatala_ml_trustworthiness_f32(device const uint* origNeighbors [[buffer(0)]], device const uint* embedNeighbors [[buffer(1)]], device const uint* origRanks [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device float* penalty [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0u];
  uint kVal = k[0u];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint kRowBase = (gid * kVal);
    uint nRowBase = (gid * nVal);
    float penaltySumAccum = as_type<float>(0x00000000u);
    uint inOriginalAccum = 0u;
    for (int embedIdx = 0; embedIdx < (int)(kVal); ++embedIdx) {
      uint embedIdxU32 = ((uint)(embedIdx));
      uint embedNeighborIdx = (kRowBase + embedIdxU32);
      uint embedNeighbor = embedNeighbors[embedNeighborIdx];
      inOriginalAccum = 0u;
      for (int origIdx = 0; origIdx < (int)(kVal); ++origIdx) {
        uint origIdxU32 = ((uint)(origIdx));
        uint origNeighborIdx = (kRowBase + origIdxU32);
        uint origNeighbor = origNeighbors[origNeighborIdx];
        bool isMatch = (embedNeighbor == origNeighbor);
        uint currentInOrig = inOriginalAccum;
        uint matchAsU32 = ((isMatch) ? (1u) : (0u));
        uint newInOrig = (currentInOrig | matchAsU32);
        inOriginalAccum = newInOrig;
      }
      uint finalInOrig = inOriginalAccum;
      bool inOriginal = (finalInOrig > 0u);
      bool isFalseNeighbor = (!inOriginal);
      if (isFalseNeighbor) {
        uint rankIdx = (nRowBase + embedNeighbor);
        uint rank = origRanks[rankIdx];
        bool rankGtK = (rank > kVal);
        if (rankGtK) {
          uint diff = (rank - kVal);
          float diffFloat = ((float)(diff));
          float currentPenalty = penaltySumAccum;
          float newPenalty = (currentPenalty + diffFloat);
          penaltySumAccum = newPenalty;
        }
      }
    }
    float finalPenaltySum = penaltySumAccum;
    penalty[gid] = finalPenaltySum;
  }
}
