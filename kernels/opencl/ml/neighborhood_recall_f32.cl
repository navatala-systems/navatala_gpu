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

__kernel void navatala_ml_neighborhood_recall_f32(__global const uint* origNeighbors, __global const uint* embedNeighbors, __global const uint* n, __global const uint* k, __global float* recall) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[(uint)(0u)];
  uint kVal = k[(uint)(0u)];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint rowBase = (gid * kVal);
    uint intersectCountAccum = (uint)(0u);
    uint foundAccum = (uint)(0u);
    for (int origIdx = 0; origIdx < (int)(kVal); ++origIdx) {
      uint origIdxU32 = ((uint)(origIdx));
      uint origNeighborIdx = (rowBase + origIdxU32);
      uint origNeighbor = origNeighbors[origNeighborIdx];
      foundAccum = (uint)(0u);
      for (int embedIdx = 0; embedIdx < (int)(kVal); ++embedIdx) {
        uint embedIdxU32 = ((uint)(embedIdx));
        uint embedNeighborIdx = (rowBase + embedIdxU32);
        uint embedNeighbor = embedNeighbors[embedNeighborIdx];
        bool isMatch = (origNeighbor == embedNeighbor);
        uint currentFound = foundAccum;
        uint matchAsU32 = ((isMatch) ? ((uint)(1u)) : ((uint)(0u)));
        uint newFound = (currentFound | matchAsU32);
        foundAccum = newFound;
      }
      uint finalFound = foundAccum;
      bool wasFound = (finalFound > (uint)(0u));
      if (wasFound) {
        uint currentCount = intersectCountAccum;
        uint newCount = (currentCount + (uint)(1u));
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
