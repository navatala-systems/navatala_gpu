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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_ml_trustworthiness_f64(__global const uint* origNeighbors, __global const uint* embedNeighbors, __global const uint* origRanks, __global const uint* n, __global const uint* k, __global double* penalty) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[(uint)(0u)];
  uint kVal = k[(uint)(0u)];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint kRowBase = (gid * kVal);
    uint nRowBase = (gid * nVal);
    double penaltySumAccum = as_double(0x0000000000000000ul);
    uint inOriginalAccum = (uint)(0u);
    for (int embedIdx = 0; embedIdx < (int)(kVal); ++embedIdx) {
      uint embedIdxU32 = ((uint)(embedIdx));
      uint embedNeighborIdx = (kRowBase + embedIdxU32);
      uint embedNeighbor = embedNeighbors[embedNeighborIdx];
      inOriginalAccum = (uint)(0u);
      for (int origIdx = 0; origIdx < (int)(kVal); ++origIdx) {
        uint origIdxU32 = ((uint)(origIdx));
        uint origNeighborIdx = (kRowBase + origIdxU32);
        uint origNeighbor = origNeighbors[origNeighborIdx];
        bool isMatch = (embedNeighbor == origNeighbor);
        uint currentInOrig = inOriginalAccum;
        uint matchAsU32 = ((isMatch) ? ((uint)(1u)) : ((uint)(0u)));
        uint newInOrig = (currentInOrig | matchAsU32);
        inOriginalAccum = newInOrig;
      }
      uint finalInOrig = inOriginalAccum;
      bool inOriginal = (finalInOrig > (uint)(0u));
      bool isFalseNeighbor = (!inOriginal);
      if (isFalseNeighbor) {
        uint rankIdx = (nRowBase + embedNeighbor);
        uint rank = origRanks[rankIdx];
        bool rankGtK = (rank > kVal);
        if (rankGtK) {
          uint diff = (rank - kVal);
          double diffFloat = ((double)(diff));
          double currentPenalty = penaltySumAccum;
          double newPenalty = (currentPenalty + diffFloat);
          penaltySumAccum = newPenalty;
        }
      }
    }
    double finalPenaltySum = penaltySumAccum;
    penalty[gid] = finalPenaltySum;
  }
}
