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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_ml_continuity_f64(const unsigned int* origNeighbors, const unsigned int* embedNeighbors, const unsigned int* embedRanks, const unsigned int* n, const unsigned int* k, double* penalty) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0u];
  unsigned int kVal = k[0u];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int kRowBase = (gid * kVal);
    unsigned int nRowBase = (gid * nVal);
    double penaltySumAccum = __longlong_as_double(0x0000000000000000ull);
    unsigned int inEmbedAccum = 0u;
    for (int origIdx = 0; origIdx < (int)(kVal); ++origIdx) {
      unsigned int origIdxU32 = ((unsigned int)(origIdx));
      unsigned int origNeighborIdx = (kRowBase + origIdxU32);
      unsigned int origNeighbor = origNeighbors[origNeighborIdx];
      inEmbedAccum = 0u;
      for (int embedIdx = 0; embedIdx < (int)(kVal); ++embedIdx) {
        unsigned int embedIdxU32 = ((unsigned int)(embedIdx));
        unsigned int embedNeighborIdx = (kRowBase + embedIdxU32);
        unsigned int embedNeighbor = embedNeighbors[embedNeighborIdx];
        bool isMatch = (origNeighbor == embedNeighbor);
        unsigned int currentInEmbed = inEmbedAccum;
        unsigned int matchAsU32 = ((isMatch) ? (1u) : (0u));
        unsigned int newInEmbed = (currentInEmbed | matchAsU32);
        inEmbedAccum = newInEmbed;
      }
      unsigned int finalInEmbed = inEmbedAccum;
      bool inEmbed = (finalInEmbed > 0u);
      bool isMissingNeighbor = (!inEmbed);
      if (isMissingNeighbor) {
        unsigned int rankIdx = (nRowBase + origNeighbor);
        unsigned int rank = embedRanks[rankIdx];
        bool rankGtK = (rank > kVal);
        if (rankGtK) {
          unsigned int diff = (rank - kVal);
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
