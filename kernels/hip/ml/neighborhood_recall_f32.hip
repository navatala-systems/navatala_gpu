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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_neighborhood_recall_f32(const unsigned int* origNeighbors, const unsigned int* embedNeighbors, const unsigned int* n, const unsigned int* k, float* recall) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0u];
  unsigned int kVal = k[0u];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int rowBase = (gid * kVal);
    unsigned int intersectCountAccum = 0u;
    unsigned int foundAccum = 0u;
    for (int origIdx = 0; origIdx < (int)(kVal); ++origIdx) {
      unsigned int origIdxU32 = ((unsigned int)(origIdx));
      unsigned int origNeighborIdx = (rowBase + origIdxU32);
      unsigned int origNeighbor = origNeighbors[origNeighborIdx];
      foundAccum = 0u;
      for (int embedIdx = 0; embedIdx < (int)(kVal); ++embedIdx) {
        unsigned int embedIdxU32 = ((unsigned int)(embedIdx));
        unsigned int embedNeighborIdx = (rowBase + embedIdxU32);
        unsigned int embedNeighbor = embedNeighbors[embedNeighborIdx];
        bool isMatch = (origNeighbor == embedNeighbor);
        unsigned int currentFound = foundAccum;
        unsigned int matchAsU32 = ((isMatch) ? (1u) : (0u));
        unsigned int newFound = (currentFound | matchAsU32);
        foundAccum = newFound;
      }
      unsigned int finalFound = foundAccum;
      bool wasFound = (finalFound > 0u);
      if (wasFound) {
        unsigned int currentCount = intersectCountAccum;
        unsigned int newCount = (currentCount + 1u);
        intersectCountAccum = newCount;
      }
    }
    unsigned int finalIntersectCount = intersectCountAccum;
    float kFloat = ((float)(kVal));
    float countFloat = ((float)(finalIntersectCount));
    float recallVal = (countFloat / kFloat);
    recall[gid] = recallVal;
  }
}
