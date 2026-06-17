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
extern "C" __global__ void navatala_dataframe_nested_loop_join_probe(const int* leftKeys, const int* rightKeys, const unsigned int* leftCount, const unsigned int* rightCount, const unsigned int* maxMatches, unsigned int* matchCount, int* matchLeft, int* matchRight) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lCount = leftCount[0u];
  unsigned int rCount = rightCount[0u];
  unsigned int maxM = maxMatches[0u];
  bool inBounds = (gid < lCount);
  if (inBounds) {
    int leftKey = leftKeys[gid];
    for (int rightIdx = 0; rightIdx < (int)(rCount); ++rightIdx) {
      int rightKey = rightKeys[rightIdx];
      bool isMatch = (leftKey == rightKey);
      if (isMatch) {
        atomicAdd(&matchCount[0u], 1u);
        unsigned int outPos = matchCount[0u];
        unsigned int outPosAdj = (outPos - 1u);
        bool canStore = (outPosAdj < maxM);
        if (canStore) {
          int leftIdxI32 = ((int)(gid));
          int rightIdxI32 = ((int)(rightIdx));
          matchLeft[outPosAdj] = leftIdxI32;
          matchRight[outPosAdj] = rightIdxI32;
        }
      }
    }
  }
}
