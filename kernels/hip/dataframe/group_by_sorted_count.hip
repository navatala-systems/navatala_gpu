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
extern "C" __global__ void navatala_dataframe_group_by_sorted_count(const int* sortedKeys, const unsigned int* validity, const unsigned int* count, unsigned int* groupCounts, int* groupBounds) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int wordIdx = (gid / 32u);
    unsigned int bitIdx = (gid % 32u);
    unsigned int validWord = validity[wordIdx];
    unsigned int validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    int currentKey = sortedKeys[gid];
    bool isFirst = (gid == 0u);
    unsigned int prevIdx = (gid - 1u);
    unsigned int safePrevIdx = ((isFirst) ? (gid) : (prevIdx));
    int prevKey = sortedKeys[safePrevIdx];
    bool keyChanged = (currentKey != prevKey);
    bool isBoundary = (isFirst || keyChanged);
    unsigned int groupIdx = ((unsigned int)(currentKey));
    if (isValid) {
      atomicAdd(&groupCounts[groupIdx], 1u);
    }
    if (isBoundary) {
      int boundaryIdx = ((int)(gid));
      groupBounds[groupIdx] = boundaryIdx;
    }
  }
}
