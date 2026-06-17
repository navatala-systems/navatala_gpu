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
extern "C" __global__ void navatala_dataframe_probe_hash_table_i32(const int* rightKeys, const unsigned int* rightValidity, const unsigned int* rightHashes, const unsigned int* rightCount, const int* tableKeys, const unsigned int* tableRowIds, const unsigned int* tableSize, int* leftMatches, unsigned int* rightMatches) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = rightCount[0u];
  unsigned int tSize = tableSize[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int wordIdx = (gid / 32u);
    unsigned int bitIdx = (gid % 32u);
    unsigned int validWord = rightValidity[wordIdx];
    unsigned int validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    int rightKey = rightKeys[gid];
    unsigned int startIdx = rightHashes[gid];
    unsigned int probeIdx = startIdx;
    bool foundMatch = false;
    bool done = (!isValid);
    unsigned int matchedRowId = 0u;
    for (int __iter = 0; __iter < 65536; ++__iter) {
      if (!((!done))) break;
      unsigned int curIdx = probeIdx;
      int tableKey = tableKeys[curIdx];
      bool isEmpty = (tableKey == 2147483647);
      bool keyMatch = (rightKey == tableKey);
      if (isEmpty) {
        done = true;
      } else {
        if (keyMatch) {
          unsigned int rowId = tableRowIds[curIdx];
          matchedRowId = rowId;
          foundMatch = true;
          done = true;
        } else {
          unsigned int nextIdx = ((curIdx + 1u) % tSize);
          probeIdx = nextIdx;
        }
      }
    }
    int leftMatch = ((foundMatch) ? (((int)(matchedRowId))) : (-1));
    leftMatches[gid] = leftMatch;
    rightMatches[gid] = gid;
  }
}
