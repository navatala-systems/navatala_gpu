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
extern "C" __global__ void navatala_dataframe_str_strip_trailing_count(const int* offsets, const signed char* chars, const unsigned int* validity, const unsigned int* count, int* trailingSpaces) {
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
    int startOffset = offsets[gid];
    unsigned int nextIdx = (gid + 1u);
    int endOffset = offsets[nextIdx];
    int length = (endOffset - startOffset);
    bool hasLen1 = (length >= 1);
    bool hasLen2 = (length >= 2);
    bool hasLen3 = (length >= 3);
    bool hasLen4 = (length >= 4);
    unsigned int idx0 = ((unsigned int)((endOffset - 1)));
    unsigned int idx1 = ((unsigned int)((endOffset - 2)));
    unsigned int idx2 = ((unsigned int)((endOffset - 3)));
    unsigned int idx3 = ((unsigned int)((endOffset - 4)));
    signed char c0 = ((hasLen1) ? (chars[idx0]) : (0));
    signed char c1 = ((hasLen2) ? (chars[idx1]) : (0));
    signed char c2 = ((hasLen3) ? (chars[idx2]) : (0));
    signed char c3 = ((hasLen4) ? (chars[idx3]) : (0));
    int c0Int = ((int)(c0));
    int c1Int = ((int)(c1));
    int c2Int = ((int)(c2));
    int c3Int = ((int)(c3));
    bool isWs0 = (((c0Int == 32) || (c0Int == 9)) || ((c0Int == 10) || (c0Int == 13)));
    bool isWs1 = (((c1Int == 32) || (c1Int == 9)) || ((c1Int == 10) || (c1Int == 13)));
    bool isWs2 = (((c2Int == 32) || (c2Int == 9)) || ((c2Int == 10) || (c2Int == 13)));
    bool isWs3 = (((c3Int == 32) || (c3Int == 9)) || ((c3Int == 10) || (c3Int == 13)));
    int trail0 = (((hasLen1 && isWs0)) ? (1) : (0));
    int trail1 = ((((hasLen2 && isWs0) && isWs1)) ? (1) : (0));
    int trail2 = (((((hasLen3 && isWs0) && isWs1) && isWs2)) ? (1) : (0));
    int trail3 = ((((((hasLen4 && isWs0) && isWs1) && isWs2) && isWs3)) ? (1) : (0));
    int trailCount = (trail0 + (trail1 + (trail2 + trail3)));
    int result = ((isValid) ? (trailCount) : (0));
    trailingSpaces[gid] = result;
  }
}
