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
extern "C" __global__ void navatala_dataframe_str_strip_both_count(const int* offsets, const signed char* chars, const unsigned int* validity, const unsigned int* count, int* leadingCounts, int* trailingCounts) {
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
    unsigned int idxL0 = ((unsigned int)(startOffset));
    unsigned int idxL1 = (idxL0 + 1u);
    unsigned int idxL2 = (idxL0 + 2u);
    unsigned int idxL3 = (idxL0 + 3u);
    unsigned int idxT0 = ((unsigned int)((endOffset - 1)));
    unsigned int idxT1 = ((unsigned int)((endOffset - 2)));
    unsigned int idxT2 = ((unsigned int)((endOffset - 3)));
    unsigned int idxT3 = ((unsigned int)((endOffset - 4)));
    signed char cL0 = ((hasLen1) ? (chars[idxL0]) : (0));
    signed char cL1 = ((hasLen2) ? (chars[idxL1]) : (0));
    signed char cL2 = ((hasLen3) ? (chars[idxL2]) : (0));
    signed char cL3 = ((hasLen4) ? (chars[idxL3]) : (0));
    signed char cT0 = ((hasLen1) ? (chars[idxT0]) : (0));
    signed char cT1 = ((hasLen2) ? (chars[idxT1]) : (0));
    signed char cT2 = ((hasLen3) ? (chars[idxT2]) : (0));
    signed char cT3 = ((hasLen4) ? (chars[idxT3]) : (0));
    int cL0Int = ((int)(cL0));
    int cL1Int = ((int)(cL1));
    int cL2Int = ((int)(cL2));
    int cL3Int = ((int)(cL3));
    int cT0Int = ((int)(cT0));
    int cT1Int = ((int)(cT1));
    int cT2Int = ((int)(cT2));
    int cT3Int = ((int)(cT3));
    bool isWsL0 = (((cL0Int == 32) || (cL0Int == 9)) || ((cL0Int == 10) || (cL0Int == 13)));
    bool isWsL1 = (((cL1Int == 32) || (cL1Int == 9)) || ((cL1Int == 10) || (cL1Int == 13)));
    bool isWsL2 = (((cL2Int == 32) || (cL2Int == 9)) || ((cL2Int == 10) || (cL2Int == 13)));
    bool isWsL3 = (((cL3Int == 32) || (cL3Int == 9)) || ((cL3Int == 10) || (cL3Int == 13)));
    bool isWsT0 = (((cT0Int == 32) || (cT0Int == 9)) || ((cT0Int == 10) || (cT0Int == 13)));
    bool isWsT1 = (((cT1Int == 32) || (cT1Int == 9)) || ((cT1Int == 10) || (cT1Int == 13)));
    bool isWsT2 = (((cT2Int == 32) || (cT2Int == 9)) || ((cT2Int == 10) || (cT2Int == 13)));
    bool isWsT3 = (((cT3Int == 32) || (cT3Int == 9)) || ((cT3Int == 10) || (cT3Int == 13)));
    int lead0 = (((hasLen1 && isWsL0)) ? (1) : (0));
    int lead1 = ((((hasLen2 && isWsL0) && isWsL1)) ? (1) : (0));
    int lead2 = (((((hasLen3 && isWsL0) && isWsL1) && isWsL2)) ? (1) : (0));
    int lead3 = ((((((hasLen4 && isWsL0) && isWsL1) && isWsL2) && isWsL3)) ? (1) : (0));
    int trail0 = (((hasLen1 && isWsT0)) ? (1) : (0));
    int trail1 = ((((hasLen2 && isWsT0) && isWsT1)) ? (1) : (0));
    int trail2 = (((((hasLen3 && isWsT0) && isWsT1) && isWsT2)) ? (1) : (0));
    int trail3 = ((((((hasLen4 && isWsT0) && isWsT1) && isWsT2) && isWsT3)) ? (1) : (0));
    int leadCount = (lead0 + (lead1 + (lead2 + lead3)));
    int trailCount = (trail0 + (trail1 + (trail2 + trail3)));
    int resultLead = ((isValid) ? (leadCount) : (0));
    int resultTrail = ((isValid) ? (trailCount) : (0));
    leadingCounts[gid] = resultLead;
    trailingCounts[gid] = resultTrail;
  }
}
