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
extern "C" __global__ void navatala_dataframe_sort_merge_join_probe(const int* leftKeys, const unsigned int* leftValidity, const unsigned int* leftCount, const int* rightKeys, const unsigned int* rightValidity, const unsigned int* rightCount, int* leftMatches, int* rightMatches) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nLeft = leftCount[0u];
  unsigned int nRight = rightCount[0u];
  bool inBounds = (gid < nLeft);
  if (inBounds) {
    unsigned int wordIdx = (gid / 32u);
    unsigned int bitIdx = (gid % 32u);
    unsigned int validWord = leftValidity[wordIdx];
    unsigned int validBit = ((validWord >> bitIdx) & 1u);
    bool isLeftValid = (validBit == 1u);
    int leftKey = leftKeys[gid];
    unsigned int lo0 = 0u;
    unsigned int hi0 = nRight;
    unsigned int mid1 = ((lo0 + hi0) / 2u);
    int midKey1 = (((mid1 < nRight)) ? (rightKeys[mid1]) : (0));
    bool goLeft1 = (leftKey < midKey1);
    unsigned int lo1 = ((goLeft1) ? (lo0) : ((mid1 + 1u)));
    unsigned int hi1 = ((goLeft1) ? (mid1) : (hi0));
    unsigned int mid2 = ((lo1 + hi1) / 2u);
    int midKey2 = (((mid2 < nRight)) ? (rightKeys[mid2]) : (0));
    bool goLeft2 = (leftKey < midKey2);
    unsigned int lo2 = ((goLeft2) ? (lo1) : ((mid2 + 1u)));
    unsigned int hi2 = ((goLeft2) ? (mid2) : (hi1));
    unsigned int mid3 = ((lo2 + hi2) / 2u);
    int midKey3 = (((mid3 < nRight)) ? (rightKeys[mid3]) : (0));
    bool goLeft3 = (leftKey < midKey3);
    unsigned int lo3 = ((goLeft3) ? (lo2) : ((mid3 + 1u)));
    unsigned int hi3 = ((goLeft3) ? (mid3) : (hi2));
    unsigned int mid4 = ((lo3 + hi3) / 2u);
    int midKey4 = (((mid4 < nRight)) ? (rightKeys[mid4]) : (0));
    bool goLeft4 = (leftKey < midKey4);
    unsigned int lo4 = ((goLeft4) ? (lo3) : ((mid4 + 1u)));
    unsigned int hi4 = ((goLeft4) ? (mid4) : (hi3));
    unsigned int mid5 = ((lo4 + hi4) / 2u);
    int midKey5 = (((mid5 < nRight)) ? (rightKeys[mid5]) : (0));
    bool goLeft5 = (leftKey < midKey5);
    unsigned int lo5 = ((goLeft5) ? (lo4) : ((mid5 + 1u)));
    unsigned int hi5 = ((goLeft5) ? (mid5) : (hi4));
    unsigned int mid6 = ((lo5 + hi5) / 2u);
    int midKey6 = (((mid6 < nRight)) ? (rightKeys[mid6]) : (0));
    bool goLeft6 = (leftKey < midKey6);
    unsigned int lo6 = ((goLeft6) ? (lo5) : ((mid6 + 1u)));
    unsigned int hi6 = ((goLeft6) ? (mid6) : (hi5));
    unsigned int mid7 = ((lo6 + hi6) / 2u);
    int midKey7 = (((mid7 < nRight)) ? (rightKeys[mid7]) : (0));
    bool goLeft7 = (leftKey < midKey7);
    unsigned int lo7 = ((goLeft7) ? (lo6) : ((mid7 + 1u)));
    unsigned int hi7 = ((goLeft7) ? (mid7) : (hi6));
    unsigned int mid8 = ((lo7 + hi7) / 2u);
    int midKey8 = (((mid8 < nRight)) ? (rightKeys[mid8]) : (0));
    bool goLeft8 = (leftKey < midKey8);
    unsigned int lo8 = ((goLeft8) ? (lo7) : ((mid8 + 1u)));
    unsigned int hi8 = ((goLeft8) ? (mid8) : (hi7));
    unsigned int foundIdx = lo8;
    bool foundInRange = (foundIdx < nRight);
    int foundKey = ((foundInRange) ? (rightKeys[foundIdx]) : (0));
    bool isMatch = (foundInRange && (leftKey == foundKey));
    unsigned int rightWordIdx = (foundIdx / 32u);
    unsigned int rightBitIdx = (foundIdx % 32u);
    unsigned int rightValidWord = ((foundInRange) ? (rightValidity[rightWordIdx]) : (0u));
    unsigned int rightValidBit = ((rightValidWord >> rightBitIdx) & 1u);
    bool isRightValid = (rightValidBit == 1u);
    bool hasValidMatch = (isLeftValid && (isRightValid && isMatch));
    int leftOut = ((int)(gid));
    leftMatches[gid] = leftOut;
    int rightOut = ((hasValidMatch) ? (((int)(foundIdx))) : (-1));
    rightMatches[gid] = rightOut;
  }
}
