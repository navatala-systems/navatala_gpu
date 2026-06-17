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
extern "C" __global__ void navatala_dataframe_str_substring_bounds(const int* offsets, const unsigned int* validity, const int* startIdx, const int* subLen, const unsigned int* count, int* outStart, int* outLen, unsigned int* outputValid) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  int start = startIdx[0u];
  int maxLen = subLen[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int wordIdx = (gid / 32u);
    unsigned int bitIdx = (gid % 32u);
    unsigned int validWord = validity[wordIdx];
    unsigned int validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    int strStart = offsets[gid];
    unsigned int nextIdx = (gid + 1u);
    int strEnd = offsets[nextIdx];
    int strLen = (strEnd - strStart);
    bool isNegative = (start < 0);
    int adjustedStart = ((isNegative) ? ((strLen + start)) : (start));
    int clampedStartLow = (((adjustedStart < 0)) ? (0) : (adjustedStart));
    int clampedStart = (((clampedStartLow > strLen)) ? (strLen) : (clampedStartLow));
    int remaining = (strLen - clampedStart);
    int actualLen = (((maxLen < remaining)) ? (maxLen) : (remaining));
    int actualLenNonNeg = (((actualLen < 0)) ? (0) : (actualLen));
    int newStart = (strStart + clampedStart);
    int resultStart = ((isValid) ? (newStart) : (0));
    int resultLen = ((isValid) ? (actualLenNonNeg) : (0));
    outStart[gid] = resultStart;
    outLen[gid] = resultLen;
    unsigned int outBit = (validBit << bitIdx);
    atomicOr(&outputValid[wordIdx], outBit);
  }
}
