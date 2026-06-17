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
extern "C" __global__ void navatala_dataframe_str_zfill(const int* offsets, const signed char* chars, const unsigned int* validity, const int* targetWidth, const unsigned int* count, int* zeroCount, int* hasSign) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  int width = targetWidth[0u];
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
    bool hasChars = (length > 0);
    unsigned int firstIdx = ((unsigned int)(startOffset));
    signed char firstChar = ((hasChars) ? (chars[firstIdx]) : (0));
    int firstCharInt = ((int)(firstChar));
    bool isPlus = (firstCharInt == 43);
    bool isMinus = (firstCharInt == 45);
    bool hasSignChar = (isPlus || isMinus);
    int diff = (width - length);
    int zerosNeeded = (((diff > 0)) ? (diff) : (0));
    int resultZeros = ((isValid) ? (zerosNeeded) : (0));
    int resultSign = (((isValid && hasSignChar)) ? (1) : (0));
    zeroCount[gid] = resultZeros;
    hasSign[gid] = resultSign;
  }
}
