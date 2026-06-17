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
extern "C" __global__ void navatala_dataframe_str_is_space(const int* offsets, const signed char* chars, const unsigned int* validity, const unsigned int* count, unsigned int* result, unsigned int* outputValid) {
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
    bool hasChars = (length > 0);
    unsigned int firstIdx = ((unsigned int)(startOffset));
    signed char firstChar = ((hasChars) ? (chars[firstIdx]) : (0));
    int firstInt = ((int)(firstChar));
    bool isSpace = (firstInt == 32);
    bool isTab = (firstInt == 9);
    bool isNewline = (firstInt == 10);
    bool isCR = (firstInt == 13);
    bool isWhitespace = (isSpace || (isTab || (isNewline || isCR)));
    unsigned int _out = (((isValid && (hasChars && isWhitespace))) ? (1u) : (0u));
    result[gid] = _out;
    unsigned int outBit = (validBit << bitIdx);
    atomicOr(&outputValid[wordIdx], outBit);
  }
}
