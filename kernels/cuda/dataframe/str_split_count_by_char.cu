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
extern "C" __global__ void navatala_dataframe_str_split_count_by_char(const int* offsets, const signed char* chars, const unsigned int* validity, const signed char* delimChar, const unsigned int* count, int* splitCounts, unsigned int* outputValid) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  signed char delim = delimChar[0u];
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
    unsigned int idx0 = ((unsigned int)(startOffset));
    unsigned int idx1 = (idx0 + 1u);
    unsigned int idx2 = (idx0 + 2u);
    unsigned int idx3 = (idx0 + 3u);
    signed char c0 = (((length >= 1)) ? (chars[idx0]) : (0));
    signed char c1 = (((length >= 2)) ? (chars[idx1]) : (0));
    signed char c2 = (((length >= 3)) ? (chars[idx2]) : (0));
    signed char c3 = (((length >= 4)) ? (chars[idx3]) : (0));
    int match0 = (((c0 == delim)) ? (1) : (0));
    int match1 = (((c1 == delim)) ? (1) : (0));
    int match2 = (((c2 == delim)) ? (1) : (0));
    int match3 = (((c3 == delim)) ? (1) : (0));
    int delimCount = (match0 + (match1 + (match2 + match3)));
    int splitCount = (delimCount + 1);
    int outCount = ((isValid) ? (splitCount) : (0));
    splitCounts[gid] = outCount;
    unsigned int outBit = (validBit << bitIdx);
    atomicOr(&outputValid[wordIdx], outBit);
  }
}
