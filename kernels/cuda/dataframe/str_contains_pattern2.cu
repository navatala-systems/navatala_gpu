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
extern "C" __global__ void navatala_dataframe_str_contains_pattern2(const int* offsets, const signed char* chars, const unsigned int* validity, const signed char* pattern1, const signed char* pattern2, const unsigned int* count, unsigned int* found, unsigned int* outputValid) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  signed char p1 = pattern1[0u];
  signed char p2 = pattern2[0u];
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
    bool hasEnough = (length >= 2);
    unsigned int idx0 = ((unsigned int)(startOffset));
    unsigned int idx1 = (idx0 + 1u);
    signed char c0 = ((hasEnough) ? (chars[idx0]) : (0));
    signed char c1 = ((hasEnough) ? (chars[idx1]) : (0));
    bool matchFirst = ((c0 == p1) && (c1 == p2));
    unsigned int result = (((isValid && (hasEnough && matchFirst))) ? (1u) : (0u));
    found[gid] = result;
    unsigned int outBit = (validBit << bitIdx);
    atomicOr(&outputValid[wordIdx], outBit);
  }
}
