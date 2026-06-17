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
extern "C" __global__ void navatala_dataframe_str_concat_copy_chars(const signed char* charsA, const signed char* charsB, const int* offsetsA, const int* offsetsB, const int* outputOffsets, const unsigned int* count, signed char* outputChars) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    int srcStartA = offsetsA[gid];
    unsigned int nextIdx = (gid + 1u);
    int srcEndA = offsetsA[nextIdx];
    int lenA = (srcEndA - srcStartA);
    int srcStartB = offsetsB[gid];
    int srcEndB = offsetsB[nextIdx];
    int lenB = (srcEndB - srcStartB);
    int dstStart = outputOffsets[gid];
    bool hasCharA = (lenA > 0);
    unsigned int srcIdxA0 = ((unsigned int)(srcStartA));
    signed char charA0 = charsA[srcIdxA0];
    unsigned int dstIdxA0 = ((unsigned int)(dstStart));
    if (hasCharA) {
      outputChars[dstIdxA0] = charA0;
    }
    bool hasCharA1 = (lenA > 1);
    unsigned int srcIdxA1 = (srcIdxA0 + 1u);
    signed char charA1 = charsA[srcIdxA1];
    unsigned int dstIdxA1 = (dstIdxA0 + 1u);
    if (hasCharA1) {
      outputChars[dstIdxA1] = charA1;
    }
    bool hasCharA2 = (lenA > 2);
    unsigned int srcIdxA2 = (srcIdxA0 + 2u);
    signed char charA2 = charsA[srcIdxA2];
    unsigned int dstIdxA2 = (dstIdxA0 + 2u);
    if (hasCharA2) {
      outputChars[dstIdxA2] = charA2;
    }
    bool hasCharA3 = (lenA > 3);
    unsigned int srcIdxA3 = (srcIdxA0 + 3u);
    signed char charA3 = charsA[srcIdxA3];
    unsigned int dstIdxA3 = (dstIdxA0 + 3u);
    if (hasCharA3) {
      outputChars[dstIdxA3] = charA3;
    }
    int dstStartB = (dstStart + lenA);
    bool hasCharB = (lenB > 0);
    unsigned int srcIdxB0 = ((unsigned int)(srcStartB));
    signed char charB0 = charsB[srcIdxB0];
    unsigned int dstIdxB0 = ((unsigned int)(dstStartB));
    if (hasCharB) {
      outputChars[dstIdxB0] = charB0;
    }
    bool hasCharB1 = (lenB > 1);
    unsigned int srcIdxB1 = (srcIdxB0 + 1u);
    signed char charB1 = charsB[srcIdxB1];
    unsigned int dstIdxB1 = (dstIdxB0 + 1u);
    if (hasCharB1) {
      outputChars[dstIdxB1] = charB1;
    }
    bool hasCharB2 = (lenB > 2);
    unsigned int srcIdxB2 = (srcIdxB0 + 2u);
    signed char charB2 = charsB[srcIdxB2];
    unsigned int dstIdxB2 = (dstIdxB0 + 2u);
    if (hasCharB2) {
      outputChars[dstIdxB2] = charB2;
    }
    bool hasCharB3 = (lenB > 3);
    unsigned int srcIdxB3 = (srcIdxB0 + 3u);
    signed char charB3 = charsB[srcIdxB3];
    unsigned int dstIdxB3 = (dstIdxB0 + 3u);
    if (hasCharB3) {
      outputChars[dstIdxB3] = charB3;
    }
  }
}
