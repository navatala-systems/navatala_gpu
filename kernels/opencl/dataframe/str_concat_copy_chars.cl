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

__kernel void navatala_dataframe_str_concat_copy_chars(__global const char* charsA, __global const char* charsB, __global const int* offsetsA, __global const int* offsetsB, __global const int* outputOffsets, __global const uint* count, __global char* outputChars) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    int srcStartA = offsetsA[gid];
    uint nextIdx = (gid + (uint)(1u));
    int srcEndA = offsetsA[nextIdx];
    int lenA = (srcEndA - srcStartA);
    int srcStartB = offsetsB[gid];
    int srcEndB = offsetsB[nextIdx];
    int lenB = (srcEndB - srcStartB);
    int dstStart = outputOffsets[gid];
    bool hasCharA = (lenA > 0);
    uint srcIdxA0 = ((uint)(srcStartA));
    char charA0 = charsA[srcIdxA0];
    uint dstIdxA0 = ((uint)(dstStart));
    if (hasCharA) {
      outputChars[dstIdxA0] = charA0;
    }
    bool hasCharA1 = (lenA > 1);
    uint srcIdxA1 = (srcIdxA0 + (uint)(1u));
    char charA1 = charsA[srcIdxA1];
    uint dstIdxA1 = (dstIdxA0 + (uint)(1u));
    if (hasCharA1) {
      outputChars[dstIdxA1] = charA1;
    }
    bool hasCharA2 = (lenA > 2);
    uint srcIdxA2 = (srcIdxA0 + (uint)(2u));
    char charA2 = charsA[srcIdxA2];
    uint dstIdxA2 = (dstIdxA0 + (uint)(2u));
    if (hasCharA2) {
      outputChars[dstIdxA2] = charA2;
    }
    bool hasCharA3 = (lenA > 3);
    uint srcIdxA3 = (srcIdxA0 + (uint)(3u));
    char charA3 = charsA[srcIdxA3];
    uint dstIdxA3 = (dstIdxA0 + (uint)(3u));
    if (hasCharA3) {
      outputChars[dstIdxA3] = charA3;
    }
    int dstStartB = (dstStart + lenA);
    bool hasCharB = (lenB > 0);
    uint srcIdxB0 = ((uint)(srcStartB));
    char charB0 = charsB[srcIdxB0];
    uint dstIdxB0 = ((uint)(dstStartB));
    if (hasCharB) {
      outputChars[dstIdxB0] = charB0;
    }
    bool hasCharB1 = (lenB > 1);
    uint srcIdxB1 = (srcIdxB0 + (uint)(1u));
    char charB1 = charsB[srcIdxB1];
    uint dstIdxB1 = (dstIdxB0 + (uint)(1u));
    if (hasCharB1) {
      outputChars[dstIdxB1] = charB1;
    }
    bool hasCharB2 = (lenB > 2);
    uint srcIdxB2 = (srcIdxB0 + (uint)(2u));
    char charB2 = charsB[srcIdxB2];
    uint dstIdxB2 = (dstIdxB0 + (uint)(2u));
    if (hasCharB2) {
      outputChars[dstIdxB2] = charB2;
    }
    bool hasCharB3 = (lenB > 3);
    uint srcIdxB3 = (srcIdxB0 + (uint)(3u));
    char charB3 = charsB[srcIdxB3];
    uint dstIdxB3 = (dstIdxB0 + (uint)(3u));
    if (hasCharB3) {
      outputChars[dstIdxB3] = charB3;
    }
  }
}
