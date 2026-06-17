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
extern "C" __global__ void navatala_dataframe_str_compare(const int* offsetsA, const int* offsetsB, const signed char* charsA, const signed char* charsB, const unsigned int* validityA, const unsigned int* validityB, const unsigned int* count, int* result, unsigned int* outputValid) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int wordIdx = (gid / 32u);
    unsigned int bitIdx = (gid % 32u);
    unsigned int validWordA = validityA[wordIdx];
    unsigned int validBitA = ((validWordA >> bitIdx) & 1u);
    bool isValidA = (validBitA == 1u);
    unsigned int validWordB = validityB[wordIdx];
    unsigned int validBitB = ((validWordB >> bitIdx) & 1u);
    bool isValidB = (validBitB == 1u);
    bool bothValid = (isValidA && isValidB);
    int startA = offsetsA[gid];
    unsigned int nextIdx = (gid + 1u);
    int endA = offsetsA[nextIdx];
    int lenA = (endA - startA);
    int startB = offsetsB[gid];
    int endB = offsetsB[nextIdx];
    int lenB = (endB - startB);
    bool hasCharsA = (lenA > 0);
    bool hasCharsB = (lenB > 0);
    unsigned int idxA = ((unsigned int)(startA));
    unsigned int idxB = ((unsigned int)(startB));
    signed char charA = ((hasCharsA) ? (charsA[idxA]) : (0));
    signed char charB = ((hasCharsB) ? (charsB[idxB]) : (0));
    int charAInt = ((int)(charA));
    int charBInt = ((int)(charB));
    bool charLt = (charAInt < charBInt);
    bool charGt = (charAInt > charBInt);
    bool charEq = (charAInt == charBInt);
    bool lenLt = (lenA < lenB);
    bool lenGt = (lenA > lenB);
    int cmpResult = ((charLt) ? (-1) : (((charGt) ? (1) : (((lenLt) ? (-1) : (((lenGt) ? (1) : (0))))))));
    int _out = ((bothValid) ? (cmpResult) : (0));
    result[gid] = _out;
    unsigned int outValidBit = ((bothValid) ? (1u) : (0u));
    unsigned int outBit = (outValidBit << bitIdx);
    atomicOr(&outputValid[wordIdx], outBit);
  }
}
