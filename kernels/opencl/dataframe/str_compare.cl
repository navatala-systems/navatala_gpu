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

__kernel void navatala_dataframe_str_compare(__global const int* offsetsA, __global const int* offsetsB, __global const char* charsA, __global const char* charsB, __global const uint* validityA, __global const uint* validityB, __global const uint* count, __global int* result, __global uint* outputValid) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / (uint)(32u));
    uint bitIdx = (gid % (uint)(32u));
    uint validWordA = validityA[wordIdx];
    uint validBitA = ((validWordA >> bitIdx) & (uint)(1u));
    bool isValidA = (validBitA == (uint)(1u));
    uint validWordB = validityB[wordIdx];
    uint validBitB = ((validWordB >> bitIdx) & (uint)(1u));
    bool isValidB = (validBitB == (uint)(1u));
    bool bothValid = (isValidA && isValidB);
    int startA = offsetsA[gid];
    uint nextIdx = (gid + (uint)(1u));
    int endA = offsetsA[nextIdx];
    int lenA = (endA - startA);
    int startB = offsetsB[gid];
    int endB = offsetsB[nextIdx];
    int lenB = (endB - startB);
    bool hasCharsA = (lenA > 0);
    bool hasCharsB = (lenB > 0);
    uint idxA = ((uint)(startA));
    uint idxB = ((uint)(startB));
    char charA = ((hasCharsA) ? (charsA[idxA]) : (0));
    char charB = ((hasCharsB) ? (charsB[idxB]) : (0));
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
    uint outValidBit = ((bothValid) ? ((uint)(1u)) : ((uint)(0u)));
    uint outBit = (outValidBit << bitIdx);
    atomic_or(&outputValid[wordIdx], outBit);
  }
}
