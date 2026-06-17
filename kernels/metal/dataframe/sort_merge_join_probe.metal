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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_dataframe_sort_merge_join_probe(device const int* leftKeys [[buffer(0)]], device const uint* leftValidity [[buffer(1)]], device const uint* leftCount [[buffer(2)]], device const int* rightKeys [[buffer(3)]], device const uint* rightValidity [[buffer(4)]], device const uint* rightCount [[buffer(5)]], device int* leftMatches [[buffer(6)]], device int* rightMatches [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nLeft = leftCount[0u];
  uint nRight = rightCount[0u];
  bool inBounds = (gid < nLeft);
  if (inBounds) {
    uint wordIdx = (gid / 32u);
    uint bitIdx = (gid % 32u);
    uint validWord = leftValidity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & 1u);
    bool isLeftValid = (validBit == 1u);
    int leftKey = leftKeys[gid];
    uint lo0 = 0u;
    uint hi0 = nRight;
    uint mid1 = ((lo0 + hi0) / 2u);
    int midKey1 = (((mid1 < nRight)) ? (rightKeys[mid1]) : (0));
    bool goLeft1 = (leftKey < midKey1);
    uint lo1 = ((goLeft1) ? (lo0) : ((mid1 + 1u)));
    uint hi1 = ((goLeft1) ? (mid1) : (hi0));
    uint mid2 = ((lo1 + hi1) / 2u);
    int midKey2 = (((mid2 < nRight)) ? (rightKeys[mid2]) : (0));
    bool goLeft2 = (leftKey < midKey2);
    uint lo2 = ((goLeft2) ? (lo1) : ((mid2 + 1u)));
    uint hi2 = ((goLeft2) ? (mid2) : (hi1));
    uint mid3 = ((lo2 + hi2) / 2u);
    int midKey3 = (((mid3 < nRight)) ? (rightKeys[mid3]) : (0));
    bool goLeft3 = (leftKey < midKey3);
    uint lo3 = ((goLeft3) ? (lo2) : ((mid3 + 1u)));
    uint hi3 = ((goLeft3) ? (mid3) : (hi2));
    uint mid4 = ((lo3 + hi3) / 2u);
    int midKey4 = (((mid4 < nRight)) ? (rightKeys[mid4]) : (0));
    bool goLeft4 = (leftKey < midKey4);
    uint lo4 = ((goLeft4) ? (lo3) : ((mid4 + 1u)));
    uint hi4 = ((goLeft4) ? (mid4) : (hi3));
    uint mid5 = ((lo4 + hi4) / 2u);
    int midKey5 = (((mid5 < nRight)) ? (rightKeys[mid5]) : (0));
    bool goLeft5 = (leftKey < midKey5);
    uint lo5 = ((goLeft5) ? (lo4) : ((mid5 + 1u)));
    uint hi5 = ((goLeft5) ? (mid5) : (hi4));
    uint mid6 = ((lo5 + hi5) / 2u);
    int midKey6 = (((mid6 < nRight)) ? (rightKeys[mid6]) : (0));
    bool goLeft6 = (leftKey < midKey6);
    uint lo6 = ((goLeft6) ? (lo5) : ((mid6 + 1u)));
    uint hi6 = ((goLeft6) ? (mid6) : (hi5));
    uint mid7 = ((lo6 + hi6) / 2u);
    int midKey7 = (((mid7 < nRight)) ? (rightKeys[mid7]) : (0));
    bool goLeft7 = (leftKey < midKey7);
    uint lo7 = ((goLeft7) ? (lo6) : ((mid7 + 1u)));
    uint hi7 = ((goLeft7) ? (mid7) : (hi6));
    uint mid8 = ((lo7 + hi7) / 2u);
    int midKey8 = (((mid8 < nRight)) ? (rightKeys[mid8]) : (0));
    bool goLeft8 = (leftKey < midKey8);
    uint lo8 = ((goLeft8) ? (lo7) : ((mid8 + 1u)));
    uint hi8 = ((goLeft8) ? (mid8) : (hi7));
    uint foundIdx = lo8;
    bool foundInRange = (foundIdx < nRight);
    int foundKey = ((foundInRange) ? (rightKeys[foundIdx]) : (0));
    bool isMatch = (foundInRange && (leftKey == foundKey));
    uint rightWordIdx = (foundIdx / 32u);
    uint rightBitIdx = (foundIdx % 32u);
    uint rightValidWord = ((foundInRange) ? (rightValidity[rightWordIdx]) : (0u));
    uint rightValidBit = ((rightValidWord >> rightBitIdx) & 1u);
    bool isRightValid = (rightValidBit == 1u);
    bool hasValidMatch = (isLeftValid && (isRightValid && isMatch));
    int leftOut = ((int)(gid));
    leftMatches[gid] = leftOut;
    int rightOut = ((hasValidMatch) ? (((int)(foundIdx))) : (-1));
    rightMatches[gid] = rightOut;
  }
}
