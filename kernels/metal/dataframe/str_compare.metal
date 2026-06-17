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

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_dataframe_str_compare(device const int* offsetsA [[buffer(0)]], device const int* offsetsB [[buffer(1)]], device const char* charsA [[buffer(2)]], device const char* charsB [[buffer(3)]], device const uint* validityA [[buffer(4)]], device const uint* validityB [[buffer(5)]], device const uint* count [[buffer(6)]], device int* result [[buffer(7)]], device uint* outputValid [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / 32u);
    uint bitIdx = (gid % 32u);
    uint validWordA = validityA[wordIdx];
    uint validBitA = ((validWordA >> bitIdx) & 1u);
    bool isValidA = (validBitA == 1u);
    uint validWordB = validityB[wordIdx];
    uint validBitB = ((validWordB >> bitIdx) & 1u);
    bool isValidB = (validBitB == 1u);
    bool bothValid = (isValidA && isValidB);
    int startA = offsetsA[gid];
    uint nextIdx = (gid + 1u);
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
    uint outValidBit = ((bothValid) ? (1u) : (0u));
    uint outBit = (outValidBit << bitIdx);
    atomic_fetch_or_explicit((device atomic_uint*)(&outputValid[wordIdx]), outBit, memory_order_relaxed);
  }
}
