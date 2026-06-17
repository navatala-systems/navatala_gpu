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

kernel void navatala_dataframe_str_contains_char(device const int* offsets [[buffer(0)]], device const char* chars [[buffer(1)]], device const uint* validity [[buffer(2)]], device const char* targetChar [[buffer(3)]], device const uint* count [[buffer(4)]], device uint* _output [[buffer(5)]], device uint* outputValid [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  char target = targetChar[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / 32u);
    uint bitIdx = (gid % 32u);
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    int startOffset = offsets[gid];
    uint nextIdx = (gid + 1u);
    int endOffset = offsets[nextIdx];
    uint startU = ((uint)(startOffset));
    uint endU = ((uint)(endOffset));
    uint len = (endU - startU);
    bool hasLen1 = (len >= 1u);
    uint idx0 = startU;
    char c0 = chars[idx0];
    bool match0 = (hasLen1 && (c0 == target));
    bool hasLen2 = (len >= 2u);
    uint idx1 = (startU + 1u);
    char c1 = chars[idx1];
    bool match1 = (hasLen2 && (c1 == target));
    bool hasLen3 = (len >= 3u);
    uint idx2 = (startU + 2u);
    char c2 = chars[idx2];
    bool match2 = (hasLen3 && (c2 == target));
    bool hasLen4 = (len >= 4u);
    uint idx3 = (startU + 3u);
    char c3 = chars[idx3];
    bool match3 = (hasLen4 && (c3 == target));
    bool anyMatch = (match0 || (match1 || (match2 || match3)));
    uint result = (((isValid && anyMatch)) ? (1u) : (0u));
    _output[gid] = result;
    uint outBit = (validBit << bitIdx);
    atomic_fetch_or_explicit((device atomic_uint*)(&outputValid[wordIdx]), outBit, memory_order_relaxed);
  }
}
