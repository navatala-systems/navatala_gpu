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

kernel void navatala_dataframe_str_split_by_char(device const int* offsets [[buffer(0)]], device const char* chars [[buffer(1)]], device const uint* validity [[buffer(2)]], device const char* delimChar [[buffer(3)]], device const uint* count [[buffer(4)]], device int* outputOffsets [[buffer(5)]], device uint* outputValid [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  char delim = delimChar[0u];
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
    int length = (endOffset - startOffset);
    uint idx0 = ((uint)(startOffset));
    uint idx1 = (idx0 + 1u);
    uint idx2 = (idx0 + 2u);
    uint idx3 = (idx0 + 3u);
    char c0 = (((length >= 1)) ? (chars[idx0]) : (0));
    char c1 = (((length >= 2)) ? (chars[idx1]) : (0));
    char c2 = (((length >= 3)) ? (chars[idx2]) : (0));
    char c3 = (((length >= 4)) ? (chars[idx3]) : (0));
    bool isDelim0 = (c0 == delim);
    bool isDelim1 = (c1 == delim);
    bool isDelim2 = (c2 == delim);
    bool isDelim3 = (c3 == delim);
    int seg0End = ((isDelim0) ? (startOffset) : (((isDelim1) ? ((startOffset + 1)) : (((isDelim2) ? ((startOffset + 2)) : (((isDelim3) ? ((startOffset + 3)) : (endOffset))))))));
    int outOffset = ((isValid) ? (seg0End) : (0));
    outputOffsets[gid] = outOffset;
    uint outBit = (validBit << bitIdx);
    atomic_fetch_or_explicit((device atomic_uint*)(&outputValid[wordIdx]), outBit, memory_order_relaxed);
  }
}
