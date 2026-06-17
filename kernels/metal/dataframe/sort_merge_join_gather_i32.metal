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

kernel void navatala_dataframe_sort_merge_join_gather_i32(device const int* leftData [[buffer(0)]], device const int* rightData [[buffer(1)]], device const int* leftIndices [[buffer(2)]], device const int* rightIndices [[buffer(3)]], device const uint* outputCount [[buffer(4)]], device int* leftOutput [[buffer(5)]], device int* rightOutput [[buffer(6)]], device uint* outputValid [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = outputCount[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    int leftIdx = leftIndices[gid];
    int rightIdx = rightIndices[gid];
    bool hasLeftMatch = (leftIdx >= 0);
    bool hasRightMatch = (rightIdx >= 0);
    bool hasBothMatches = (hasLeftMatch && hasRightMatch);
    uint leftIdxU = ((uint)(leftIdx));
    uint rightIdxU = ((uint)(rightIdx));
    int leftVal = ((hasLeftMatch) ? (leftData[leftIdxU]) : (0));
    int rightVal = ((hasRightMatch) ? (rightData[rightIdxU]) : (0));
    leftOutput[gid] = leftVal;
    rightOutput[gid] = rightVal;
    uint wordIdx = (gid / 32u);
    uint bitIdx = (gid % 32u);
    uint validBit = ((hasBothMatches) ? ((1u << bitIdx)) : (0u));
    atomic_fetch_or_explicit((device atomic_uint*)(&outputValid[wordIdx]), validBit, memory_order_relaxed);
  }
}
