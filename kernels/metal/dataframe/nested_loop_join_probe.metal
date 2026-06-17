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

kernel void navatala_dataframe_nested_loop_join_probe(device const int* leftKeys [[buffer(0)]], device const int* rightKeys [[buffer(1)]], device const uint* leftCount [[buffer(2)]], device const uint* rightCount [[buffer(3)]], device const uint* maxMatches [[buffer(4)]], device uint* matchCount [[buffer(5)]], device int* matchLeft [[buffer(6)]], device int* matchRight [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lCount = leftCount[0u];
  uint rCount = rightCount[0u];
  uint maxM = maxMatches[0u];
  bool inBounds = (gid < lCount);
  if (inBounds) {
    int leftKey = leftKeys[gid];
    for (int rightIdx = 0; rightIdx < (int)(rCount); ++rightIdx) {
      int rightKey = rightKeys[rightIdx];
      bool isMatch = (leftKey == rightKey);
      if (isMatch) {
        atomic_fetch_add_explicit((device atomic_uint*)(&matchCount[0u]), 1u, memory_order_relaxed);
        uint outPos = matchCount[0u];
        uint outPosAdj = (outPos - 1u);
        bool canStore = (outPosAdj < maxM);
        if (canStore) {
          int leftIdxI32 = ((int)(gid));
          int rightIdxI32 = ((int)(rightIdx));
          matchLeft[outPosAdj] = leftIdxI32;
          matchRight[outPosAdj] = rightIdxI32;
        }
      }
    }
  }
}
