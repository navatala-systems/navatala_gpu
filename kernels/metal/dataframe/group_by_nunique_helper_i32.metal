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

kernel void navatala_dataframe_group_by_nunique_helper_i32(device const int* data [[buffer(0)]], device const uint* validity [[buffer(1)]], device const uint* groupIds [[buffer(2)]], device const uint* count [[buffer(3)]], device uint* hashOutput [[buffer(4)]], device uint* validOutput [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / 32u);
    uint bitIdx = (gid % 32u);
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    int val = data[gid];
    uint groupId = groupIds[gid];
    uint k = ((uint)(val));
    uint c1 = 3432918353u;
    uint c2 = 461845907u;
    uint k1 = (k * c1);
    uint k2 = ((k1 << 15u) | (k1 >> 17u));
    uint k3 = (k2 * c2);
    uint h1 = (groupId ^ k3);
    uint h2 = ((h1 << 13u) | (h1 >> 19u));
    uint h3 = ((h2 * 5u) + 3864292196u);
    uint h4 = (h3 ^ 4u);
    uint h5 = (h4 ^ (h4 >> 16u));
    uint h6 = (h5 * 2246822507u);
    uint h7 = (h6 ^ (h6 >> 13u));
    uint h8 = (h7 * 3266489909u);
    uint hash = (h8 ^ (h8 >> 16u));
    uint outHash = ((isValid) ? (hash) : (0u));
    hashOutput[gid] = outHash;
    uint outBit = (validBit << bitIdx);
    atomic_fetch_or_explicit((device atomic_uint*)(&validOutput[wordIdx]), outBit, memory_order_relaxed);
  }
}
