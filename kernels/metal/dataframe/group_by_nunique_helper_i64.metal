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

kernel void navatala_dataframe_group_by_nunique_helper_i64(device const long* data [[buffer(0)]], device const uint* validity [[buffer(1)]], device const uint* groupIds [[buffer(2)]], device const uint* count [[buffer(3)]], device uint* hashOutput [[buffer(4)]], device uint* validOutput [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / 32u);
    uint bitIdx = (gid % 32u);
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    long val = data[gid];
    uint groupId = groupIds[gid];
    uint kLow = ((uint)(val));
    uint kHigh = ((uint)((val >> 32)));
    uint c1 = 3432918353u;
    uint c2 = 461845907u;
    uint k1a = (kLow * c1);
    uint k1b = ((k1a << 15u) | (k1a >> 17u));
    uint k1c = (k1b * c2);
    uint k2a = (kHigh * c1);
    uint k2b = ((k2a << 15u) | (k2a >> 17u));
    uint k2c = (k2b * c2);
    uint h1 = (groupId ^ k1c);
    uint h2 = (h1 ^ k2c);
    uint h3 = ((h2 << 13u) | (h2 >> 19u));
    uint h4 = ((h3 * 5u) + 3864292196u);
    uint h5 = (h4 ^ 8u);
    uint h6 = (h5 ^ (h5 >> 16u));
    uint h7 = (h6 * 2246822507u);
    uint h8 = (h7 ^ (h7 >> 13u));
    uint h9 = (h8 * 3266489909u);
    uint hash = (h9 ^ (h9 >> 16u));
    uint outHash = ((isValid) ? (hash) : (0u));
    hashOutput[gid] = outHash;
    uint outBit = (validBit << bitIdx);
    atomic_fetch_or_explicit((device atomic_uint*)(&validOutput[wordIdx]), outBit, memory_order_relaxed);
  }
}
