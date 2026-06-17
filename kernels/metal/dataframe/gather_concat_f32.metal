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

kernel void navatala_dataframe_gather_concat_f32(device const float* src [[buffer(0)]], device const uint* srcValid [[buffer(1)]], device const uint* srcLen [[buffer(2)]], device const uint* destOffset [[buffer(3)]], device float* dest [[buffer(4)]], device uint* destValid [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = srcLen[0u];
  uint offset = destOffset[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float val = src[gid];
    uint destIdx = (offset + gid);
    dest[destIdx] = val;
    uint srcWordIdx = (gid / 32u);
    uint srcBitIdx = (gid % 32u);
    uint srcValidWord = srcValid[srcWordIdx];
    uint srcValidBit = ((srcValidWord >> srcBitIdx) & 1u);
    uint destWordIdx = (destIdx / 32u);
    uint destBitIdx = (destIdx % 32u);
    uint outBit = (srcValidBit << destBitIdx);
    atomic_fetch_or_explicit((device atomic_uint*)(&destValid[destWordIdx]), outBit, memory_order_relaxed);
  }
}
