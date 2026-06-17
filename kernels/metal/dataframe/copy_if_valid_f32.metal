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

kernel void navatala_dataframe_copy_if_valid_f32(device const float* src [[buffer(0)]], device const uint* srcValid [[buffer(1)]], device const uint* dstValid [[buffer(2)]], device const uint* count [[buffer(3)]], device float* dst [[buffer(4)]], device uint* outValid [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / 32u);
    uint bitIdx = (gid % 32u);
    uint srcValidWord = srcValid[wordIdx];
    uint srcValidBit = ((srcValidWord >> bitIdx) & 1u);
    bool srcIsValid = (srcValidBit == 1u);
    uint dstValidWord = dstValid[wordIdx];
    uint dstValidBit = ((dstValidWord >> bitIdx) & 1u);
    bool dstIsValid = (dstValidBit == 1u);
    bool bothValid = (srcIsValid && dstIsValid);
    if (bothValid) {
      float val = src[gid];
      dst[gid] = val;
      uint outBit = (1u << bitIdx);
      atomic_fetch_or_explicit((device atomic_uint*)(&outValid[wordIdx]), outBit, memory_order_relaxed);
    }
  }
}
