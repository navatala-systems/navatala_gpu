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

kernel void navatala_dataframe_coalesce_f32(device const float* col1 [[buffer(0)]], device const uint* col1Valid [[buffer(1)]], device const float* col2 [[buffer(2)]], device const uint* col2Valid [[buffer(3)]], device const uint* count [[buffer(4)]], device float* _output [[buffer(5)]], device uint* outputValid [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / 32u);
    uint bitIdx = (gid % 32u);
    uint v1Word = col1Valid[wordIdx];
    uint v1Bit = ((v1Word >> bitIdx) & 1u);
    bool col1IsValid = (v1Bit == 1u);
    uint v2Word = col2Valid[wordIdx];
    uint v2Bit = ((v2Word >> bitIdx) & 1u);
    bool col2IsValid = (v2Bit == 1u);
    float val1 = col1[gid];
    float val2 = col2[gid];
    float outVal = ((col1IsValid) ? (val1) : (val2));
    _output[gid] = outVal;
    bool eitherValid = (col1IsValid || col2IsValid);
    uint outBit = ((eitherValid) ? ((1u << bitIdx)) : (0u));
    atomic_fetch_or_explicit((device atomic_uint*)(&outputValid[wordIdx]), outBit, memory_order_relaxed);
  }
}
