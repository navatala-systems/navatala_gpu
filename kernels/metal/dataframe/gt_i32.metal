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

kernel void navatala_dataframe_gt_i32(device const int* left [[buffer(0)]], device const uint* leftValid [[buffer(1)]], device const int* right [[buffer(2)]], device const uint* rightValid [[buffer(3)]], device const uint* count [[buffer(4)]], device uchar* _output [[buffer(5)]], device uint* outputValid [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    int lval = left[gid];
    int rval = right[gid];
    bool cmp = (lval > rval);
    uchar result = ((cmp) ? (((uchar)(1u))) : (((uchar)(0u))));
    _output[gid] = result;
    uint wordIdx = (gid / 32u);
    uint bitIdx = (gid % 32u);
    uint leftValidWord = leftValid[wordIdx];
    uint leftValidBit = ((leftValidWord >> bitIdx) & 1u);
    uint rightValidWord = rightValid[wordIdx];
    uint rightValidBit = ((rightValidWord >> bitIdx) & 1u);
    uint outValidBit = (leftValidBit & rightValidBit);
    uint outBit = (outValidBit << bitIdx);
    atomic_fetch_or_explicit((device atomic_uint*)(&outputValid[wordIdx]), outBit, memory_order_relaxed);
  }
}
