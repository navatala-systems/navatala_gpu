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

kernel void navatala_cfd_mg_coarse_matvec(device const int* edgeU [[buffer(0)]], device const int* edgeV [[buffer(1)]], device const float* edgeCf [[buffer(2)]], device const float* diag [[buffer(3)]], device const float* x [[buffer(4)]], device float* outAx [[buffer(5)]], device const int* mgCounts [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int total = (((int)(mgCounts[1])) + ((int)(mgCounts[2])));
  if ((int(__gid.x) >= total)) {
    return;
  } else {
    if ((int(__gid.x) < ((int)(mgCounts[1])))) {
      atomic_fetch_add_explicit((device atomic_float*)(&outAx[int(__gid.x)]), (diag[int(__gid.x)] * x[int(__gid.x)]), memory_order_relaxed);
    } else {
      int e = (int(__gid.x) - ((int)(mgCounts[1])));
      int u = edgeU[e];
      int v = edgeV[e];
      float cf = edgeCf[e];
      atomic_fetch_add_explicit((device atomic_float*)(&outAx[u]), (cf * x[v]), memory_order_relaxed);
      atomic_fetch_add_explicit((device atomic_float*)(&outAx[v]), (cf * x[u]), memory_order_relaxed);
    }
  }
}
