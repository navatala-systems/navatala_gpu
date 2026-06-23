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

kernel void navatala_sparse_compute_residual_norm_f32(device const float* residual [[buffer(0)]], device const uint* n [[buffer(1)]], device float* normSq [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  threadgroup float sdata[256];
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint i = gid;
  uint nVal = n[0u];
  bool inBounds = (i < nVal);
  float ri = ((inBounds) ? (residual[i]) : (as_type<float>(0x00000000u)));
  float riSq = (ri * ri);
  sdata[lid] = riSq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int s = 0; s < (int)(128u); ++s) {
    uint sU32 = ((uint)(s));
    if (lid < sU32) {
      float other = sdata[(lid + sU32)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    atomic_fetch_add_explicit((device atomic_float*)(&normSq[0u]), sdata[0u], memory_order_relaxed);
  }
}
