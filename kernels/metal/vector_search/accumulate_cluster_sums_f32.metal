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

kernel void navatala_vector_search_accumulate_cluster_sums_f32(device const float* data [[buffer(0)]], device const uint* labels [[buffer(1)]], device const uint* n_points [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* cluster_sums [[buffer(5)]], device atomic_uint* cluster_counts [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = n_points[0];
  uint d = dim[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint label = labels[gid];
    uint _old_count = atomic_fetch_add_explicit(((device atomic_uint*)(&(cluster_counts[label]))), 1u, memory_order_relaxed);
    for (int j = 0; j < (int)(d); ++j) {
      uint data_idx = ((gid * d) + j);
      float data_val = data[data_idx];
      uint sum_idx = ((label * d) + j);
      atomic_fetch_add_explicit((device atomic_float*)(&cluster_sums[sum_idx]), data_val, memory_order_relaxed);
    }
  }
}
