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

kernel void navatala_vector_search_reorder_vertices_by_degree(device const uint* degrees [[buffer(0)]], device atomic_uint* prefix_sums [[buffer(1)]], device const uint* n_vertices [[buffer(2)]], device const uint* n_bins [[buffer(3)]], device uint* old_to_new [[buffer(4)]], device uint* new_to_old [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint vid = ((uint)(int(__gid.x)));
  uint nv = n_vertices[0];
  uint nb = n_bins[0];
  if (vid < nv) {
    uint deg = degrees[vid];
    uint bin = (((deg >= nb)) ? ((nb - 1u)) : (deg));
    uint new_id = atomic_fetch_add_explicit(((device atomic_uint*)(&(prefix_sums[bin]))), 1u, memory_order_relaxed);
    old_to_new[vid] = new_id;
    new_to_old[new_id] = vid;
  }
}
