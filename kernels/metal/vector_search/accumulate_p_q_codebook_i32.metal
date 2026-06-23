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

kernel void navatala_vector_search_accumulate_p_q_codebook_i32(device const float* subvectors [[buffer(0)]], device const uint* assignments [[buffer(1)]], device const uint* n_vectors [[buffer(2)]], device const uint* sub_dim [[buffer(3)]], device atomic_int* codebook_sums_i32 [[buffer(4)]], device atomic_uint* codebook_counts [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint vid = ((uint)(int(__gid.x)));
  uint nv = n_vectors[0];
  uint sd = sub_dim[0];
  float scale = as_type<float>(0x47800000u);
  if (vid < nv) {
    uint cw = assignments[vid];
    for (int k = 0; k < (int)(sd); ++k) {
      uint src_idx = ((vid * sd) + k);
      uint dst_idx = ((cw * sd) + k);
      float val_f32 = subvectors[src_idx];
      float scaled_f32 = (val_f32 * scale);
      int scaled_i32 = ((int)(scaled_f32));
      int _old = atomic_fetch_add_explicit(((device atomic_int*)(&(codebook_sums_i32[dst_idx]))), scaled_i32, memory_order_relaxed);
    }
    uint _cnt = atomic_fetch_add_explicit(((device atomic_uint*)(&(codebook_counts[cw]))), 1u, memory_order_relaxed);
  }
}
