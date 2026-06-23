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

kernel void navatala_vector_search_reduce_inertia_f32(device const float* min_distances [[buffer(0)]], device const uint* n_points [[buffer(1)]], device float* partial_sums [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint group_id = ((uint)(int(__tgid.x)));
  uint n = n_points[0];
  threadgroup float sdata[256];
  bool inBounds = (gid < n);
  float val = ((inBounds) ? (min_distances[gid]) : (as_type<float>(0x00000000u)));
  sdata[lid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride = 128u;
  for (int __iter = 0; __iter < 8; ++__iter) {
    if (!(stride > 0u)) break;
    uint s = stride;
    bool should_reduce = (lid < s);
    if (should_reduce) {
      uint other_idx = (lid + s);
      float my_val = sdata[lid];
      float other_val = sdata[other_idx];
      float sum = (my_val + other_val);
      sdata[lid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stride = (s / 2u);
  }
  bool is_thread0 = (lid == 0u);
  if (is_thread0) {
    float block_sum = sdata[0u];
    partial_sums[group_id] = block_sum;
  }
}
