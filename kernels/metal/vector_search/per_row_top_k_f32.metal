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

kernel void navatala_vector_search_per_row_top_k_f32(device const float* data [[buffer(0)]], device const uint* n_rows [[buffer(1)]], device const uint* row_size [[buffer(2)]], device const uint* k [[buffer(3)]], device float* top_values [[buffer(4)]], device uint* top_indices [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nr = n_rows[0];
  uint rs = row_size[0];
  uint kVal = k[0];
  uint total = (nr * kVal);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row_idx = (gid / kVal);
    uint k_idx = (gid % kVal);
    uint row_start = (row_idx * rs);
    float best_val = as_type<float>(0x7e967699u);
    uint best_idx = 0u;
    uint count_smaller = 0u;
    for (int i = 0; i < (int)(rs); ++i) {
      uint data_idx = (row_start + i);
      float val = data[data_idx];
      uint smaller_count = 0u;
      for (int j = 0; j < (int)(rs); ++j) {
        uint other_idx = (row_start + j);
        float other_val = data[other_idx];
        bool is_smaller = (other_val < val);
        bool is_equal_earlier = ((other_val == val) && (j < i));
        bool counts = (is_smaller || is_equal_earlier);
        uint old_cnt = smaller_count;
        uint incr = ((counts) ? (1u) : (0u));
        smaller_count = (old_cnt + incr);
      }
      uint cnt = smaller_count;
      bool is_kth = (cnt == k_idx);
      if (is_kth) {
        best_val = val;
        best_idx = i;
      }
    }
    uint out_idx = ((row_idx * kVal) + k_idx);
    float final_val = best_val;
    uint final_idx = best_idx;
    top_values[out_idx] = final_val;
    top_indices[out_idx] = final_idx;
  }
}
