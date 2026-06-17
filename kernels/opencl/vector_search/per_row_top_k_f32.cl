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

__kernel void navatala_vector_search_per_row_top_k_f32(__global const float* data, __global const uint* n_rows, __global const uint* row_size, __global const uint* k, __global float* top_values, __global uint* top_indices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nr = n_rows[0];
  uint rs = row_size[0];
  uint kVal = k[0];
  uint total = (nr * kVal);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row_idx = (gid / kVal);
    uint k_idx = (gid % kVal);
    uint row_start = (row_idx * rs);
    float best_val = as_float(0x7e967699u);
    uint best_idx = (uint)(0u);
    uint count_smaller = (uint)(0u);
    for (int i = 0; i < (int)(rs); ++i) {
      uint data_idx = (row_start + i);
      float val = data[data_idx];
      uint smaller_count = (uint)(0u);
      for (int j = 0; j < (int)(rs); ++j) {
        uint other_idx = (row_start + j);
        float other_val = data[other_idx];
        bool is_smaller = (other_val < val);
        bool is_equal_earlier = ((other_val == val) && (j < i));
        bool counts = (is_smaller || is_equal_earlier);
        uint old_cnt = smaller_count;
        uint incr = ((counts) ? ((uint)(1u)) : ((uint)(0u)));
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
