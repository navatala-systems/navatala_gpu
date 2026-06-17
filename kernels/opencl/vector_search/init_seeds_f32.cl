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

__kernel void navatala_vector_search_init_seeds_f32(__global const float* queries, __global const float* dataset, __global const uint* seed_indices, __global const uint* n_queries, __global const uint* n_seeds, __global const uint* dim, __global const uint* itopk_size, __global uint* candidates, __global float* distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint ns = n_seeds[0];
  uint d = dim[0];
  uint itopk = itopk_size[0];
  uint total = (nq * ns);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint query_idx = (gid / ns);
    uint seed_local = (gid % ns);
    uint node_idx = seed_indices[seed_local];
    float dist_acc = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint q_idx = ((query_idx * d) + k);
      float q_val = queries[q_idx];
      uint d_idx = ((node_idx * d) + k);
      float d_val = dataset[d_idx];
      float diff = (q_val - d_val);
      float diff_sq = (diff * diff);
      float old = dist_acc;
      dist_acc = (old + diff_sq);
    }
    uint out_idx = ((query_idx * itopk) + seed_local);
    candidates[out_idx] = node_idx;
    float final_dist = dist_acc;
    distances[out_idx] = final_dist;
  }
}
