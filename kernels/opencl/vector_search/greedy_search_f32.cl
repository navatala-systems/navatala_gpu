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

__kernel void navatala_vector_search_greedy_search_f32(__global const float* queries, __global const float* dataset, __global const uint* graph, __global const uint* graph_degrees, __global const uint* entry_point, __global const uint* n_queries, __global const uint* dim, __global const uint* max_degree, __global const uint* search_list_size, __global uint* candidates, __global float* candidate_dists, __global uint* n_candidates) {
  int gid0 = (int)get_global_id(0);
  uint query_id = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint d = dim[0];
  uint entry = entry_point[0];
  uint ls = search_list_size[0];
  if ((query_id < nq)) {
    float entry_dist = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint q_idx = ((query_id * d) + k);
      uint e_idx = ((entry * d) + k);
      float q_val = queries[q_idx];
      float e_val = dataset[e_idx];
      float diff = (q_val - e_val);
      float sq = (diff * diff);
      float old_ed = entry_dist;
      entry_dist = (old_ed + sq);
    }
    uint cand_base = (query_id * ls);
    candidates[cand_base] = entry;
    float ed = entry_dist;
    candidate_dists[cand_base] = ed;
    n_candidates[query_id] = (uint)(1u);
  }
}
