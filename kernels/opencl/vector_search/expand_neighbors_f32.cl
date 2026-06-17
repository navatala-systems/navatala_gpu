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

__kernel void navatala_vector_search_expand_neighbors_f32(__global const float* queries, __global const float* dataset, __global const uint* graph, __global const uint* top_candidates, __global const uint* n_queries, __global const uint* search_width, __global const uint* graph_degree, __global const uint* dim, __global uint* new_candidates, __global float* new_distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint sw = search_width[0];
  uint gd = graph_degree[0];
  uint d = dim[0];
  uint neighbors_per_query = (sw * gd);
  uint total = (nq * neighbors_per_query);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint query_idx = (gid / neighbors_per_query);
    uint local_idx = (gid % neighbors_per_query);
    uint candidate_idx = (local_idx / gd);
    uint neighbor_idx = (local_idx % gd);
    uint tc_idx = ((query_idx * sw) + candidate_idx);
    uint parent_node = top_candidates[tc_idx];
    uint graph_idx = ((parent_node * gd) + neighbor_idx);
    uint neighbor_node = graph[graph_idx];
    float dist_acc = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint q_idx = ((query_idx * d) + k);
      float q_val = queries[q_idx];
      uint d_idx = ((neighbor_node * d) + k);
      float d_val = dataset[d_idx];
      float diff = (q_val - d_val);
      float diff_sq = (diff * diff);
      float old = dist_acc;
      dist_acc = (old + diff_sq);
    }
    uint out_idx = ((query_idx * neighbors_per_query) + local_idx);
    new_candidates[out_idx] = neighbor_node;
    float final_dist = dist_acc;
    new_distances[out_idx] = final_dist;
  }
}
