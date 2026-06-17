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

__kernel void navatala_vector_search_search_layer_greedy_f32(__global const float* queries, __global const float* dataset, __global const uint* graph, __global const uint* graph_degrees, __global const uint* entry_points, __global const uint* n_queries, __global const uint* dim, __global const uint* ef, __global const uint* max_degree, __global uint* candidates, __global float* candidate_distances, __global uint* n_found) {
  int gid0 = (int)get_global_id(0);
  uint query_id = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint d = dim[0];
  if ((query_id < nq)) {
    uint entry = entry_points[query_id];
    float best_dist = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint q_idx = ((query_id * d) + k);
      uint e_idx = ((entry * d) + k);
      float q_val = queries[q_idx];
      float e_val = dataset[e_idx];
      float diff = (q_val - e_val);
      float sq = (diff * diff);
      float old_bd = best_dist;
      best_dist = (old_bd + sq);
    }
    uint ef_val = ef[0];
    uint cand_idx = (query_id * ef_val);
    candidates[cand_idx] = entry;
    float init_dist = best_dist;
    candidate_distances[cand_idx] = init_dist;
    n_found[query_id] = (uint)(1u);
  }
}
