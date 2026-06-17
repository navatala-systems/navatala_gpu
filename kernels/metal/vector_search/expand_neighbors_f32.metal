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

kernel void navatala_vector_search_expand_neighbors_f32(device const float* queries [[buffer(0)]], device const float* dataset [[buffer(1)]], device const uint* graph [[buffer(2)]], device const uint* top_candidates [[buffer(3)]], device const uint* n_queries [[buffer(4)]], device const uint* search_width [[buffer(5)]], device const uint* graph_degree [[buffer(6)]], device const uint* dim [[buffer(7)]], device uint* new_candidates [[buffer(8)]], device float* new_distances [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
    float dist_acc = as_type<float>(0x00000000u);
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
