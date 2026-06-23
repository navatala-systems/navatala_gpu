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

kernel void navatala_vector_search_search_layer_greedy_f32(device const float* queries [[buffer(0)]], device const float* dataset [[buffer(1)]], device const uint* graph [[buffer(2)]], device const uint* graph_degrees [[buffer(3)]], device const uint* entry_points [[buffer(4)]], device const uint* n_queries [[buffer(5)]], device const uint* dim [[buffer(6)]], device const uint* ef [[buffer(7)]], device const uint* max_degree [[buffer(8)]], device uint* candidates [[buffer(9)]], device float* candidate_distances [[buffer(10)]], device uint* n_found [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint query_id = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint d = dim[0];
  if (query_id < nq) {
    uint entry = entry_points[query_id];
    float best_dist = as_type<float>(0x00000000u);
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
    n_found[query_id] = 1u;
  }
}
