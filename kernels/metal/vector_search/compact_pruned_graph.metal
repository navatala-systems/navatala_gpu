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

kernel void navatala_vector_search_compact_pruned_graph(device const uint* pruned_graph [[buffer(0)]], device const float* pruned_distances [[buffer(1)]], device const uint* n_nodes [[buffer(2)]], device const uint* graph_degree [[buffer(3)]], device uint* compact_graph [[buffer(4)]], device float* compact_distances [[buffer(5)]], device uint* degree_counts [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint node_idx = ((uint)(int(__gid.x)));
  uint nn = n_nodes[0];
  uint deg = graph_degree[0];
  bool inBounds = (node_idx < nn);
  if (inBounds) {
    uint base = (node_idx * deg);
    uint write_pos = 0u;
    for (int i = 0; i < (int)(deg); ++i) {
      uint read_idx = (base + i);
      uint neighbor = pruned_graph[read_idx];
      bool is_valid = (neighbor != 4294967295u);
      if (is_valid) {
        uint wp = write_pos;
        uint write_idx = (base + wp);
        float dist = pruned_distances[read_idx];
        compact_graph[write_idx] = neighbor;
        compact_distances[write_idx] = dist;
        write_pos = (wp + 1u);
      }
    }
    uint final_count = write_pos;
    degree_counts[node_idx] = final_count;
  }
}
