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

__kernel void navatala_vector_search_compact_pruned_graph(__global const uint* pruned_graph, __global const float* pruned_distances, __global const uint* n_nodes, __global const uint* graph_degree, __global uint* compact_graph, __global float* compact_distances, __global uint* degree_counts) {
  int gid0 = (int)get_global_id(0);
  uint node_idx = ((uint)((int)(get_global_id(0))));
  uint nn = n_nodes[0];
  uint deg = graph_degree[0];
  bool inBounds = (node_idx < nn);
  if (inBounds) {
    uint base = (node_idx * deg);
    uint write_pos = (uint)(0u);
    for (int i = 0; i < (int)(deg); ++i) {
      uint read_idx = (base + i);
      uint neighbor = pruned_graph[read_idx];
      bool is_valid = (neighbor != (uint)(4294967295u));
      if (is_valid) {
        uint wp = write_pos;
        uint write_idx = (base + wp);
        float dist = pruned_distances[read_idx];
        compact_graph[write_idx] = neighbor;
        compact_distances[write_idx] = dist;
        write_pos = (wp + (uint)(1u));
      }
    }
    uint final_count = write_pos;
    degree_counts[node_idx] = final_count;
  }
}
