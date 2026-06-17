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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_compact_pruned_graph(const unsigned int* pruned_graph, const float* pruned_distances, const unsigned int* n_nodes, const unsigned int* graph_degree, unsigned int* compact_graph, float* compact_distances, unsigned int* degree_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int node_idx = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_nodes[0];
  unsigned int deg = graph_degree[0];
  bool inBounds = (node_idx < nn);
  if (inBounds) {
    unsigned int base = (node_idx * deg);
    unsigned int write_pos = 0u;
    for (int i = 0; i < (int)(deg); ++i) {
      unsigned int read_idx = (base + i);
      unsigned int neighbor = pruned_graph[read_idx];
      bool is_valid = (neighbor != 4294967295u);
      if (is_valid) {
        unsigned int wp = write_pos;
        unsigned int write_idx = (base + wp);
        float dist = pruned_distances[read_idx];
        compact_graph[write_idx] = neighbor;
        compact_distances[write_idx] = dist;
        write_pos = (wp + 1u);
      }
    }
    unsigned int final_count = write_pos;
    degree_counts[node_idx] = final_count;
  }
}
