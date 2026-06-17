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

__kernel void navatala_vector_search_initialize_random_graph(__global const uint* n_nodes, __global const uint* graph_degree, __global const uint* seed, __global uint* graph, __global float* graph_distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nn = n_nodes[0];
  uint deg = graph_degree[0];
  uint s = seed[0];
  uint total = (nn * deg);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint node_idx = (gid / deg);
    uint slot = (gid % deg);
    uint state0 = ((gid * (uint)(1103515245u)) + (s + (uint)(12345u)));
    uint rand_raw = (state0 % nn);
    bool is_self = (rand_raw == node_idx);
    uint alt = ((node_idx + (uint)(1u)) % nn);
    uint neighbor = ((is_self) ? (alt) : (rand_raw));
    graph[gid] = neighbor;
    graph_distances[gid] = as_float(0x7f7fc99eu);
  }
}
