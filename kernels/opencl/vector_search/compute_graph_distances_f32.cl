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

__kernel void navatala_vector_search_compute_graph_distances_f32(__global const float* dataset, __global const uint* graph, __global const uint* n_nodes, __global const uint* graph_degree, __global const uint* dim, __global float* graph_distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nn = n_nodes[0];
  uint deg = graph_degree[0];
  uint d = dim[0];
  uint total = (nn * deg);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint node_idx = (gid / deg);
    uint neighbor = graph[gid];
    uint node_base = (node_idx * d);
    uint neighbor_base = (neighbor * d);
    float dist_acc = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint n_off = (node_base + k);
      uint nb_off = (neighbor_base + k);
      float n_val = dataset[n_off];
      float nb_val = dataset[nb_off];
      float diff = (n_val - nb_val);
      float sq = (diff * diff);
      float old_d = dist_acc;
      dist_acc = (old_d + sq);
    }
    float final_d = dist_acc;
    graph_distances[gid] = final_d;
  }
}
