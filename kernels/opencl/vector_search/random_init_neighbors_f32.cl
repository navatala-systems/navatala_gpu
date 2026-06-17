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

__kernel void navatala_vector_search_random_init_neighbors_f32(__global const uint* n_vectors, __global const uint* max_degree, __global const uint* init_degree, __global const uint* seed, __global uint* graph, __global uint* graph_degrees) {
  int gid0 = (int)get_global_id(0);
  uint vid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vectors[0];
  uint md = max_degree[0];
  uint id = init_degree[0];
  uint s = seed[0];
  if ((vid < nv)) {
    uint rng = (vid + s);
    for (int i = 0; i < (int)(id); ++i) {
      uint old_rng = rng;
      uint temp1 = (old_rng * (uint)(1103515245u));
      uint temp2 = (temp1 + (uint)(12345u));
      uint new_rng = (temp2 & (uint)(2147483647u));
      rng = new_rng;
      uint neighbor_raw = (new_rng % nv);
      uint neighbor = (((neighbor_raw == vid)) ? (((neighbor_raw + (uint)(1u)) % nv)) : (neighbor_raw));
      uint graph_idx = ((vid * md) + i);
      graph[graph_idx] = neighbor;
    }
    graph_degrees[vid] = id;
  }
}
