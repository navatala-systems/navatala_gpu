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

__kernel void navatala_vector_search_insert_vertex_f32(__global const uint* vertex_id, __global const uint* candidates, __global const uint* n_candidates, __global const uint* max_degree, __global uint* graph, __global uint* graph_degrees) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_global_id(0))));
  uint vid = vertex_id[0];
  uint nc = n_candidates[0];
  uint md = max_degree[0];
  uint n_neighbors = (((nc < md)) ? (nc) : (md));
  if ((tid < n_neighbors)) {
    uint neighbor = candidates[tid];
    uint graph_idx = ((vid * md) + tid);
    graph[graph_idx] = neighbor;
  }
  if ((tid == (uint)(0u))) {
    graph_degrees[vid] = n_neighbors;
  }
}
