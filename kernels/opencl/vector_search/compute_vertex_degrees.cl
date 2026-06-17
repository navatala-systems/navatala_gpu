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

__kernel void navatala_vector_search_compute_vertex_degrees(__global const uint* graph, __global const uint* n_vertices, __global const uint* max_degree, __global const uint* invalid_id, __global uint* degrees) {
  int gid0 = (int)get_global_id(0);
  uint vid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vertices[0];
  uint md = max_degree[0];
  uint inv = invalid_id[0];
  if ((vid < nv)) {
    uint count = (uint)(0u);
    for (int i = 0; i < (int)(md); ++i) {
      uint idx = ((vid * md) + i);
      uint neighbor = graph[idx];
      if ((neighbor != inv)) {
        uint old_cnt = count;
        count = (old_cnt + (uint)(1u));
      }
    }
    uint final_count = count;
    degrees[vid] = final_count;
  }
}
