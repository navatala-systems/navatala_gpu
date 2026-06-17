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

__kernel void navatala_vector_search_init_search_state_f32(__global const uint* global_entry, __global const uint* n_queries, __global const uint* visited_size, __global uint* entry_points, __global uint* visited) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint entry = global_entry[0];
  uint nq = n_queries[0];
  uint vs = visited_size[0];
  if ((gid < nq)) {
    entry_points[gid] = entry;
  }
  uint total_visited = (nq * vs);
  if ((gid < total_visited)) {
    uint q_idx = (gid / vs);
    uint v_idx = (gid % vs);
    uint flat_idx = ((q_idx * vs) + v_idx);
    visited[flat_idx] = (uint)(0u);
  }
}
