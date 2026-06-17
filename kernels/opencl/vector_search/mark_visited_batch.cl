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

__kernel void navatala_vector_search_mark_visited_batch(__global const uint* node_ids, __global const uint* query_ids, __global const uint* n_nodes, __global const uint* visited_stride, __global uint* visited) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_global_id(0))));
  uint nn = n_nodes[0];
  uint stride = visited_stride[0];
  if ((tid < nn)) {
    uint node = node_ids[tid];
    uint query = query_ids[tid];
    uint word_idx = (node / (uint)(32u));
    uint bit_idx = (node % (uint)(32u));
    uint bit_mask = ((uint)(1u) << bit_idx);
    uint flat_idx = ((query * stride) + word_idx);
    atomic_or(&visited[flat_idx], bit_mask);
  }
}
