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

kernel void navatala_vector_search_random_init_neighbors_f32(device const uint* n_vectors [[buffer(0)]], device const uint* max_degree [[buffer(1)]], device const uint* init_degree [[buffer(2)]], device const uint* seed [[buffer(3)]], device uint* graph [[buffer(4)]], device uint* graph_degrees [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint vid = ((uint)(int(__gid.x)));
  uint nv = n_vectors[0];
  uint md = max_degree[0];
  uint id = init_degree[0];
  uint s = seed[0];
  if ((vid < nv)) {
    uint rng = (vid + s);
    for (int i = 0; i < (int)(id); ++i) {
      uint old_rng = rng;
      uint temp1 = (old_rng * 1103515245u);
      uint temp2 = (temp1 + 12345u);
      uint new_rng = (temp2 & 2147483647u);
      rng = new_rng;
      uint neighbor_raw = (new_rng % nv);
      uint neighbor = (((neighbor_raw == vid)) ? (((neighbor_raw + 1u) % nv)) : (neighbor_raw));
      uint graph_idx = ((vid * md) + i);
      graph[graph_idx] = neighbor;
    }
    graph_degrees[vid] = id;
  }
}
