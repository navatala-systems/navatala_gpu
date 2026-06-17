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

kernel void navatala_vector_search_compute_graph_distances_f32(device const float* dataset [[buffer(0)]], device const uint* graph [[buffer(1)]], device const uint* n_nodes [[buffer(2)]], device const uint* graph_degree [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* graph_distances [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
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
    float dist_acc = as_type<float>(0x00000000u);
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
