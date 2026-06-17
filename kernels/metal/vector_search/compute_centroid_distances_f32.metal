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

kernel void navatala_vector_search_compute_centroid_distances_f32(device const float* queries [[buffer(0)]], device const float* centroids [[buffer(1)]], device const uint* n_queries [[buffer(2)]], device const uint* n_lists [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* centroid_distances [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint nl = n_lists[0];
  uint d = dim[0];
  uint total_pairs = (nq * nl);
  bool inBounds = (gid < total_pairs);
  if (inBounds) {
    uint query_idx = (gid / nl);
    uint list_idx = (gid % nl);
    uint query_base = (query_idx * d);
    uint centroid_base = (list_idx * d);
    float acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint q_offset = (query_base + k);
      uint c_offset = (centroid_base + k);
      float q_val = queries[q_offset];
      float c_val = centroids[c_offset];
      float diff = (q_val - c_val);
      float sq = (diff * diff);
      float old_acc = acc;
      acc = (old_acc + sq);
    }
    float final_acc = acc;
    centroid_distances[gid] = final_acc;
  }
}
