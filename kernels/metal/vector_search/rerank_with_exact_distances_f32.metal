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

kernel void navatala_vector_search_rerank_with_exact_distances_f32(device const float* queries [[buffer(0)]], device const float* dataset [[buffer(1)]], device const uint* candidate_ids [[buffer(2)]], device const uint* n_queries [[buffer(3)]], device const uint* k [[buffer(4)]], device const uint* dim [[buffer(5)]], device float* exact_distances [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint k_val = k[0];
  uint d = dim[0];
  uint total = (nq * k_val);
  if (gid < total) {
    uint query_id = (gid / k_val);
    uint k_idx = (gid % k_val);
    uint cand_id = candidate_ids[gid];
    float dist = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(d); ++i) {
      uint q_idx = ((query_id * d) + i);
      uint c_idx = ((cand_id * d) + i);
      float q_val = queries[q_idx];
      float c_val = dataset[c_idx];
      float diff = (q_val - c_val);
      float sq = (diff * diff);
      float old_dist = dist;
      dist = (old_dist + sq);
    }
    float final_dist = dist;
    exact_distances[gid] = final_dist;
  }
}
