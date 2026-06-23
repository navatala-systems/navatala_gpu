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

kernel void navatala_vector_search_batched_greedy_search_f32(device const float* queries [[buffer(0)]], device const float* dataset [[buffer(1)]], device const uint* graph [[buffer(2)]], device const uint* graph_degrees [[buffer(3)]], device const uint* entry_point [[buffer(4)]], device const uint* n_queries [[buffer(5)]], device const uint* n_vectors [[buffer(6)]], device const uint* dim [[buffer(7)]], device const uint* max_degree [[buffer(8)]], device const uint* k [[buffer(9)]], device uint* result_ids [[buffer(10)]], device float* result_dists [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint query_id = ((uint)(int(__tgid.x)));
  uint tid = ((uint)(int(__tid.x)));
  uint nq = n_queries[0];
  uint d = dim[0];
  uint entry = entry_point[0];
  uint k_val = k[0];
  threadgroup float partial_sums[64];
  if (query_id < nq) {
    if (tid == 0u) {
      uint base_idx = (query_id * k_val);
      result_ids[base_idx] = entry;
      result_dists[base_idx] = as_type<float>(0x00000000u);
    }
    if (tid < k_val) {
      if (tid != 0u) {
        uint out_idx = ((query_id * k_val) + tid);
        result_ids[out_idx] = 4294967295u;
        result_dists[out_idx] = as_type<float>(0x7e967699u);
      }
    }
  }
}
