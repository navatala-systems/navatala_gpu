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

kernel void navatala_vector_search_extract_search_results_f32(device const uint* candidates [[buffer(0)]], device const float* candidate_dists [[buffer(1)]], device const uint* n_candidates [[buffer(2)]], device const uint* n_queries [[buffer(3)]], device const uint* k [[buffer(4)]], device const uint* search_list_size [[buffer(5)]], device uint* result_ids [[buffer(6)]], device float* result_dists [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint k_val = k[0];
  uint ls = search_list_size[0];
  uint query_id = (gid / k_val);
  uint k_idx = (gid % k_val);
  if (query_id < nq) {
    uint nc = n_candidates[query_id];
    uint src_idx = ((query_id * ls) + k_idx);
    uint dst_idx = ((query_id * k_val) + k_idx);
    if (k_idx < nc) {
      uint cand_id = candidates[src_idx];
      float cand_dist = candidate_dists[src_idx];
      result_ids[dst_idx] = cand_id;
      result_dists[dst_idx] = cand_dist;
    } else {
      result_ids[dst_idx] = 4294967295u;
      result_dists[dst_idx] = as_type<float>(0x7e967699u);
    }
  }
}
