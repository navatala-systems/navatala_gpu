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

kernel void navatala_vector_search_filter_by_threshold_f32(device const float* distances [[buffer(0)]], device uint* candidate_ids [[buffer(1)]], device const uint* n_queries [[buffer(2)]], device const uint* k [[buffer(3)]], device const float* threshold [[buffer(4)]], device const uint* invalid_id [[buffer(5)]], device uint* filtered_counts [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint query_id = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint k_val = k[0];
  float thresh = threshold[0];
  uint inv = invalid_id[0];
  if (query_id < nq) {
    uint count = 0u;
    for (int i = 0; i < (int)(k_val); ++i) {
      uint idx = ((query_id * k_val) + i);
      float dist = distances[idx];
      if (dist > thresh) {
        candidate_ids[idx] = inv;
      } else {
        uint old_cnt = count;
        count = (old_cnt + 1u);
      }
    }
    uint final_count = count;
    filtered_counts[query_id] = final_count;
  }
}
