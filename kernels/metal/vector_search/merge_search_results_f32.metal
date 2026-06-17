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

kernel void navatala_vector_search_merge_search_results_f32(device const uint* ids_a [[buffer(0)]], device const float* dists_a [[buffer(1)]], device const uint* ids_b [[buffer(2)]], device const float* dists_b [[buffer(3)]], device const uint* n_queries [[buffer(4)]], device const uint* k_a [[buffer(5)]], device const uint* k_b [[buffer(6)]], device const uint* k_out [[buffer(7)]], device uint* merged_ids [[buffer(8)]], device float* merged_dists [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint query_id = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint ka = k_a[0];
  uint kb = k_b[0];
  uint ko = k_out[0];
  if ((query_id < nq)) {
    uint ptr_a = 0u;
    uint ptr_b = 0u;
    uint out_idx = 0u;
    for (int _iter = 0; _iter < (int)(ko); ++_iter) {
      uint pa = ptr_a;
      uint pb = ptr_b;
      uint oi = out_idx;
      bool a_valid = (pa < ka);
      bool b_valid = (pb < kb);
      uint idx_a = ((query_id * ka) + pa);
      uint idx_b = ((query_id * kb) + pb);
      float dist_a_raw = dists_a[idx_a];
      float dist_b_raw = dists_b[idx_b];
      float dist_a = ((a_valid) ? (dist_a_raw) : (as_type<float>(0x7e967699u)));
      float dist_b = ((b_valid) ? (dist_b_raw) : (as_type<float>(0x7e967699u)));
      bool choose_a = (dist_a <= dist_b);
      uint chosen_id = ((choose_a) ? (ids_a[idx_a]) : (ids_b[idx_b]));
      float chosen_dist = ((choose_a) ? (dist_a) : (dist_b));
      uint out_flat = ((query_id * ko) + oi);
      merged_ids[out_flat] = chosen_id;
      merged_dists[out_flat] = chosen_dist;
      if (choose_a) {
        ptr_a = (pa + 1u);
      } else {
        ptr_b = (pb + 1u);
      }
      out_idx = (oi + 1u);
    }
  }
}
