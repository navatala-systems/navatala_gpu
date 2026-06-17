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

kernel void navatala_vector_search_scan_inverted_list_f32(device const float* queries [[buffer(0)]], device const float* list_data [[buffer(1)]], device const uint* list_indices [[buffer(2)]], device const uint* list_offsets [[buffer(3)]], device const uint* probe_indices [[buffer(4)]], device const uint* n_queries [[buffer(5)]], device const uint* n_probes [[buffer(6)]], device const uint* dim [[buffer(7)]], device const uint* max_list_size [[buffer(8)]], device float* scan_distances [[buffer(9)]], device uint* scan_indices [[buffer(10)]], device uint* scan_counts [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tid = ((uint)(int(__tid.x)));
  uint wgid = ((uint)(int(__tgid.x)));
  uint wg_size = 256u;
  uint nq = n_queries[0];
  uint np = n_probes[0];
  uint d = dim[0];
  uint mls = max_list_size[0];
  uint total_pairs = (nq * np);
  bool valid_wg = (wgid < total_pairs);
  if (valid_wg) {
    uint query_idx = (wgid / np);
    uint probe_idx = (wgid % np);
    uint probe_offset = ((query_idx * np) + probe_idx);
    uint list_idx = probe_indices[probe_offset];
    uint list_start = list_offsets[list_idx];
    uint list_idx_plus1 = (list_idx + 1u);
    uint list_end = list_offsets[list_idx_plus1];
    uint list_size = (list_end - list_start);
    uint query_base = (query_idx * d);
    uint out_base = (((query_idx * np) * mls) + (probe_idx * mls));
    uint iters = ((list_size / wg_size) + 1u);
    for (int iter = 0; iter < (int)(iters); ++iter) {
      uint local_vec_idx = ((iter * wg_size) + tid);
      bool valid_vec = (local_vec_idx < list_size);
      if (valid_vec) {
        uint global_vec_idx = (list_start + local_vec_idx);
        uint vec_base = (global_vec_idx * d);
        float dist_acc = as_type<float>(0x00000000u);
        for (int k = 0; k < (int)(d); ++k) {
          uint q_off = (query_base + k);
          uint v_off = (vec_base + k);
          float q_val = queries[q_off];
          float v_val = list_data[v_off];
          float diff = (q_val - v_val);
          float sq = (diff * diff);
          float old_d = dist_acc;
          dist_acc = (old_d + sq);
        }
        uint orig_idx = list_indices[global_vec_idx];
        uint out_offset = (out_base + local_vec_idx);
        float final_d = dist_acc;
        scan_distances[out_offset] = final_d;
        scan_indices[out_offset] = orig_idx;
      }
    }
    bool is_t0 = (tid == 0u);
    if (is_t0) {
      uint count_offset = ((query_idx * np) + probe_idx);
      scan_counts[count_offset] = list_size;
    }
  }
}
