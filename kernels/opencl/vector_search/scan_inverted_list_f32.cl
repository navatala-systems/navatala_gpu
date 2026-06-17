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

__kernel void navatala_vector_search_scan_inverted_list_f32(__global const float* queries, __global const float* list_data, __global const uint* list_indices, __global const uint* list_offsets, __global const uint* probe_indices, __global const uint* n_queries, __global const uint* n_probes, __global const uint* dim, __global const uint* max_list_size, __global float* scan_distances, __global uint* scan_indices, __global uint* scan_counts) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_local_id(0))));
  uint wgid = ((uint)((int)(get_group_id(0))));
  uint wg_size = (uint)(256u);
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
    uint list_idx_plus1 = (list_idx + (uint)(1u));
    uint list_end = list_offsets[list_idx_plus1];
    uint list_size = (list_end - list_start);
    uint query_base = (query_idx * d);
    uint out_base = (((query_idx * np) * mls) + (probe_idx * mls));
    uint iters = ((list_size / wg_size) + (uint)(1u));
    for (int iter = 0; iter < (int)(iters); ++iter) {
      uint local_vec_idx = ((iter * wg_size) + tid);
      bool valid_vec = (local_vec_idx < list_size);
      if (valid_vec) {
        uint global_vec_idx = (list_start + local_vec_idx);
        uint vec_base = (global_vec_idx * d);
        float dist_acc = as_float(0x00000000u);
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
    bool is_t0 = (tid == (uint)(0u));
    if (is_t0) {
      uint count_offset = ((query_idx * np) + probe_idx);
      scan_counts[count_offset] = list_size;
    }
  }
}
