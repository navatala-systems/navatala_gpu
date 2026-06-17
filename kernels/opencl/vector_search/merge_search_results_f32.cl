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

__kernel void navatala_vector_search_merge_search_results_f32(__global const uint* ids_a, __global const float* dists_a, __global const uint* ids_b, __global const float* dists_b, __global const uint* n_queries, __global const uint* k_a, __global const uint* k_b, __global const uint* k_out, __global uint* merged_ids, __global float* merged_dists) {
  int gid0 = (int)get_global_id(0);
  uint query_id = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint ka = k_a[0];
  uint kb = k_b[0];
  uint ko = k_out[0];
  if ((query_id < nq)) {
    uint ptr_a = (uint)(0u);
    uint ptr_b = (uint)(0u);
    uint out_idx = (uint)(0u);
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
      float dist_a = ((a_valid) ? (dist_a_raw) : (as_float(0x7e967699u)));
      float dist_b = ((b_valid) ? (dist_b_raw) : (as_float(0x7e967699u)));
      bool choose_a = (dist_a <= dist_b);
      uint chosen_id = ((choose_a) ? (ids_a[idx_a]) : (ids_b[idx_b]));
      float chosen_dist = ((choose_a) ? (dist_a) : (dist_b));
      uint out_flat = ((query_id * ko) + oi);
      merged_ids[out_flat] = chosen_id;
      merged_dists[out_flat] = chosen_dist;
      if (choose_a) {
        ptr_a = (pa + (uint)(1u));
      } else {
        ptr_b = (pb + (uint)(1u));
      }
      out_idx = (oi + (uint)(1u));
    }
  }
}
