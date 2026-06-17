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

__kernel void navatala_vector_search_extract_search_results_f32(__global const uint* candidates, __global const float* candidate_dists, __global const uint* n_candidates, __global const uint* n_queries, __global const uint* k, __global const uint* search_list_size, __global uint* result_ids, __global float* result_dists) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint k_val = k[0];
  uint ls = search_list_size[0];
  uint query_id = (gid / k_val);
  uint k_idx = (gid % k_val);
  if ((query_id < nq)) {
    uint nc = n_candidates[query_id];
    uint src_idx = ((query_id * ls) + k_idx);
    uint dst_idx = ((query_id * k_val) + k_idx);
    if ((k_idx < nc)) {
      uint cand_id = candidates[src_idx];
      float cand_dist = candidate_dists[src_idx];
      result_ids[dst_idx] = cand_id;
      result_dists[dst_idx] = cand_dist;
    } else {
      result_ids[dst_idx] = (uint)(4294967295u);
      result_dists[dst_idx] = as_float(0x7e967699u);
    }
  }
}
