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

__kernel void navatala_vector_search_rerank_with_exact_distances_f32(__global const float* queries, __global const float* dataset, __global const uint* candidate_ids, __global const uint* n_queries, __global const uint* k, __global const uint* dim, __global float* exact_distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint k_val = k[0];
  uint d = dim[0];
  uint total = (nq * k_val);
  if (gid < total) {
    uint query_id = (gid / k_val);
    uint k_idx = (gid % k_val);
    uint cand_id = candidate_ids[gid];
    float dist = as_float(0x00000000u);
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
