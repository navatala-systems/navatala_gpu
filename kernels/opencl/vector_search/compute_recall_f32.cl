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

__kernel void navatala_vector_search_compute_recall_f32(__global const uint* approx_ids, __global const uint* ground_truth_ids, __global const uint* n_queries, __global const uint* k, __global float* recall_per_query) {
  int gid0 = (int)get_global_id(0);
  uint query_id = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint k_val = k[0];
  if (query_id < nq) {
    uint hits = (uint)(0u);
    for (int i = 0; i < (int)(k_val); ++i) {
      uint approx_idx = ((query_id * k_val) + i);
      uint approx_id = approx_ids[approx_idx];
      uint found = (uint)(0u);
      for (int j = 0; j < (int)(k_val); ++j) {
        uint gt_idx = ((query_id * k_val) + j);
        uint gt_id = ground_truth_ids[gt_idx];
        if (approx_id == gt_id) {
          found = (uint)(1u);
        }
      }
      uint was_found = found;
      uint old_hits = hits;
      hits = (old_hits + was_found);
    }
    uint total_hits = hits;
    float hits_f32 = ((float)(total_hits));
    float k_f32 = ((float)(k_val));
    float recall = (hits_f32 / k_f32);
    recall_per_query[query_id] = recall;
  }
}
