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

__kernel void navatala_vector_search_filter_by_threshold_f32(__global const float* distances, __global uint* candidate_ids, __global const uint* n_queries, __global const uint* k, __global const float* threshold, __global const uint* invalid_id, __global uint* filtered_counts) {
  int gid0 = (int)get_global_id(0);
  uint query_id = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint k_val = k[0];
  float thresh = threshold[0];
  uint inv = invalid_id[0];
  if (query_id < nq) {
    uint count = (uint)(0u);
    for (int i = 0; i < (int)(k_val); ++i) {
      uint idx = ((query_id * k_val) + i);
      float dist = distances[idx];
      if (dist > thresh) {
        candidate_ids[idx] = inv;
      } else {
        uint old_cnt = count;
        count = (old_cnt + (uint)(1u));
      }
    }
    uint final_count = count;
    filtered_counts[query_id] = final_count;
  }
}
