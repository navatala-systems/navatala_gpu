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

__kernel void navatala_vector_search_batched_greedy_search_f32(__global const float* queries, __global const float* dataset, __global const uint* graph, __global const uint* graph_degrees, __global const uint* entry_point, __global const uint* n_queries, __global const uint* n_vectors, __global const uint* dim, __global const uint* max_degree, __global const uint* k, __global uint* result_ids, __global float* result_dists) {
  int gid0 = (int)get_global_id(0);
  uint query_id = ((uint)((int)(get_group_id(0))));
  uint tid = ((uint)((int)(get_local_id(0))));
  uint nq = n_queries[0];
  uint d = dim[0];
  uint entry = entry_point[0];
  uint k_val = k[0];
  __local float partial_sums[64];
  if ((query_id < nq)) {
    if ((tid == (uint)(0u))) {
      uint base_idx = (query_id * k_val);
      result_ids[base_idx] = entry;
      result_dists[base_idx] = as_float(0x00000000u);
    }
    if ((tid < k_val)) {
      if ((tid != (uint)(0u))) {
        uint out_idx = ((query_id * k_val) + tid);
        result_ids[out_idx] = (uint)(4294967295u);
        result_dists[out_idx] = as_float(0x7e967699u);
      }
    }
  }
}
