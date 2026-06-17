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

__kernel void navatala_vector_search_extract_results_f32(__global const uint* sorted_candidates, __global const float* sorted_distances, __global const uint* n_queries, __global const uint* k, __global const uint* itopk_size, __global uint* result_indices, __global float* result_distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint kVal = k[0];
  uint itopk = itopk_size[0];
  uint total = (nq * kVal);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint query_idx = (gid / kVal);
    uint k_idx = (gid % kVal);
    uint src_idx = ((query_idx * itopk) + k_idx);
    uint idx_val = sorted_candidates[src_idx];
    float dist_val = sorted_distances[src_idx];
    result_indices[gid] = idx_val;
    result_distances[gid] = dist_val;
  }
}
