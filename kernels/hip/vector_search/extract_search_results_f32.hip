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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_extract_search_results_f32(const unsigned int* candidates, const float* candidate_dists, const unsigned int* n_candidates, const unsigned int* n_queries, const unsigned int* k, const unsigned int* search_list_size, unsigned int* result_ids, float* result_dists) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int k_val = k[0];
  unsigned int ls = search_list_size[0];
  unsigned int query_id = (gid / k_val);
  unsigned int k_idx = (gid % k_val);
  if (query_id < nq) {
    unsigned int nc = n_candidates[query_id];
    unsigned int src_idx = ((query_id * ls) + k_idx);
    unsigned int dst_idx = ((query_id * k_val) + k_idx);
    if (k_idx < nc) {
      unsigned int cand_id = candidates[src_idx];
      float cand_dist = candidate_dists[src_idx];
      result_ids[dst_idx] = cand_id;
      result_dists[dst_idx] = cand_dist;
    } else {
      result_ids[dst_idx] = 4294967295u;
      result_dists[dst_idx] = __uint_as_float(0x7e967699u);
    }
  }
}
