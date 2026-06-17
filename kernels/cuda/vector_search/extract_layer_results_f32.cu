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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_vector_search_extract_layer_results_f32(const unsigned int* candidates, const float* distances, const unsigned int* n_found, const unsigned int* n_queries, const unsigned int* k, unsigned int* result_ids, float* result_dists) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int k_val = k[0];
  unsigned int query_id = (gid / k_val);
  unsigned int neighbor_idx = (gid % k_val);
  if ((query_id < nq)) {
    unsigned int found = n_found[query_id];
    unsigned int src_idx = ((query_id * k_val) + neighbor_idx);
    if ((neighbor_idx < found)) {
      unsigned int cand_id = candidates[src_idx];
      float cand_dist = distances[src_idx];
      result_ids[src_idx] = cand_id;
      result_dists[src_idx] = cand_dist;
    } else {
      result_ids[src_idx] = 4294967295u;
      result_dists[src_idx] = __uint_as_float(0x7e967699u);
    }
  }
}
