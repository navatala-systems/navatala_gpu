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
extern "C" __global__ void navatala_vector_search_rerank_with_exact_distances_f32(const float* queries, const float* dataset, const unsigned int* candidate_ids, const unsigned int* n_queries, const unsigned int* k, const unsigned int* dim, float* exact_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int k_val = k[0];
  unsigned int d = dim[0];
  unsigned int total = (nq * k_val);
  if ((gid < total)) {
    unsigned int query_id = (gid / k_val);
    unsigned int k_idx = (gid % k_val);
    unsigned int cand_id = candidate_ids[gid];
    float dist = __uint_as_float(0x00000000u);
    for (int i = 0; i < (int)(d); ++i) {
      unsigned int q_idx = ((query_id * d) + i);
      unsigned int c_idx = ((cand_id * d) + i);
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
