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
extern "C" __global__ void navatala_vector_search_compute_recall_f32(const unsigned int* approx_ids, const unsigned int* ground_truth_ids, const unsigned int* n_queries, const unsigned int* k, float* recall_per_query) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int query_id = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int k_val = k[0];
  if ((query_id < nq)) {
    unsigned int hits = 0u;
    for (int i = 0; i < (int)(k_val); ++i) {
      unsigned int approx_idx = ((query_id * k_val) + i);
      unsigned int approx_id = approx_ids[approx_idx];
      unsigned int found = 0u;
      for (int j = 0; j < (int)(k_val); ++j) {
        unsigned int gt_idx = ((query_id * k_val) + j);
        unsigned int gt_id = ground_truth_ids[gt_idx];
        if ((approx_id == gt_id)) {
          found = 1u;
        }
      }
      unsigned int was_found = found;
      unsigned int old_hits = hits;
      hits = (old_hits + was_found);
    }
    unsigned int total_hits = hits;
    float hits_f32 = ((float)(total_hits));
    float k_f32 = ((float)(k_val));
    float recall = (hits_f32 / k_f32);
    recall_per_query[query_id] = recall;
  }
}
