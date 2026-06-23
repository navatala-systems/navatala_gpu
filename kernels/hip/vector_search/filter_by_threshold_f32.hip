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
extern "C" __global__ void navatala_vector_search_filter_by_threshold_f32(const float* distances, unsigned int* candidate_ids, const unsigned int* n_queries, const unsigned int* k, const float* threshold, const unsigned int* invalid_id, unsigned int* filtered_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int query_id = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int k_val = k[0];
  float thresh = threshold[0];
  unsigned int inv = invalid_id[0];
  if (query_id < nq) {
    unsigned int count = 0u;
    for (int i = 0; i < (int)(k_val); ++i) {
      unsigned int idx = ((query_id * k_val) + i);
      float dist = distances[idx];
      if (dist > thresh) {
        candidate_ids[idx] = inv;
      } else {
        unsigned int old_cnt = count;
        count = (old_cnt + 1u);
      }
    }
    unsigned int final_count = count;
    filtered_counts[query_id] = final_count;
  }
}
