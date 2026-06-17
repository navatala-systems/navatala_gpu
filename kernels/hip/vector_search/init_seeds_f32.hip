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
extern "C" __global__ void navatala_vector_search_init_seeds_f32(const float* queries, const float* dataset, const unsigned int* seed_indices, const unsigned int* n_queries, const unsigned int* n_seeds, const unsigned int* dim, const unsigned int* itopk_size, unsigned int* candidates, float* distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int ns = n_seeds[0];
  unsigned int d = dim[0];
  unsigned int itopk = itopk_size[0];
  unsigned int total = (nq * ns);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int query_idx = (gid / ns);
    unsigned int seed_local = (gid % ns);
    unsigned int node_idx = seed_indices[seed_local];
    float dist_acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int q_idx = ((query_idx * d) + k);
      float q_val = queries[q_idx];
      unsigned int d_idx = ((node_idx * d) + k);
      float d_val = dataset[d_idx];
      float diff = (q_val - d_val);
      float diff_sq = (diff * diff);
      float old = dist_acc;
      dist_acc = (old + diff_sq);
    }
    unsigned int out_idx = ((query_idx * itopk) + seed_local);
    candidates[out_idx] = node_idx;
    float final_dist = dist_acc;
    distances[out_idx] = final_dist;
  }
}
