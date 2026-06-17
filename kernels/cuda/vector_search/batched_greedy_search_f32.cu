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
extern "C" __global__ void navatala_vector_search_batched_greedy_search_f32(const float* queries, const float* dataset, const unsigned int* graph, const unsigned int* graph_degrees, const unsigned int* entry_point, const unsigned int* n_queries, const unsigned int* n_vectors, const unsigned int* dim, const unsigned int* max_degree, const unsigned int* k, unsigned int* result_ids, float* result_dists) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int query_id = ((unsigned int)((int)(blockIdx.x)));
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int d = dim[0];
  unsigned int entry = entry_point[0];
  unsigned int k_val = k[0];
  __shared__ float partial_sums[64];
  if ((query_id < nq)) {
    if ((tid == 0u)) {
      unsigned int base_idx = (query_id * k_val);
      result_ids[base_idx] = entry;
      result_dists[base_idx] = __uint_as_float(0x00000000u);
    }
    if ((tid < k_val)) {
      if ((tid != 0u)) {
        unsigned int out_idx = ((query_id * k_val) + tid);
        result_ids[out_idx] = 4294967295u;
        result_dists[out_idx] = __uint_as_float(0x7e967699u);
      }
    }
  }
}
