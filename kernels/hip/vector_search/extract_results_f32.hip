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
extern "C" __global__ void navatala_vector_search_extract_results_f32(const unsigned int* sorted_candidates, const float* sorted_distances, const unsigned int* n_queries, const unsigned int* k, const unsigned int* itopk_size, unsigned int* result_indices, float* result_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int kVal = k[0];
  unsigned int itopk = itopk_size[0];
  unsigned int total = (nq * kVal);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int query_idx = (gid / kVal);
    unsigned int k_idx = (gid % kVal);
    unsigned int src_idx = ((query_idx * itopk) + k_idx);
    unsigned int idx_val = sorted_candidates[src_idx];
    float dist_val = sorted_distances[src_idx];
    result_indices[gid] = idx_val;
    result_distances[gid] = dist_val;
  }
}
