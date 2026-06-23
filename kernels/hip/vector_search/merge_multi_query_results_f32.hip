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
extern "C" __global__ void navatala_vector_search_merge_multi_query_results_f32(const unsigned int* layer_results, const float* layer_distances, const unsigned int* n_queries, const unsigned int* k, const unsigned int* n_layers, unsigned int* final_ids, float* final_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int k_val = k[0];
  unsigned int query_id = (gid / k_val);
  unsigned int k_idx = (gid % k_val);
  if (query_id < nq) {
    unsigned int flat_idx = ((query_id * k_val) + k_idx);
    unsigned int result_id = layer_results[flat_idx];
    float result_dist = layer_distances[flat_idx];
    final_ids[flat_idx] = result_id;
    final_distances[flat_idx] = result_dist;
  }
}
