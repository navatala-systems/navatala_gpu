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
extern "C" __global__ void navatala_vector_search_search_layer_greedy_f32(const float* queries, const float* dataset, const unsigned int* graph, const unsigned int* graph_degrees, const unsigned int* entry_points, const unsigned int* n_queries, const unsigned int* dim, const unsigned int* ef, const unsigned int* max_degree, unsigned int* candidates, float* candidate_distances, unsigned int* n_found) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int query_id = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int d = dim[0];
  if ((query_id < nq)) {
    unsigned int entry = entry_points[query_id];
    float best_dist = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int q_idx = ((query_id * d) + k);
      unsigned int e_idx = ((entry * d) + k);
      float q_val = queries[q_idx];
      float e_val = dataset[e_idx];
      float diff = (q_val - e_val);
      float sq = (diff * diff);
      float old_bd = best_dist;
      best_dist = (old_bd + sq);
    }
    unsigned int ef_val = ef[0];
    unsigned int cand_idx = (query_id * ef_val);
    candidates[cand_idx] = entry;
    float init_dist = best_dist;
    candidate_distances[cand_idx] = init_dist;
    n_found[query_id] = 1u;
  }
}
