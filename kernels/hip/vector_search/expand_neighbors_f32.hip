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
extern "C" __global__ void navatala_vector_search_expand_neighbors_f32(const float* queries, const float* dataset, const unsigned int* graph, const unsigned int* top_candidates, const unsigned int* n_queries, const unsigned int* search_width, const unsigned int* graph_degree, const unsigned int* dim, unsigned int* new_candidates, float* new_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int sw = search_width[0];
  unsigned int gd = graph_degree[0];
  unsigned int d = dim[0];
  unsigned int neighbors_per_query = (sw * gd);
  unsigned int total = (nq * neighbors_per_query);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int query_idx = (gid / neighbors_per_query);
    unsigned int local_idx = (gid % neighbors_per_query);
    unsigned int candidate_idx = (local_idx / gd);
    unsigned int neighbor_idx = (local_idx % gd);
    unsigned int tc_idx = ((query_idx * sw) + candidate_idx);
    unsigned int parent_node = top_candidates[tc_idx];
    unsigned int graph_idx = ((parent_node * gd) + neighbor_idx);
    unsigned int neighbor_node = graph[graph_idx];
    float dist_acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int q_idx = ((query_idx * d) + k);
      float q_val = queries[q_idx];
      unsigned int d_idx = ((neighbor_node * d) + k);
      float d_val = dataset[d_idx];
      float diff = (q_val - d_val);
      float diff_sq = (diff * diff);
      float old = dist_acc;
      dist_acc = (old + diff_sq);
    }
    unsigned int out_idx = ((query_idx * neighbors_per_query) + local_idx);
    new_candidates[out_idx] = neighbor_node;
    float final_dist = dist_acc;
    new_distances[out_idx] = final_dist;
  }
}
