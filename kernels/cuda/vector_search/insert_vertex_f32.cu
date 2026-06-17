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
extern "C" __global__ void navatala_vector_search_insert_vertex_f32(const unsigned int* vertex_id, const unsigned int* candidates, const unsigned int* n_candidates, const unsigned int* max_degree, unsigned int* graph, unsigned int* graph_degrees) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int vid = vertex_id[0];
  unsigned int nc = n_candidates[0];
  unsigned int md = max_degree[0];
  unsigned int n_neighbors = (((nc < md)) ? (nc) : (md));
  if ((tid < n_neighbors)) {
    unsigned int neighbor = candidates[tid];
    unsigned int graph_idx = ((vid * md) + tid);
    graph[graph_idx] = neighbor;
  }
  if ((tid == 0u)) {
    graph_degrees[vid] = n_neighbors;
  }
}
