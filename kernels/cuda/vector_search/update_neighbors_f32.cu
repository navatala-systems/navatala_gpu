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
extern "C" __global__ void navatala_vector_search_update_neighbors_f32(const unsigned int* new_vertex, const unsigned int* new_neighbors, const unsigned int* n_new_neighbors, const unsigned int* max_degree, unsigned int* graph, unsigned int* graph_degrees) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = new_vertex[0];
  unsigned int nn = n_new_neighbors[0];
  unsigned int md = max_degree[0];
  if (tid < nn) {
    unsigned int neighbor = new_neighbors[tid];
    unsigned int old_degree = atomicAdd(&(graph_degrees[neighbor]), 1u);
    if (old_degree < md) {
      unsigned int graph_idx = ((neighbor * md) + old_degree);
      graph[graph_idx] = nv;
    } else {
      unsigned int _ignore = atomicAdd(&(graph_degrees[neighbor]), -(1u));
    }
  }
}
