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
extern "C" __global__ void navatala_vector_search_random_init_neighbors_f32(const unsigned int* n_vectors, const unsigned int* max_degree, const unsigned int* init_degree, const unsigned int* seed, unsigned int* graph, unsigned int* graph_degrees) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int md = max_degree[0];
  unsigned int id = init_degree[0];
  unsigned int s = seed[0];
  if (vid < nv) {
    unsigned int rng = (vid + s);
    for (int i = 0; i < (int)(id); ++i) {
      unsigned int old_rng = rng;
      unsigned int temp1 = (old_rng * 1103515245u);
      unsigned int temp2 = (temp1 + 12345u);
      unsigned int new_rng = (temp2 & 2147483647u);
      rng = new_rng;
      unsigned int neighbor_raw = (new_rng % nv);
      unsigned int neighbor = (((neighbor_raw == vid)) ? (((neighbor_raw + 1u) % nv)) : (neighbor_raw));
      unsigned int graph_idx = ((vid * md) + i);
      graph[graph_idx] = neighbor;
    }
    graph_degrees[vid] = id;
  }
}
