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
extern "C" __global__ void navatala_vector_search_initialize_random_graph(const unsigned int* n_nodes, const unsigned int* graph_degree, const unsigned int* seed, unsigned int* graph, float* graph_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_nodes[0];
  unsigned int deg = graph_degree[0];
  unsigned int s = seed[0];
  unsigned int total = (nn * deg);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int node_idx = (gid / deg);
    unsigned int slot = (gid % deg);
    unsigned int state0 = ((gid * 1103515245u) + (s + 12345u));
    unsigned int rand_raw = (state0 % nn);
    bool is_self = (rand_raw == node_idx);
    unsigned int alt = ((node_idx + 1u) % nn);
    unsigned int neighbor = ((is_self) ? (alt) : (rand_raw));
    graph[gid] = neighbor;
    graph_distances[gid] = __uint_as_float(0x7f7fc99eu);
  }
}
