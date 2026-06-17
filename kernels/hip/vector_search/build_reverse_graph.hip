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
extern "C" __global__ void navatala_vector_search_build_reverse_graph(const unsigned int* graph, const unsigned int* n_nodes, const unsigned int* graph_degree, unsigned int* reverse_graph, unsigned int* reverse_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_nodes[0];
  unsigned int deg = graph_degree[0];
  unsigned int total = (nn * deg);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int source = (gid / deg);
    unsigned int target = graph[gid];
    unsigned int rev_slot = atomicAdd(&(reverse_counts[target]), 1u);
    bool has_space = (rev_slot < deg);
    if (has_space) {
      unsigned int rev_idx = ((target * deg) + rev_slot);
      reverse_graph[rev_idx] = source;
    }
  }
}
