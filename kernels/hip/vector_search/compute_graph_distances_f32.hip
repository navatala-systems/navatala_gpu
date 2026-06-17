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
extern "C" __global__ void navatala_vector_search_compute_graph_distances_f32(const float* dataset, const unsigned int* graph, const unsigned int* n_nodes, const unsigned int* graph_degree, const unsigned int* dim, float* graph_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_nodes[0];
  unsigned int deg = graph_degree[0];
  unsigned int d = dim[0];
  unsigned int total = (nn * deg);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int node_idx = (gid / deg);
    unsigned int neighbor = graph[gid];
    unsigned int node_base = (node_idx * d);
    unsigned int neighbor_base = (neighbor * d);
    float dist_acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int n_off = (node_base + k);
      unsigned int nb_off = (neighbor_base + k);
      float n_val = dataset[n_off];
      float nb_val = dataset[nb_off];
      float diff = (n_val - nb_val);
      float sq = (diff * diff);
      float old_d = dist_acc;
      dist_acc = (old_d + sq);
    }
    float final_d = dist_acc;
    graph_distances[gid] = final_d;
  }
}
