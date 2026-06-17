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
extern "C" __global__ void navatala_vector_search_compact_graph_edges(unsigned int* graph, const unsigned int* n_vertices, const unsigned int* max_degree, const unsigned int* invalid_id, unsigned int* degrees) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vertices[0];
  unsigned int md = max_degree[0];
  unsigned int inv = invalid_id[0];
  if ((vid < nv)) {
    unsigned int write_pos = 0u;
    for (int i = 0; i < (int)(md); ++i) {
      unsigned int read_idx = ((vid * md) + i);
      unsigned int neighbor = graph[read_idx];
      if ((neighbor != inv)) {
        unsigned int wp = write_pos;
        unsigned int write_idx = ((vid * md) + wp);
        graph[write_idx] = neighbor;
        write_pos = (wp + 1u);
      }
    }
    unsigned int final_wp = write_pos;
    for (int j = 0; j < (int)(md); ++j) {
      if ((j >= final_wp)) {
        unsigned int fill_idx = ((vid * md) + j);
        graph[fill_idx] = inv;
      }
    }
    degrees[vid] = final_wp;
  }
}
