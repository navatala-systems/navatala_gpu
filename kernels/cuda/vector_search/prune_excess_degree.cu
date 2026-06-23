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
extern "C" __global__ void navatala_vector_search_prune_excess_degree(unsigned int* graph, const float* neighbor_distances, unsigned int* degrees, const unsigned int* n_vertices, const unsigned int* current_max_degree, const unsigned int* target_max_degree, const unsigned int* invalid_id) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vertices[0];
  unsigned int cmd = current_max_degree[0];
  unsigned int tmd = target_max_degree[0];
  unsigned int inv = invalid_id[0];
  if (vid < nv) {
    unsigned int deg = degrees[vid];
    if (deg > tmd) {
      for (int i = 0; i < (int)(cmd); ++i) {
        if (i >= tmd) {
          unsigned int idx = ((vid * cmd) + i);
          graph[idx] = inv;
        }
      }
      degrees[vid] = tmd;
    }
  }
}
