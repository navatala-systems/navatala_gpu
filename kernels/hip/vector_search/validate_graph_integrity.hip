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
extern "C" __global__ void navatala_vector_search_validate_graph_integrity(const unsigned int* graph, const unsigned int* degrees, const unsigned int* n_vertices, const unsigned int* max_degree, unsigned int* error_flags) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vertices[0];
  unsigned int md = max_degree[0];
  if (vid < nv) {
    unsigned int errors = 0u;
    unsigned int deg = degrees[vid];
    for (int i = 0; i < (int)(deg); ++i) {
      unsigned int idx = ((vid * md) + i);
      unsigned int neighbor = graph[idx];
      if (neighbor == vid) {
        unsigned int old_err = errors;
        errors = (old_err | 1u);
      }
      if (neighbor >= nv) {
        unsigned int old_err2 = errors;
        errors = (old_err2 | 2u);
      }
    }
    unsigned int final_errors = errors;
    error_flags[vid] = final_errors;
  }
}
