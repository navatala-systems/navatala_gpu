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
extern "C" __global__ void navatala_vector_search_init_search_state_f32(const unsigned int* global_entry, const unsigned int* n_queries, const unsigned int* visited_size, unsigned int* entry_points, unsigned int* visited) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int entry = global_entry[0];
  unsigned int nq = n_queries[0];
  unsigned int vs = visited_size[0];
  if ((gid < nq)) {
    entry_points[gid] = entry;
  }
  unsigned int total_visited = (nq * vs);
  if ((gid < total_visited)) {
    unsigned int q_idx = (gid / vs);
    unsigned int v_idx = (gid % vs);
    unsigned int flat_idx = ((q_idx * vs) + v_idx);
    visited[flat_idx] = 0u;
  }
}
