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
extern "C" __global__ void navatala_vector_search_mark_visited_batch(const unsigned int* node_ids, const unsigned int* query_ids, const unsigned int* n_nodes, const unsigned int* visited_stride, unsigned int* visited) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_nodes[0];
  unsigned int stride = visited_stride[0];
  if (tid < nn) {
    unsigned int node = node_ids[tid];
    unsigned int query = query_ids[tid];
    unsigned int word_idx = (node / 32u);
    unsigned int bit_idx = (node % 32u);
    unsigned int bit_mask = (1u << bit_idx);
    unsigned int flat_idx = ((query * stride) + word_idx);
    atomicOr(&visited[flat_idx], bit_mask);
  }
}
