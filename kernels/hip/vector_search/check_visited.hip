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
extern "C" __global__ void navatala_vector_search_check_visited(const unsigned int* candidates, unsigned int* visited, const unsigned int* n, const unsigned int* hash_size, unsigned int* is_new) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int num = n[0];
  unsigned int hs = hash_size[0];
  bool inBounds = (gid < num);
  if (inBounds) {
    unsigned int node = candidates[gid];
    unsigned int hash_idx = (node % hs);
    unsigned int old_val = atomicCAS((int*)(&(visited[hash_idx])), (int)(0u), (int)((node + 1u)));
    bool was_empty = (old_val == 0u);
    unsigned int result = ((was_empty) ? (1u) : (0u));
    is_new[gid] = result;
  }
}
