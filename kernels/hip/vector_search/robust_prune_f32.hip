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
extern "C" __global__ void navatala_vector_search_robust_prune_f32(const unsigned int* vertex, const unsigned int* candidates, const float* distances, const float* dataset, const unsigned int* n_candidates, const unsigned int* dim, const unsigned int* max_degree, const unsigned int* alpha, unsigned int* selected, unsigned int* n_selected) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nc = n_candidates[0];
  unsigned int md = max_degree[0];
  if (tid == 0u) {
    unsigned int limit = (((nc < md)) ? (nc) : (md));
    for (int i = 0; i < (int)(limit); ++i) {
      unsigned int cand = candidates[i];
      selected[i] = cand;
    }
    n_selected[0u] = limit;
  }
}
