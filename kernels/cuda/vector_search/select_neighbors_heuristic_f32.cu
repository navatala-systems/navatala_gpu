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
extern "C" __global__ void navatala_vector_search_select_neighbors_heuristic_f32(const float* query, const unsigned int* candidates, const float* distances, const float* dataset, const unsigned int* n_candidates, const unsigned int* dim, const unsigned int* m, unsigned int* selected, unsigned int* n_selected) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n_cand = n_candidates[0];
  unsigned int m_val = m[0];
  __shared__ unsigned int selected_ids[64];
  __shared__ unsigned int n_sel[1];
  if (tid == 0u) {
    n_sel[0u] = 0u;
  }
  __syncthreads();
  if (tid < n_cand) {
    unsigned int cand_id = candidates[tid];
    selected[tid] = cand_id;
  }
  __syncthreads();
  if (tid == 0u) {
    unsigned int final_count = (((n_cand < m_val)) ? (n_cand) : (m_val));
    n_selected[0u] = final_count;
  }
}
