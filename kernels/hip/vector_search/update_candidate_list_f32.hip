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
extern "C" __global__ void navatala_vector_search_update_candidate_list_f32(unsigned int* candidates, float* distances, const unsigned int* new_candidates, const float* new_distances, const unsigned int* n_current, const unsigned int* n_new, const unsigned int* ef, unsigned int* n_merged) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n_curr = n_current[0];
  unsigned int n_new_val = n_new[0];
  unsigned int ef_val = ef[0];
  __shared__ unsigned int merge_ids[512];
  __shared__ float merge_dists[512];
  if (tid < n_curr) {
    unsigned int cid = candidates[tid];
    float cdist = distances[tid];
    merge_ids[tid] = cid;
    merge_dists[tid] = cdist;
  }
  __syncthreads();
  if (tid < n_new_val) {
    unsigned int offset = (n_curr + tid);
    unsigned int nid = new_candidates[tid];
    float ndist = new_distances[tid];
    merge_ids[offset] = nid;
    merge_dists[offset] = ndist;
  }
  __syncthreads();
  if (tid == 0u) {
    unsigned int total = (n_curr + n_new_val);
    unsigned int final_n = (((total < ef_val)) ? (total) : (ef_val));
    n_merged[0u] = final_n;
  }
}
