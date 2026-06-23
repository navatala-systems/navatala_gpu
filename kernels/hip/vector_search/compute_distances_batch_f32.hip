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
extern "C" __global__ void navatala_vector_search_compute_distances_batch_f32(const float* query, const float* candidates, const unsigned int* candidate_ids, const unsigned int* n_candidates, const unsigned int* dim, float* distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n_cand = n_candidates[0];
  unsigned int d = dim[0];
  if (gid < n_cand) {
    unsigned int cand_id = candidate_ids[gid];
    float acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      float q_val = query[k];
      unsigned int c_idx = ((cand_id * d) + k);
      float c_val = candidates[c_idx];
      float diff = (q_val - c_val);
      float sq = (diff * diff);
      float old_acc = acc;
      acc = (old_acc + sq);
    }
    float final_acc = acc;
    distances[gid] = final_acc;
  }
}
